"""LSTM、序列数据源/算子及专属可视化分析元件测试。"""

from __future__ import annotations

import asyncio
import json

import torch
from aiohttp.test_utils import TestClient, TestServer

from netpulse.core.compiler import GraphCompiler
from netpulse.core.erc import run_erc
from netpulse.core.ir import EdgeIR, GraphIR, LossConfig, NodeIR, TrainerConfig
from netpulse.core.registry import get_default_registry
from netpulse.core.trainer import Trainer
from netpulse.core.types import Signal
from netpulse.webui.server import create_app


def test_lstm_and_sequence_components_registered_and_erc() -> None:
    reg = get_default_registry()
    for name in (
        "seq_wave_data",
        "image_to_seq",
        "seq_pool",
        "lstm",
        "gate_dynamics",
        "hidden_trajectory",
        "step_energy",
    ):
        assert reg.has(name), f"{name} 未注册"

    graph = GraphIR.from_file("examples/lstm_seq_observatory.json")
    erc = run_erc(graph, reg)
    assert erc.passed, erc.format_report()
    assert erc.node_out_specs["lstm"]["out"].layout == "BLD"
    assert erc.node_out_specs["lstm"]["out"].has_mask is True
    assert erc.node_out_specs["lstm"]["last_h"].shape == (32, 32)
    assert set(erc.node_tap_specs["lstm"].keys()) >= {"gate_i", "gate_f", "gate_o", "cell_state", "hidden_states"}


def test_lstm_mask_awareness_and_e2e_training() -> None:
    reg = get_default_registry()
    graph = GraphIR.from_file("examples/lstm_seq_observatory.json")
    graph.trainer.epochs = 4
    graph.trainer.lr = 0.01

    compiled = GraphCompiler.compile(graph, reg)
    res = Trainer(compiled).fit()
    assert res.steps > 0
    assert res.final_val_acc is not None and res.final_val_acc >= 0.70

    # 验证 3 个专属探针均产生完整快照且不引入可训练参数
    assert sum(p.numel() for n, p in compiled.named_parameters() if "probe_" in n) == 0

    snap_gates = compiled.nodes_mod["probe_gates"].capture_snapshot()
    assert snap_gates is not None
    assert snap_gates["view"] == "gate_dynamics"
    gd = snap_gates["gate_dynamics"]
    assert set(gd["gates"].keys()) == {"i", "f", "o", "c"}
    assert gd["valid_len"] <= gd["total_len"]
    assert len(gd["strip_rows"]) > 0

    snap_traj = compiled.nodes_mod["probe_traj"].capture_snapshot()
    assert snap_traj is not None
    assert snap_traj["view"] == "trajectory"
    traj = snap_traj["trajectory"]
    assert len(traj["points"]) == traj["valid_len"]
    assert 0.0 <= traj["explained_var"] <= 1.01

    snap_energy = compiled.nodes_mod["probe_energy"].capture_snapshot()
    assert snap_energy is not None
    assert snap_energy["view"] == "step_energy"
    se = snap_energy["step_energy"]
    assert len(se["energy"]) == len(se["delta"]) == se["valid_len"]

    # 校验 LSTM 变长 Mask 冻结特性：padding 步不改变 last_h
    lstm_mod = compiled.nodes_mod["lstm"]
    lstm_mod.eval()
    x = torch.randn(2, 10, 4)
    x_mod = x.clone()
    x_mod[0, 5:] += 99.0  # 仅修改第一个样本的 padding 部分
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 5:] = False
    out1 = lstm_mod({"x": Signal(data=x, mask=mask, meta={"layout": "BLD"})})
    out2 = lstm_mod({"x": Signal(data=x_mod, mask=mask, meta={"layout": "BLD"})})
    assert torch.allclose(out1["last_h"].data[0], out2["last_h"].data[0], atol=1e-5)


def test_image_to_seq_with_lstm_and_autobridge() -> None:
    reg = get_default_registry()
    graph = GraphIR(
        name="img_lstm",
        nodes=[
            NodeIR(id="img_src", type="mnist_data", params={"subset_train": 64, "subset_val": 32}),
            NodeIR(id="i2s", type="image_to_seq", params={"mode": "rows"}),
            NodeIR(id="lstm", type="lstm", params={"hidden_size": 16, "in_dim": 16}),
            NodeIR(id="head", type="linear", params={"out_features": 10}),
            NodeIR(id="ce", type="cross_entropy_loss", params={}),
        ],
        edges=[
            EdgeIR(id="e1", src_node="img_src", src_port="x", dst_node="i2s", dst_port="x"),
            EdgeIR(id="e2", src_node="i2s", src_port="out", dst_node="lstm", dst_port="x", bridge="auto"),
            EdgeIR(id="e3", src_node="lstm", src_port="last_h", dst_node="head", dst_port="x"),
            EdgeIR(id="e4", src_node="head", src_port="out", dst_node="ce", dst_port="logits"),
            EdgeIR(id="e5", src_node="img_src", src_port="y", dst_node="ce", dst_port="target"),
        ],
        losses=[LossConfig(node_id="ce", weight=1.0)],
        trainer=TrainerConfig(epochs=1, batch_size=16),
    )
    erc = run_erc(graph, reg)
    assert erc.passed, erc.format_report()
    assert "e2" in erc.auto_bridges  # 28 -> 16 自动矩阵桥

    compiled = GraphCompiler.compile(graph, reg)
    batch = {
        "img_src": {
            "x": Signal(data=torch.randn(4, 1, 28, 28), meta={"layout": "BCHW"}),
            "y": Signal(data=torch.randint(0, 10, (4,)), meta={"layout": "B"}),
        }
    }
    out = compiled(batch, mode="train")
    assert out.outputs["lstm"]["out"].shape == (4, 28, 16)
    assert out.total_loss is not None
    out.total_loss.backward()


def test_webui_ws_streams_lstm_observatory_probes() -> None:
    async def _run() -> None:
        app = create_app(graphs_dir="examples", plugins_dir=None)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            r_graph = await client.get("/api/graph?name=lstm_seq_observatory.json")
            assert r_graph.status == 200
            g_data = await r_graph.json()
            assert g_data["erc"]["passed"] is True

            async with client.ws_connect("/ws") as ws:
                await ws.receive_str(timeout=3)  # hello
                await client.post(
                    "/api/train/start",
                    json={"graph_name": "lstm_seq_observatory.json", "step_delay_ms": 15, "epochs": 1},
                )
                saw = False
                for _ in range(30):
                    msg = json.loads(await ws.receive_str(timeout=5))
                    if msg["type"] == "step" and msg.get("probes"):
                        probes = msg["probes"]
                        assert probes["probe_gates"]["view"] == "gate_dynamics"
                        assert probes["probe_traj"]["view"] == "trajectory"
                        assert probes["probe_energy"]["view"] == "step_energy"
                        saw = True
                        break
                assert saw, "WebSocket 未收到 LSTM 专属探针快照"
                await client.post("/api/train/stop")
        finally:
            await client.close()

    asyncio.run(_run())
