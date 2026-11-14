"""输出监视工具节点（monitor / scalar_watch / heatmap_preview / tensor_peek）测试。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from aiohttp.test_utils import TestClient, TestServer

from netpulse.core.compiler import GraphCompiler
from netpulse.core.erc import run_erc
from netpulse.core.ir import EdgeIR, GraphIR, NodeIR
from netpulse.core.registry import get_default_registry
from netpulse.core.types import Signal
from netpulse.webui.server import create_app


def test_output_components_registered_and_erc_compatible() -> None:
    reg = get_default_registry()
    for name in ("monitor", "scalar_watch", "heatmap_preview", "tensor_peek"):
        assert reg.has(name), f"{name} 未注册"
        cls = reg.get(name)
        assert cls.category == "output"
        assert len(cls.param_schema()) >= 3

    base = GraphIR.from_file("examples/spiral_quickstart.json")
    base.nodes.extend([
        NodeIR(id="watch_loss", type="scalar_watch", params={"reduce": "mean"}),
        NodeIR(id="heat_out", type="heatmap_preview", params={"max_size": 32, "colormap": "coolwarm"}),
        NodeIR(id="peek_in", type="tensor_peek", params={"max_rows": 4, "max_cols": 4}),
        NodeIR(id="mon_tap", type="monitor", params={"mode": "auto"}),
    ])
    base.edges.extend([
        EdgeIR(id="eo1", src_node="ce", src_port="loss", dst_node="watch_loss", dst_port="x"),
        EdgeIR(id="eo2", src_node="mlp", src_port="out", dst_node="heat_out", dst_port="x"),
        EdgeIR(id="eo3", src_node="src", src_port="x", dst_node="peek_in", dst_port="x"),
        EdgeIR(id="eo4", src_node="mlp", src_port="layer_0", dst_node="mon_tap", dst_port="x", edge_kind="tap"),
    ])

    erc = run_erc(base, reg)
    assert erc.passed, erc.format_report()

    compiled = GraphCompiler.compile(base, reg)
    # 输出节点不引入任何可训练参数
    assert sum(p.numel() for n, p in compiled.named_parameters() if "watch_loss" in n or "heat_out" in n) == 0

    batch = {
        "src": {
            "x": Signal(data=torch.randn(16, 2), meta={"layout": "BD"}),
            "y": Signal(data=torch.randint(0, 3, (16,)), meta={"layout": "B"}),
        }
    }
    out = compiled(batch, mode="train")
    assert out.total_loss is not None
    out.total_loss.backward()

    # infer 模式下 ce 被跳过，依赖 ce.loss 的 watch_loss 应安全跳过而不报错
    infer_out = compiled(batch, mode="infer", compute_losses=False)
    assert "heat_out" in infer_out.outputs
    assert "watch_loss" not in infer_out.outputs

    # 校验各类快照结构
    snap_loss = compiled.nodes_mod["watch_loss"].capture_snapshot()
    assert snap_loss is not None
    assert snap_loss["view"] == "scalar"
    assert isinstance(snap_loss["scalar"], float)

    snap_heat = compiled.nodes_mod["heat_out"].capture_snapshot()
    assert snap_heat is not None
    assert snap_heat["view"] == "heatmap"
    assert snap_heat["heatmap"]["rows"] == 16
    assert snap_heat["heatmap"]["cols"] == 3
    assert len(snap_heat["heatmap"]["data"]) == 48

    snap_peek = compiled.nodes_mod["peek_in"].capture_snapshot()
    assert snap_peek is not None
    assert snap_peek["view"] == "raw"
    assert snap_peek["raw"]["rows"] == 4
    assert snap_peek["raw"]["cols"] == 2
    assert len(snap_peek["hist"]["counts"]) == 16

    snap_tap = compiled.nodes_mod["mon_tap"].capture_snapshot()
    assert snap_tap is not None
    assert snap_tap["view"] == "heatmap"


def test_probe_image_downsample_and_nan_safety() -> None:
    reg = get_default_registry()
    peek = reg.get("tensor_peek")().build({"view": "auto"}, {})
    img_batch = torch.rand(8, 1, 28, 28)
    peek({"x": Signal(data=img_batch, meta={"layout": "BCHW"})})
    snap = peek.capture_snapshot()
    assert snap is not None
    assert snap["view"] == "image"
    assert snap["images"] is not None
    assert snap["images"]["count"] == 4
    assert snap["images"]["height"] == 28 and snap["images"]["width"] == 28

    # 超大矩阵与 NaN/Inf 安全降采样
    heat = reg.get("heatmap_preview")().build({"max_size": 16, "symmetric": True}, {})
    big = torch.randn(100, 80)
    big[0, 0] = float("nan")
    big[1, 1] = float("inf")
    heat({"x": Signal(data=big, meta={"layout": "BD"})})
    hsnap = heat.capture_snapshot()
    assert hsnap is not None
    assert hsnap["heatmap"]["rows"] == 16
    assert hsnap["heatmap"]["cols"] == 16
    assert hsnap["heatmap"]["vmin"] == -hsnap["heatmap"]["vmax"]


def test_webui_ws_streams_probe_snapshots() -> None:
    import asyncio

    async def _run() -> None:
        app = create_app(graphs_dir="examples", plugins_dir=None)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            spiral = json.loads(Path("examples/spiral_quickstart.json").read_text(encoding="utf-8"))
            spiral["nodes"].extend([
                {"id": "watch_ce", "type": "scalar_watch", "params": {"reduce": "mean"}},
                {"id": "heat_mlp", "type": "heatmap_preview", "params": {"max_size": 32}},
            ])
            spiral["edges"].extend([
                {"id": "ep1", "src_node": "ce", "src_port": "loss", "dst_node": "watch_ce", "dst_port": "x"},
                {"id": "ep2", "src_node": "mlp", "src_port": "out", "dst_node": "heat_mlp", "dst_port": "x"},
            ])

            r_erc = await client.post("/api/erc/check", json={"graph": spiral})
            assert r_erc.status == 200
            erc_data = await r_erc.json()
            assert erc_data["erc"]["passed"] is True
            watch_node = next(n for n in erc_data["nodes"] if n["id"] == "watch_ce")
            assert watch_node["category"] == "output"
            assert "x" in watch_node["in_specs"]

            async with client.ws_connect("/ws") as ws:
                await ws.receive_str(timeout=3)  # hello
                await client.post(
                    "/api/train/start",
                    json={"graph_name": "probe_test.json", "graph": spiral, "step_delay_ms": 15, "epochs": 1},
                )
                saw_probe = False
                for _ in range(30):
                    msg = json.loads(await ws.receive_str(timeout=5))
                    if msg["type"] == "step" and msg.get("probes"):
                        assert "watch_ce" in msg["probes"]
                        assert "heat_mlp" in msg["probes"]
                        assert msg["probes"]["watch_ce"]["view"] == "scalar"
                        assert msg["probes"]["heat_mlp"]["view"] == "heatmap"
                        saw_probe = True
                        break
                assert saw_probe, "WebSocket step 消息中未包含探针快照"
                await client.post("/api/train/stop")
        finally:
            await client.close()

    asyncio.run(_run())
