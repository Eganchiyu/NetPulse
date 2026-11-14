"""原子级神经网络层、统一初始化与增强 Block 单元测试。"""

from __future__ import annotations

import torch

from netpulse.core.compiler import GraphCompiler
from netpulse.core.erc import run_erc
from netpulse.core.ir import GraphIR
from netpulse.core.registry import get_default_registry
from netpulse.core.trainer import Trainer


def test_atomic_init_and_erc_channel_check() -> None:
    reg = get_default_registry(None)
    # 验证常量初始化、orthogonal 初始化与 conv2d/linear 显式输入维度校验
    graph_bad = GraphIR.model_validate(
        {
            "nodes": [
                {"id": "src", "type": "mnist_data", "params": {}},
                {"id": "c1", "type": "conv2d", "params": {"in_channels": 3, "out_channels": 8, "kernel_size": 5}},
            ],
            "edges": [
                {"id": "e1", "src_node": "src", "src_port": "x", "dst_node": "c1", "dst_port": "x", "bridge": "none"},
            ],
        }
    )
    res_bad = run_erc(graph_bad, reg)
    assert not res_bad.passed
    assert any(d.code == "DIM_MISMATCH" for d in res_bad.errors())

    graph_ok = GraphIR.model_validate(
        {
            "nodes": [
                {"id": "src", "type": "spiral_data", "params": {}},
                {
                    "id": "lin1",
                    "type": "linear",
                    "params": {
                        "in_features": 2,
                        "out_features": 4,
                        "init": {"type": "constant", "val": 0.25},
                        "bias_init": "zeros",
                    },
                },
                {"id": "ln", "type": "layer_norm", "params": {}},
                {"id": "act", "type": "activation", "params": {"kind": "gelu"}},
                {"id": "lin2", "type": "linear", "params": {"in_features": 8, "out_features": 3, "init": "orthogonal"}},
            ],
            "edges": [
                {"id": "e1", "src_node": "src", "src_port": "x", "dst_node": "lin1", "dst_port": "x"},
                {"id": "e2", "src_node": "lin1", "src_port": "out", "dst_node": "ln", "dst_port": "x"},
                {"id": "e3", "src_node": "ln", "src_port": "out", "dst_node": "act", "dst_port": "x"},
                {"id": "e4", "src_node": "act", "src_port": "out", "dst_node": "lin2", "dst_port": "x", "bridge": "auto"},
            ],
        }
    )
    compiled = GraphCompiler.compile(graph_ok, reg)
    assert "e4" in compiled.bridges_mod
    w1 = compiled.nodes_mod["lin1"].op.weight
    b1 = compiled.nodes_mod["lin1"].op.bias
    assert torch.allclose(w1, torch.full_like(w1, 0.25))
    assert torch.allclose(b1, torch.zeros_like(b1))


def test_enhanced_cnn_block_per_layer_cfg() -> None:
    reg = get_default_registry(None)
    graph = GraphIR.model_validate(
        {
            "nodes": [
                {"id": "src", "type": "mnist_data", "params": {}},
                {
                    "id": "cnn",
                    "type": "cnn_block",
                    "params": {
                        "layers": [
                            {"out_channels": 8, "kernel_size": 5, "padding": 2, "pool": "max", "pool_size": 2, "activation": "gelu"},
                            {"out_channels": 16, "kernel_size": 3, "padding": 1, "pool": "avg", "pool_size": 2, "activation": "relu"},
                        ],
                        "init": "kaiming_normal",
                    },
                },
            ],
            "edges": [
                {"id": "e1", "src_node": "src", "src_port": "x", "dst_node": "cnn", "dst_port": "x"},
            ],
        }
    )
    res = run_erc(graph, reg)
    assert res.passed
    assert res.node_out_specs["cnn"]["out"].shape == (64, 16, 7, 7)


def test_mnist_atomic_e2e() -> None:
    reg = get_default_registry(None)
    graph = GraphIR.from_file("examples/mnist_atomic.json")
    graph.trainer.epochs = 1
    graph.trainer.max_steps = 20
    compiled = GraphCompiler.compile(graph, reg)
    trainer = Trainer(compiled)
    res = trainer.fit(verbose=False)
    assert res.steps == 20
    assert res.final_val_acc is not None and res.final_val_acc >= 0.70
