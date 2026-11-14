"""ERC 静态检查、AutoBridge、Tap 与自定义插件加载测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from netpulse.core.compiler import GraphCompiler
from netpulse.core.erc import run_erc
from netpulse.core.ir import GraphIR
from netpulse.core.registry import ComponentRegistry, get_default_registry
from netpulse.core.types import Signal


def test_erc_cycle_detection() -> None:
    reg = get_default_registry(None)
    graph = GraphIR.model_validate(
        {
            "nodes": [
                {"id": "a", "type": "mlp_block", "params": {"in_dim": 4, "dims": [4]}},
                {"id": "b", "type": "mlp_block", "params": {"in_dim": 4, "dims": [4]}},
            ],
            "edges": [
                {"id": "e1", "src_node": "a", "src_port": "out", "dst_node": "b", "dst_port": "x"},
                {"id": "e2", "src_node": "b", "src_port": "out", "dst_node": "a", "dst_port": "x"},
            ],
        }
    )
    res = run_erc(graph, reg)
    assert not res.passed
    assert any(d.code == "CYCLE_DETECTED" for d in res.errors())


def test_erc_rank_mismatch_suggests_flatten() -> None:
    reg = get_default_registry(None)
    graph = GraphIR.model_validate(
        {
            "nodes": [
                {"id": "src", "type": "mnist_data", "params": {}},
                {"id": "head", "type": "mlp_block", "params": {"dims": [10]}},
            ],
            "edges": [
                {"id": "e1", "src_node": "src", "src_port": "x", "dst_node": "head", "dst_port": "x"},
            ],
        }
    )
    res = run_erc(graph, reg)
    assert not res.passed
    errs = [d for d in res.errors() if d.code == "RANK_MISMATCH"]
    assert errs and "Flatten" in errs[0].message


def test_erc_autobridge_dim_mismatch_and_repair() -> None:
    reg = get_default_registry(None)
    base_dict = {
        "nodes": [
            {"id": "src", "type": "spiral_data", "params": {}},
            {"id": "m1", "type": "mlp_block", "params": {"dims": [16]}},
            {"id": "m2", "type": "mlp_block", "params": {"in_dim": 8, "dims": [3]}},
            {"id": "ce", "type": "cross_entropy_loss", "params": {}},
        ],
        "edges": [
            {"id": "e1", "src_node": "src", "src_port": "x", "dst_node": "m1", "dst_port": "x"},
            {"id": "e2", "src_node": "m1", "src_port": "out", "dst_node": "m2", "dst_port": "x", "bridge": "none"},
            {"id": "e3", "src_node": "m2", "src_port": "out", "dst_node": "ce", "dst_port": "logits"},
            {"id": "e4", "src_node": "src", "src_port": "y", "dst_node": "ce", "dst_port": "target"},
        ],
    }

    # bridge="none" 时应报错 DIM_MISMATCH
    res_fail = run_erc(GraphIR.model_validate(base_dict), reg)
    assert not res_fail.passed
    assert any(d.code == "DIM_MISMATCH" for d in res_fail.errors())

    # bridge="auto" 时自动插入 MatrixBridge 并通过编译与前向反传
    base_dict["edges"][1]["bridge"] = "auto"
    graph_ok = GraphIR.model_validate(base_dict)
    compiled = GraphCompiler.compile(graph_ok, reg)
    assert "e2" in compiled.bridges_mod

    batch = {
        "src": {
            "x": Signal(data=torch.randn(8, 2), meta={"layout": "BD"}),
            "y": Signal(data=torch.randint(0, 3, (8,)), meta={"layout": "B"}),
        }
    }
    out = compiled(batch, mode="train")
    assert out.total_loss is not None
    out.total_loss.backward()
    assert compiled.bridges_mod["e2"].proj.weight.grad is not None

    # infer 模式应跳过 LossNode
    infer_out = compiled(batch, mode="infer")
    assert infer_out.total_loss is None
    assert "ce" not in infer_out.outputs


def test_tensor_add_autobridge_residual() -> None:
    reg = get_default_registry(None)
    graph = GraphIR.model_validate(
        {
            "nodes": [
                {"id": "src", "type": "spiral_data", "params": {}},
                {"id": "m1", "type": "mlp_block", "params": {"dims": [12]}},
                {"id": "add", "type": "tensor_add", "params": {}},
            ],
            "edges": [
                {"id": "ea", "src_node": "m1", "src_port": "out", "dst_node": "add", "dst_port": "a"},
                {"id": "eb", "src_node": "src", "src_port": "x", "dst_node": "add", "dst_port": "b", "bridge": "auto"},
                {"id": "em", "src_node": "src", "src_port": "x", "dst_node": "m1", "dst_port": "x"},
            ],
        }
    )
    compiled = GraphCompiler.compile(graph, reg)
    assert "eb" in compiled.bridges_mod
    batch = {
        "src": {
            "x": Signal(data=torch.randn(5, 2), meta={"layout": "BD"}),
            "y": Signal(data=torch.randint(0, 3, (5,)), meta={"layout": "B"}),
        }
    }
    out = compiled(batch, mode="infer")
    assert out.outputs["add"]["out"].shape == (5, 12)


def test_plugin_autoload(tmp_path: Path) -> None:
    plugin_file = tmp_path / "custom_scale.py"
    plugin_file.write_text(
        '''
import torch.nn as nn
from netpulse.core.component import BaseComponent
from netpulse.core.registry import register_component
from netpulse.core.types import ComponentSignature, PortSpec

class _ScaleMod(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor
    def forward(self, inputs):
        s = inputs["x"]
        return {"out": s.clone_with(data=s.data * self.factor)}

@register_component("custom_scale_test")
class CustomScale(BaseComponent):
    category = "op"
    @classmethod
    def declare_ports(cls, params):
        return ComponentSignature(
            inputs=[PortSpec(name="x", rank=[2], layout="BD")],
            outputs=[PortSpec(name="out", rank=[2], layout="BD")],
        )
    def build(self, params, in_specs):
        return _ScaleMod(float(params.get("factor", 2.0)))
''',
        encoding="utf-8",
    )

    reg = get_default_registry(tmp_path)
    assert reg.has("custom_scale_test")
    graph = GraphIR.model_validate(
        {
            "nodes": [
                {"id": "src", "type": "spiral_data", "params": {}},
                {"id": "sc", "type": "custom_scale_test", "params": {"factor": 3.0}},
            ],
            "edges": [
                {"id": "e1", "src_node": "src", "src_port": "x", "dst_node": "sc", "dst_port": "x"},
            ],
        }
    )
    compiled = GraphCompiler.compile(graph, reg)
    x = torch.ones(4, 2)
    out = compiled({"src": {"x": Signal(data=x, meta={"layout": "BD"}), "y": Signal(data=torch.zeros(4, dtype=torch.long), meta={"layout": "B"})}}, mode="infer")
    assert torch.allclose(out.outputs["sc"]["out"].data, torch.full((4, 2), 3.0))
