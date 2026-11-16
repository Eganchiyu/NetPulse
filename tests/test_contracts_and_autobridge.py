"""ERC 与 Trainer 契约对齐及 AutoBridge 初始化/元数据测试。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from netpulse.core.compiler import GraphCompiler
from netpulse.core.component import BaseComponent
from netpulse.core.erc import run_erc
from netpulse.core.ir import EdgeIR, GraphIR, LossConfig, NodeIR, TrainerConfig
from netpulse.core.registry import ComponentRegistry, get_default_registry
from netpulse.core.trainer import Trainer
from netpulse.core.types import ComponentSignature, PortSpec, ResolvedSpec, Signal


def test_autobridge_init_and_plan_metadata() -> None:
    reg = get_default_registry(None)
    graph = GraphIR(
        name="autobridge_init_test",
        nodes=[
            NodeIR(id="src", type="spiral_data", params={"points_per_class": 30}),
            NodeIR(id="lin", type="linear", params={"in_features": 7, "out_features": 3}),
            NodeIR(id="ce", type="cross_entropy_loss"),
        ],
        edges=[
            EdgeIR(
                id="e_bridge",
                src_node="src",
                src_port="x",
                dst_node="lin",
                dst_port="x",
                bridge="auto",
                bridge_init="zeros",
                bridge_bias_init="zeros",
            ),
            EdgeIR(id="e_out", src_node="lin", src_port="out", dst_node="ce", dst_port="logits"),
            EdgeIR(id="e_y", src_node="src", src_port="y", dst_node="ce", dst_port="target"),
        ],
        losses=[LossConfig(node_id="ce", weight=1.0)],
        trainer=TrainerConfig(batch_size=16, epochs=1, max_steps=2),
    )

    erc = run_erc(graph, reg)
    assert erc.passed, erc.format_report()
    plan = erc.auto_bridges["e_bridge"]
    assert plan.module_type == "Linear"
    assert plan.in_dim == 2
    assert plan.out_dim == 7
    assert plan.param_count == 2 * 7 + 7

    compiled = GraphCompiler.compile(graph, reg)
    bridge_mod = compiled.bridges_mod["e_bridge"]
    assert torch.all(bridge_mod.proj.weight == 0)
    assert torch.all(bridge_mod.proj.bias == 0)


def test_erc_trainer_batch_size_sync_and_small_batch_guards() -> None:
    reg = get_default_registry(None)

    # 1. trainer.batch_size 自动同步到 Meta 形状推导
    g_bs16 = GraphIR(
        name="bs16_sync",
        nodes=[
            NodeIR(id="src", type="spiral_data", params={}),
            NodeIR(id="mlp", type="mlp_block", params={"dims": [16, 3]}),
        ],
        edges=[EdgeIR(id="e1", src_node="src", src_port="x", dst_node="mlp", dst_port="x")],
        trainer=TrainerConfig(batch_size=16),
    )
    erc16 = run_erc(g_bs16, reg)
    assert erc16.passed, erc16.format_report()
    assert erc16.node_out_specs["src"]["x"].shape == (16, 2)
    assert erc16.node_out_specs["mlp"]["out"].shape == (16, 3)

    # 2. batch_size=1 + BatchNorm2d 静态拦截报错
    g_bn_bs1 = GraphIR(
        name="bn_bs1",
        nodes=[
            NodeIR(id="src", type="mnist_data", params={"synthetic_fallback": True}),
            NodeIR(id="conv", type="conv2d", params={"out_channels": 4}),
            NodeIR(id="bn", type="batch_norm2d"),
        ],
        edges=[
            EdgeIR(id="e1", src_node="src", src_port="x", dst_node="conv", dst_port="x"),
            EdgeIR(id="e2", src_node="conv", src_port="out", dst_node="bn", dst_port="x"),
        ],
        trainer=TrainerConfig(batch_size=1),
    )
    erc_bn = run_erc(g_bn_bs1, reg)
    assert not erc_bn.passed
    assert any(d.code == "BATCH_SIZE_TOO_SMALL_FOR_BN" and d.level == "error" for d in erc_bn.diagnostics)

    # 3. batch_size=1 + orthogonal 正则触发退化警告
    g_orth_bs1 = GraphIR(
        name="orth_bs1",
        nodes=[
            NodeIR(id="src", type="spiral_data", params={}),
            NodeIR(id="mlp", type="mlp_block", params={"dims": [8, 3]}),
            NodeIR(id="ce", type="cross_entropy_loss"),
            NodeIR(id="reg", type="activation_reg_loss", params={"mode": "orthogonal"}),
        ],
        edges=[
            EdgeIR(id="e1", src_node="src", src_port="x", dst_node="mlp", dst_port="x"),
            EdgeIR(id="e2", src_node="mlp", src_port="out", dst_node="ce", dst_port="logits"),
            EdgeIR(id="e3", src_node="src", src_port="y", dst_node="ce", dst_port="target"),
            EdgeIR(id="e4", src_node="mlp", src_port="out", dst_node="reg", dst_port="x"),
        ],
        losses=[LossConfig(node_id="ce"), LossConfig(node_id="reg", weight=0.1)],
        trainer=TrainerConfig(batch_size=1),
    )
    erc_orth = run_erc(g_orth_bs1, reg)
    assert erc_orth.passed
    assert any(d.code == "DEGENERATE_ORTHOGONAL_REG_BS1" and d.level == "warning" for d in erc_orth.diagnostics)


def test_multi_source_training_and_eval() -> None:
    reg = get_default_registry(None)
    graph = GraphIR(
        name="dual_source_fusion",
        nodes=[
            NodeIR(id="src_a", type="spiral_data", params={"points_per_class": 40, "noise": 0.1}),
            NodeIR(id="src_b", type="spiral_data", params={"points_per_class": 40, "noise": 0.2}),
            NodeIR(id="mlp_a", type="mlp_block", params={"dims": [16, 8]}),
            NodeIR(id="mlp_b", type="mlp_block", params={"dims": [16, 8]}),
            NodeIR(id="fuse", type="tensor_add", params={"alpha": 0.5, "beta": 0.5}),
            NodeIR(id="head", type="linear", params={"out_features": 3}),
            NodeIR(id="ce", type="cross_entropy_loss"),
        ],
        edges=[
            EdgeIR(id="e1", src_node="src_a", src_port="x", dst_node="mlp_a", dst_port="x"),
            EdgeIR(id="e2", src_node="src_b", src_port="x", dst_node="mlp_b", dst_port="x"),
            EdgeIR(id="e3", src_node="mlp_a", src_port="out", dst_node="fuse", dst_port="a"),
            EdgeIR(id="e4", src_node="mlp_b", src_port="out", dst_node="fuse", dst_port="b"),
            EdgeIR(id="e5", src_node="fuse", src_port="out", dst_node="head", dst_port="x"),
            EdgeIR(id="e6", src_node="head", src_port="out", dst_node="ce", dst_port="logits"),
            EdgeIR(id="e7", src_node="src_a", src_port="y", dst_node="ce", dst_port="target"),
        ],
        losses=[LossConfig(node_id="ce", weight=1.0)],
        trainer=TrainerConfig(batch_size=32, epochs=2, lr=0.01),
    )

    compiled = GraphCompiler.compile(graph, reg)
    res = Trainer(compiled).fit(verbose=False)
    assert res.steps > 0
    assert res.final_val_loss is not None
    assert res.final_val_acc is not None


class _CustomFocalLossModule(nn.Module):
    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        logits = inputs["scores"].data
        target = inputs["labels"].data.long()
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** 2 * ce).mean()
        return {"loss": Signal(data=loss, meta={"layout": "SCALAR"})}


class _CustomFocalLoss(BaseComponent):
    category = "loss"
    display_name = "FocalLoss"

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="scores", kind="tensor", rank=[2], layout="BD"),
                PortSpec(name="labels", kind="tensor", rank=[1], layout="B", dtype="int64"),
            ],
            outputs=[PortSpec(name="loss", kind="loss", rank=[0], layout="SCALAR")],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _CustomFocalLossModule()

    def compute_metrics(self, params: dict[str, Any], inputs: dict[str, Signal]) -> dict[str, tuple[float, int]]:
        scores = inputs["scores"].data
        labels = inputs["labels"].data.long()
        correct = float((scores.argmax(dim=-1) == labels).sum().item())
        return {"acc": (correct, int(labels.shape[0]))}


def test_custom_plugin_loss_compute_metrics() -> None:
    base_reg = get_default_registry(None)
    custom_reg = ComponentRegistry()
    for tid in base_reg.list_types():
        custom_reg.register(tid, base_reg.get(tid))
    custom_reg.register("custom_focal_loss", _CustomFocalLoss)

    graph = GraphIR(
        name="custom_loss_metrics",
        nodes=[
            NodeIR(id="src", type="spiral_data", params={"points_per_class": 50}),
            NodeIR(id="mlp", type="mlp_block", params={"dims": [32, 32, 3]}),
            NodeIR(id="focal", type="custom_focal_loss"),
        ],
        edges=[
            EdgeIR(id="e1", src_node="src", src_port="x", dst_node="mlp", dst_port="x"),
            EdgeIR(id="e2", src_node="mlp", src_port="out", dst_node="focal", dst_port="scores"),
            EdgeIR(id="e3", src_node="src", src_port="y", dst_node="focal", dst_port="labels"),
        ],
        losses=[LossConfig(node_id="focal", weight=1.0)],
        trainer=TrainerConfig(batch_size=32, epochs=5, lr=0.02),
    )

    compiled = GraphCompiler.compile(graph, custom_reg)
    res = Trainer(compiled).fit(verbose=False)
    assert res.final_val_acc is not None
    assert 0.0 <= res.final_val_acc <= 1.0
