"""多目标监督 Loss 与可精确计算正则化 Loss 元件。"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from netpulse.core.component import BaseComponent
from netpulse.core.registry import register_component
from netpulse.core.types import ComponentSignature, ParamSpec, PortSpec, ResolvedSpec, Signal


class _CrossEntropyModule(nn.Module):
    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        logits = inputs["logits"].data
        target = inputs["target"].data.long()
        loss = F.cross_entropy(logits, target)
        return {"loss": Signal(data=loss, meta={"layout": "SCALAR"})}


@register_component("cross_entropy_loss")
class CrossEntropyLoss(BaseComponent):
    """多分类交叉熵监督目标节点。"""

    category = "loss"
    display_name = "CrossEntropyLoss"
    title = "交叉熵损失"
    description = "多分类交叉熵监督损失目标"

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="logits", title="预测得分", description="BD 分类 logits（未过 softmax）", kind="tensor", rank=[2], layout="BD"),
                PortSpec(name="target", title="真实标签", description="B 整型类别索引", kind="tensor", rank=[1], layout="B", dtype="int64"),
            ],
            outputs=[
                PortSpec(name="loss", title="损失标量", description="0 维标量，汇总进 total_loss", kind="loss", rank=[0], layout="SCALAR"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _CrossEntropyModule()

    def compute_metrics(
        self,
        params: dict[str, Any],
        inputs: dict[str, Signal],
    ) -> dict[str, tuple[float, int]]:
        logits_sig = inputs.get("logits")
        target_sig = inputs.get("target")
        if logits_sig is None or target_sig is None:
            return {}
        logits = logits_sig.data
        target = target_sig.data
        if target.dtype not in (torch.int64, torch.int32):
            return {}
        if logits.ndim == 2 and target.ndim == 1 and logits.shape[0] == target.shape[0]:
            correct = float((logits.argmax(dim=-1) == target.long()).sum().item())
            return {"acc": (correct, int(target.shape[0]))}
        return {}


class _ActivationRegModule(nn.Module):
    """对隐层激活/投影特征施加可精确计算的正则化约束。"""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        feat = inputs["x"].data
        if feat.ndim > 2:
            feat = feat.flatten(start_dim=1)

        if self.mode == "l1":
            loss = feat.abs().mean()
        elif self.mode == "l2":
            loss = (feat ** 2).mean()
        elif self.mode == "orthogonal":
            # 批次内归一化特征列/行去相关正则：||C - I||_F^2 / D^2
            centered = feat - feat.mean(dim=0, keepdim=True)
            norm = centered.pow(2).sum(dim=0, keepdim=True).add(1e-6).sqrt()
            normed = centered / norm
            corr = normed.t() @ normed
            eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
            loss = ((corr - eye) ** 2).mean()
        else:
            raise ValueError(f"Unsupported activation reg mode: {self.mode}")

        return {"loss": Signal(data=loss, meta={"layout": "SCALAR"})}


@register_component("activation_reg_loss")
class ActivationRegLoss(BaseComponent):
    """隐层激活精确正则化节点（支持 l1 / l2 / orthogonal）。"""

    category = "loss"
    display_name = "ActivationRegLoss"
    title = "激活正则损失"
    description = "可精确计算的隐层激活正则项（L1 / L2 / 正交去相关）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("mode", "select", "orthogonal", options=["l1", "l2", "orthogonal"], description="正则化模式"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="被正则激活", description="任意隐层激活张量", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="loss", title="正则标量", description="L1/L2/正交去相关的 0 维标量", kind="loss", rank=[0], layout="SCALAR"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        mode = str(params.get("mode", "l1")).lower()
        return _ActivationRegModule(mode=mode)


class _MSELossModule(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        pred = inputs["pred"].data.float()
        target = inputs["target"].data.float()
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": Signal(data=loss, meta={"layout": "SCALAR"})}


@register_component("mse_loss")
class MSELoss(BaseComponent):
    """均方误差（MSE）回归监督损失节点。"""

    category = "loss"
    display_name = "MSELoss"
    title = "均方误差损失"
    description = "均方误差回归损失（用于时序预测、连续值回归任务）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "reduction",
                "select",
                "mean",
                options=["mean", "sum"],
                description="批次损失聚合方式",
            ),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="pred", title="预测值", description="模型输出张量", kind="tensor", rank=[1, 2, 3, 4], layout="ANY"),
                PortSpec(name="target", title="真实值", description="同形状回归真值", kind="tensor", rank=[1, 2, 3, 4], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="loss", title="损失标量", description="0 维标量，汇总进 total_loss", kind="loss", rank=[0], layout="SCALAR"),
            ],
        )

    def expected_input_spec(
        self,
        params: dict[str, Any],
        port_name: str,
        actual: ResolvedSpec,
        resolved_in_specs: dict[str, ResolvedSpec],
    ) -> Optional[ResolvedSpec]:
        if port_name == "target" and "pred" in resolved_in_specs:
            return resolved_in_specs["pred"]
        return None

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _MSELossModule(reduction=str(params.get("reduction", "mean")))
