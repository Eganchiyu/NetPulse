"""多目标监督 Loss 与可精确计算正则化 Loss 元件。"""

from __future__ import annotations

from typing import Any

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
    description = "多分类交叉熵监督损失目标"

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="logits", kind="tensor", rank=[2], layout="BD"),
                PortSpec(name="target", kind="tensor", rank=[1], layout="B", dtype="int64"),
            ],
            outputs=[PortSpec(name="loss", kind="loss", rank=[0], layout="SCALAR")],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _CrossEntropyModule()


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
    description = "可精确计算的隐层激活正则项（L1 / L2 / 正交去相关）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("mode", "select", "orthogonal", options=["l1", "l2", "orthogonal"], description="正则化模式"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[2, 3, 4], layout="ANY")],
            outputs=[PortSpec(name="loss", kind="loss", rank=[0], layout="SCALAR")],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        mode = str(params.get("mode", "l1")).lower()
        return _ActivationRegModule(mode=mode)
