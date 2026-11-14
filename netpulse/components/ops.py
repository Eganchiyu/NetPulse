"""张量变换、矩阵投影桥与算术连接元件。"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from netpulse.core.component import BaseComponent
from netpulse.core.registry import register_component
from netpulse.core.types import ComponentSignature, ParamSpec, PortSpec, ResolvedSpec, Signal


class _FlattenModule(nn.Module):
    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=start_dim)

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        return {"out": sig.clone_with(data=self.flatten(sig.data), layout="BD")}


@register_component("flatten")
class Flatten(BaseComponent):
    """将高维特征图（如 BCHW）展平为 2D 特征向量（BD）。"""

    category = "op"
    display_name = "Flatten"
    description = "高维张量展平为 BD 向量"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("start_dim", "int", 1, min=1, max=3, step=1, description="起始展平维度"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[2, 3, 4], layout="ANY")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2], layout="BD")],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _FlattenModule(start_dim=int(params.get("start_dim", 1)))


class _GlobalAvgPoolModule(nn.Module):
    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        pooled = sig.data.mean(dim=(-2, -1))
        return {"out": sig.clone_with(data=pooled, layout="BD")}


@register_component("global_avg_pool")
class GlobalAvgPool(BaseComponent):
    """空间全局平均池化（BCHW -> BD）。"""

    category = "op"
    display_name = "GlobalAvgPool"
    description = "对空间维 H×W 求全局均值（BCHW -> BD）"

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[4], layout="BCHW")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2], layout="BD")],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _GlobalAvgPoolModule()


class _MatrixBridgeModule(nn.Module):
    """可学习线性 / 1x1 Conv 投影桥，用于特征空间映射与维度对齐。"""

    def __init__(self, in_dim: int, out_dim: int, layout: str, bias: bool = True) -> None:
        super().__init__()
        if layout == "BCHW":
            self.proj: nn.Module = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)
        else:
            self.proj = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        return {"out": sig.clone_with(data=self.proj(sig.data))}


@register_component("matrix_bridge")
class MatrixBridge(BaseComponent):
    """显式矩阵连接/投影元件。"""

    category = "bridge"
    display_name = "MatrixBridge"
    description = "可学习线性 / 1×1 Conv 空间投影桥"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("out_dim", "int", 32, required=True, min=1, max=8192, step=1, description="目标投影维度"),
            ParamSpec("bias", "bool", True, description="是否启用偏置"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[2, 3, 4], layout="ANY")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2, 3, 4], layout="ANY")],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        in_spec = in_specs["x"]
        in_dim = in_spec.shape[1] if in_spec.layout == "BCHW" and in_spec.rank == 4 else in_spec.shape[-1]
        out_dim = int(params.get("out_dim", 32))
        bias = bool(params.get("bias", True))
        return _MatrixBridgeModule(in_dim=in_dim, out_dim=out_dim, layout=in_spec.layout, bias=bias)


class _TensorAddModule(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        a = inputs["a"]
        b = inputs["b"]
        out = self.alpha * a.data + self.beta * b.data
        return {"out": a.clone_with(data=out)}


@register_component("tensor_add")
class TensorAdd(BaseComponent):
    """双路张量加权融合（若 b 维度与 a 不一致且配置 bridge='auto'，ERC 自动插入矩阵桥）。"""

    category = "op"
    display_name = "TensorAdd"
    description = "双路残差/特征加权融合（alpha·a + beta·b）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("alpha", "float", 1.0, min=-10.0, max=10.0, step=0.1, description="分支 a 权重"),
            ParamSpec("beta", "float", 1.0, min=-10.0, max=10.0, step=0.1, description="分支 b 权重"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="a", kind="tensor", rank=[2, 3, 4], layout="ANY"),
                PortSpec(name="b", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2, 3, 4], layout="ANY")],
        )

    def expected_input_spec(
        self,
        params: dict[str, Any],
        port_name: str,
        actual: ResolvedSpec,
        resolved_in_specs: dict[str, ResolvedSpec],
    ) -> Optional[ResolvedSpec]:
        if port_name == "b" and "a" in resolved_in_specs:
            return resolved_in_specs["a"]
        return None

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _TensorAddModule(
            alpha=float(params.get("alpha", 1.0)),
            beta=float(params.get("beta", 1.0)),
        )
