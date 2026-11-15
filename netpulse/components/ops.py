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
    title = "张量展平"
    description = "高维张量展平为 BD 向量"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("start_dim", "int", 1, min=1, max=3, step=1, description="起始展平维度"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入张量", description="2~4 维张量，从 start_dim 起展平", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="out", title="展平结果", description="形状 BD 的二维张量", kind="tensor", rank=[2], layout="BD"),
            ],
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
    title = "全局平均池化"
    description = "对空间维 H×W 求全局均值（BCHW -> BD）"

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入特征图", description="BCHW 特征图", kind="tensor", rank=[4], layout="BCHW"),
            ],
            outputs=[
                PortSpec(name="out", title="池化向量", description="对 H×W 求均值后的 BD 向量", kind="tensor", rank=[2], layout="BD"),
            ],
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
    title = "矩阵桥接投影"
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
            inputs=[
                PortSpec(name="x", title="输入张量", description="维度与下游不一致的特征", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="out", title="投影输出", description="对齐到目标维度的同阶张量", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
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
    title = "张量相加融合"
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
                PortSpec(name="a", title="主路输入", description="alpha 加权的第一路特征", kind="tensor", rank=[2, 3, 4], layout="ANY"),
                PortSpec(name="b", title="支路输入", description="beta 加权的第二路特征", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="out", title="融合输出", description="alpha·a + beta·b 的逐元素和", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
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


class _ImageToSeqModule(nn.Module):
    """将 4D BCHW 特征图按行或列展开为 3D BLD 序列。"""

    def __init__(self, mode: str = "rows") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        x = sig.data
        b, c, h, w = x.shape
        if self.mode == "cols":
            # (B, C, H, W) -> (B, W, C * H)
            seq = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        else:
            # (B, C, H, W) -> (B, H, C * W)
            seq = x.permute(0, 2, 1, 3).reshape(b, h, c * w)
        return {"out": Signal(data=seq, mask=None, meta={**sig.meta, "layout": "BLD"})}


@register_component("image_to_seq")
class ImageToSeq(BaseComponent):
    """图像转序列算子（BCHW -> BLD），可直接把 MNIST 等图像喂给 LSTM。"""

    category = "op"
    display_name = "ImageToSeq"
    title = "图像转序列"
    description = "将 4D 图像按行/列切片展开为 3D BLD 序列"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "mode",
                "select",
                "rows",
                options=["rows", "cols"],
                description="切片方向（rows: 高作时间步; cols: 宽作时间步）",
            ),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入图像", description="BCHW 图像/特征图", kind="tensor", rank=[4], layout="BCHW"),
            ],
            outputs=[
                PortSpec(name="out", title="序列张量", description="按行或列切片展开的 BLD 序列", kind="tensor", rank=[3], layout="BLD"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _ImageToSeqModule(mode=str(params.get("mode", "rows")))


class _SeqPoolModule(nn.Module):
    """感知 Mask 的序列汇聚算子（BLD -> BD）。"""

    def __init__(self, mode: str = "last") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        x = sig.data
        mask = sig.mask
        b, l, d = x.shape

        if self.mode == "mean":
            if mask is not None:
                m = mask.unsqueeze(-1).to(x.dtype)
                pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                pooled = x.mean(dim=1)
        elif self.mode == "max":
            if mask is not None:
                m = mask.unsqueeze(-1).bool()
                pooled = x.masked_fill(~m, -1e9).max(dim=1).values
            else:
                pooled = x.max(dim=1).values
        else:
            # last valid step
            if mask is not None:
                lengths = mask.long().sum(dim=1).clamp(min=1) - 1
                idx = lengths.view(b, 1, 1).expand(b, 1, d)
                pooled = x.gather(dim=1, index=idx).squeeze(1)
            else:
                pooled = x[:, -1, :]

        return {"out": Signal(data=pooled, mask=None, meta={**sig.meta, "layout": "BD"})}


@register_component("seq_pool")
class SeqPool(BaseComponent):
    """序列汇聚算子（BLD -> BD），自动感知有效长度 Mask。"""

    category = "op"
    display_name = "SeqPool"
    title = "序列池化"
    description = "序列池化（last 取末有效步 / mean / max，支持 Mask）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "mode",
                "select",
                "last",
                options=["last", "mean", "max"],
                description="汇聚模式（last 会自动跳过 Mask 填充步）",
            ),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="序列输入", description="BLD 序列，可带 mask", kind="tensor", rank=[3], layout="BLD"),
            ],
            outputs=[
                PortSpec(name="out", title="池化向量", description="last/mean/max 归约后的 BD 向量", kind="tensor", rank=[2], layout="BD"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _SeqPoolModule(mode=str(params.get("mode", "last")))
