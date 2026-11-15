"""原子级神经网络层元件：Conv2d / Pool2d / Linear / Activation / Norm / Dropout。"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from netpulse.components.init_utils import apply_init
from netpulse.core.component import BaseComponent
from netpulse.core.registry import register_component
from netpulse.core.types import ComponentSignature, ParamSpec, PortSpec, ResolvedSpec, Signal

_INIT_OPTIONS = [
    "",
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
    "orthogonal",
    "normal",
    "uniform",
]


class _SingleModuleWrapper(nn.Module):
    """通用单输入单输出模块封装，透传 Signal 与 Mask。"""

    def __init__(self, op: nn.Module, out_layout: Optional[str] = None) -> None:
        super().__init__()
        self.op = op
        self.out_layout = out_layout

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        out_data = self.op(sig.data)
        if self.out_layout is not None:
            return {"out": sig.clone_with(data=out_data, layout=self.out_layout)}
        return {"out": sig.clone_with(data=out_data)}


@register_component("conv2d")
class Conv2dLayer(BaseComponent):
    """原子 2D 卷积层：可配核尺寸、步长、填充、空洞、分组与权重初始值。"""

    category = "backbone"
    display_name = "Conv2d"
    title = "二维卷积层"
    description = "原子 2D 卷积层（支持显式 in_channels 校验或自动推导）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("out_channels", "int", 16, required=True, min=1, max=2048, step=1, description="输出通道数"),
            ParamSpec("kernel_size", "int", 3, min=1, max=15, step=1, description="卷积核尺寸"),
            ParamSpec("stride", "int", 1, min=1, max=8, step=1, description="步长"),
            ParamSpec("padding", "int", 1, min=0, max=8, step=1, description="填充"),
            ParamSpec("dilation", "int", 1, min=1, max=8, step=1, description="空洞率"),
            ParamSpec("groups", "int", 1, min=1, max=64, step=1, description="分组卷积"),
            ParamSpec("bias", "bool", True, description="是否启用偏置"),
            ParamSpec("in_channels", "int", None, min=1, max=2048, step=1, description="可选：强制输入通道校验"),
            ParamSpec("init", "select", None, options=_INIT_OPTIONS, description="权重初始化方式"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入特征图", description="待卷积的 BCHW 图像/特征图", kind="tensor", rank=[4], layout="BCHW"),
            ],
            outputs=[
                PortSpec(name="out", title="卷积输出", description="通道数变为 out_channels 的 BCHW 特征图", kind="tensor", rank=[4], layout="BCHW"),
            ],
        )

    def expected_input_spec(
        self,
        params: dict[str, Any],
        port_name: str,
        actual: ResolvedSpec,
        resolved_in_specs: dict[str, ResolvedSpec],
    ) -> Optional[ResolvedSpec]:
        if port_name == "x" and params.get("in_channels") is not None:
            c = int(params["in_channels"])
            b, _, h, w = actual.shape
            return ResolvedSpec(shape=(b, c, h, w), dtype=actual.dtype, layout="BCHW", has_mask=actual.has_mask)
        return None

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        raw_in = params.get("in_channels")
        in_channels = int(raw_in if raw_in is not None else in_specs["x"].shape[1])
        out_channels = int(params.get("out_channels", 16))
        kernel_size = params.get("kernel_size", 3)
        stride = params.get("stride", 1)
        padding = params.get("padding", kernel_size // 2 if isinstance(kernel_size, int) else 0)
        dilation = params.get("dilation", 1)
        groups = int(params.get("groups", 1))
        bias = bool(params.get("bias", True))

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        apply_init(conv.weight, conv.bias, params.get("init") or None, params.get("bias_init") or None)
        return _SingleModuleWrapper(conv, out_layout="BCHW")


@register_component("pool2d")
class Pool2dLayer(BaseComponent):
    """原子 2D 池化层：支持 max / avg 模式与核尺寸、步长、填充配置。"""

    category = "op"
    display_name = "Pool2d"
    title = "二维池化层"
    description = "2D 空间池化（MaxPool2d / AvgPool2d）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("mode", "select", "max", options=["max", "avg"], description="池化模式"),
            ParamSpec("kernel_size", "int", 2, min=1, max=8, step=1, description="池化窗口"),
            ParamSpec("stride", "int", 2, min=1, max=8, step=1, description="步长"),
            ParamSpec("padding", "int", 0, min=0, max=4, step=1, description="填充"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入特征图", description="BCHW 特征图", kind="tensor", rank=[4], layout="BCHW"),
            ],
            outputs=[
                PortSpec(name="out", title="池化输出", description="空间尺寸被窗口缩小的 BCHW 特征图", kind="tensor", rank=[4], layout="BCHW"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        mode = str(params.get("mode", "max")).lower()
        kernel_size = params.get("kernel_size", 2)
        stride = params.get("stride", kernel_size)
        padding = params.get("padding", 0)
        if mode == "max":
            op: nn.Module = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        elif mode == "avg":
            op = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError(f"Unsupported pool2d mode: {mode}")
        return _SingleModuleWrapper(op, out_layout="BCHW")


@register_component("linear")
class LinearLayer(BaseComponent):
    """原子全连接层（可用于特征投影或末端 k 维分类头）。"""

    category = "backbone"
    display_name = "Linear"
    title = "全连接层"
    description = "全连接线性层（特征投影或分类头）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("out_features", "int", 64, required=True, min=1, max=8192, step=1, description="输出维度"),
            ParamSpec("bias", "bool", True, description="是否启用偏置"),
            ParamSpec("in_features", "int", None, min=1, max=8192, step=1, description="可选：强制输入维度校验"),
            ParamSpec("init", "select", None, options=_INIT_OPTIONS, description="权重初始化方式"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入向量", description="任意 ...D 或 BLD 张量，沿最后一维做线性映射", kind="tensor", rank=[2, 3], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="out", title="线性输出", description="最后一维变为 out_features", kind="tensor", rank=[2, 3], layout="ANY"),
            ],
        )

    def expected_input_spec(
        self,
        params: dict[str, Any],
        port_name: str,
        actual: ResolvedSpec,
        resolved_in_specs: dict[str, ResolvedSpec],
    ) -> Optional[ResolvedSpec]:
        if port_name == "x" and params.get("in_features") is not None:
            d = int(params["in_features"])
            new_shape = actual.shape[:-1] + (d,)
            return ResolvedSpec(shape=new_shape, dtype=actual.dtype, layout=actual.layout, has_mask=actual.has_mask)
        return None

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        raw_in = params.get("in_features")
        in_features = int(raw_in if raw_in is not None else in_specs["x"].shape[-1])
        out_features = int(params.get("out_features", 64))
        bias = bool(params.get("bias", True))
        linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        apply_init(linear.weight, linear.bias, params.get("init") or None, params.get("bias_init") or None)
        return _SingleModuleWrapper(linear)


@register_component("activation")
class ActivationLayer(BaseComponent):
    """原子激活函数层。"""

    category = "op"
    display_name = "Activation"
    title = "激活函数"
    description = "逐元素激活函数层（ReLU/GELU/SiLU/Tanh/Sigmoid/Softmax）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "kind",
                "select",
                "relu",
                options=["relu", "gelu", "silu", "tanh", "sigmoid", "leaky_relu", "softmax", "identity"],
                description="激活函数类型",
            ),
            ParamSpec("negative_slope", "float", 0.01, min=0.0, max=0.5, step=0.01, description="LeakyReLU 负斜率"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入张量", description="任意 2~4 维张量", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="out", title="激活输出", description="逐元素激活后的同形状张量", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        kind = str(params.get("kind", "relu")).lower()
        if kind == "relu":
            op: nn.Module = nn.ReLU()
        elif kind == "gelu":
            op = nn.GELU()
        elif kind == "silu":
            op = nn.SiLU()
        elif kind == "tanh":
            op = nn.Tanh()
        elif kind == "sigmoid":
            op = nn.Sigmoid()
        elif kind == "leaky_relu":
            op = nn.LeakyReLU(negative_slope=float(params.get("negative_slope", 0.01)))
        elif kind == "softmax":
            op = nn.Softmax(dim=int(params.get("dim", -1)))
        elif kind == "identity":
            op = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation kind: {kind}")
        return _SingleModuleWrapper(op)


@register_component("batch_norm2d")
class BatchNorm2dLayer(BaseComponent):
    """2D 批归一化层（BCHW）。"""

    category = "op"
    display_name = "BatchNorm2d"
    title = "批归一化 (2D)"
    description = "2D 批归一化（BCHW 通道自动推导）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("eps", "float", 1e-5, min=1e-8, max=1e-2, step=1e-5, description="数值稳定项"),
            ParamSpec("momentum", "float", 0.1, min=0.0, max=1.0, step=0.05, description="滑动统计动量"),
            ParamSpec("affine", "bool", True, description="是否可学习仿射参数"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入特征图", description="BCHW 特征图，按通道统计", kind="tensor", rank=[4], layout="BCHW"),
            ],
            outputs=[
                PortSpec(name="out", title="归一化输出", description="同形状 BCHW，均值 0 方差 1", kind="tensor", rank=[4], layout="BCHW"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        num_features = int(params.get("num_features", in_specs["x"].shape[1]))
        op = nn.BatchNorm2d(
            num_features=num_features,
            eps=float(params.get("eps", 1e-5)),
            momentum=float(params.get("momentum", 0.1)),
            affine=bool(params.get("affine", True)),
        )
        return _SingleModuleWrapper(op, out_layout="BCHW")


@register_component("layer_norm")
class LayerNormLayer(BaseComponent):
    """层归一化元件（对最后一维归一化，适配 BD / BLD）。"""

    category = "op"
    display_name = "LayerNorm"
    title = "层归一化"
    description = "层归一化（适配 BD / BLD）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("eps", "float", 1e-5, min=1e-8, max=1e-2, step=1e-5, description="数值稳定项"),
            ParamSpec("elementwise_affine", "bool", True, description="是否可学习缩放平移"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入张量", description="BD 或 BLD，对最后一维归一化", kind="tensor", rank=[2, 3], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="out", title="归一化输出", description="同形状张量", kind="tensor", rank=[2, 3], layout="ANY"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        norm_shape = int(params.get("normalized_shape", in_specs["x"].shape[-1]))
        op = nn.LayerNorm(
            normalized_shape=norm_shape,
            eps=float(params.get("eps", 1e-5)),
            elementwise_affine=bool(params.get("elementwise_affine", True)),
        )
        return _SingleModuleWrapper(op)


@register_component("dropout")
class DropoutLayer(BaseComponent):
    """Dropout 正则化层。"""

    category = "op"
    display_name = "Dropout"
    title = "随机失活 Dropout"
    description = "训练期随机失活正则化层"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("p", "float", 0.1, min=0.0, max=0.95, step=0.05, description="丢弃概率"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="输入张量", description="任意 2~4 维张量", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
            outputs=[
                PortSpec(name="out", title="失活输出", description="训练期随机置零并缩放的同形状张量", kind="tensor", rank=[2, 3, 4], layout="ANY"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        p = float(params.get("p", 0.1))
        return _SingleModuleWrapper(nn.Dropout(p=p))
