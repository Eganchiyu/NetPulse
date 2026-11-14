"""原子级神经网络层元件：Conv2d / Pool2d / Linear / Activation / Norm / Dropout。"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from netpulse.components.init_utils import apply_init
from netpulse.core.component import BaseComponent
from netpulse.core.registry import register_component
from netpulse.core.types import ComponentSignature, PortSpec, ResolvedSpec, Signal


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

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[4], layout="BCHW")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[4], layout="BCHW")],
        )

    def expected_input_spec(
        self,
        params: dict[str, Any],
        port_name: str,
        actual: ResolvedSpec,
        resolved_in_specs: dict[str, ResolvedSpec],
    ) -> Optional[ResolvedSpec]:
        if port_name == "x" and "in_channels" in params:
            c = int(params["in_channels"])
            b, _, h, w = actual.shape
            return ResolvedSpec(shape=(b, c, h, w), dtype=actual.dtype, layout="BCHW", has_mask=actual.has_mask)
        return None

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        in_channels = int(params.get("in_channels", in_specs["x"].shape[1]))
        out_channels = int(params["out_channels"])
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
        apply_init(conv.weight, conv.bias, params.get("init"), params.get("bias_init"))
        return _SingleModuleWrapper(conv, out_layout="BCHW")


@register_component("pool2d")
class Pool2dLayer(BaseComponent):
    """原子 2D 池化层：支持 max / avg 模式与核尺寸、步长、填充配置。"""

    category = "op"

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[4], layout="BCHW")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[4], layout="BCHW")],
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

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[2, 3], layout="ANY")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2, 3], layout="ANY")],
        )

    def expected_input_spec(
        self,
        params: dict[str, Any],
        port_name: str,
        actual: ResolvedSpec,
        resolved_in_specs: dict[str, ResolvedSpec],
    ) -> Optional[ResolvedSpec]:
        if port_name == "x" and "in_features" in params:
            d = int(params["in_features"])
            new_shape = actual.shape[:-1] + (d,)
            return ResolvedSpec(shape=new_shape, dtype=actual.dtype, layout=actual.layout, has_mask=actual.has_mask)
        return None

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        in_features = int(params.get("in_features", in_specs["x"].shape[-1]))
        out_features = int(params["out_features"])
        bias = bool(params.get("bias", True))
        linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        apply_init(linear.weight, linear.bias, params.get("init"), params.get("bias_init"))
        return _SingleModuleWrapper(linear)


@register_component("activation")
class ActivationLayer(BaseComponent):
    """原子激活函数层。"""

    category = "op"

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[2, 3, 4], layout="ANY")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2, 3, 4], layout="ANY")],
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

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[4], layout="BCHW")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[4], layout="BCHW")],
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

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[2, 3], layout="ANY")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2, 3], layout="ANY")],
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

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[2, 3, 4], layout="ANY")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2, 3, 4], layout="ANY")],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        p = float(params.get("p", 0.1))
        return _SingleModuleWrapper(nn.Dropout(p=p))
