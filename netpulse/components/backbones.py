"""主干网络宏元件：增强版 CNNBlock 与 MLPBlock（支持逐层参数与统一初始化）。"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from netpulse.components.init_utils import apply_init
from netpulse.core.component import BaseComponent
from netpulse.core.registry import register_component
from netpulse.core.types import ComponentSignature, ParamSpec, PortSpec, ResolvedSpec, Signal, TapSpec


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _normalize_cnn_layer_cfgs(params: dict[str, Any]) -> list[dict[str, Any]]:
    if "layers" in params:
        return [dict(item) for item in params["layers"]]

    channels = [int(c) for c in params.get("channels", [16, 32])]
    n = len(channels)

    def _expand(val: Any, default: Any) -> list[Any]:
        v = default if val is None else val
        if isinstance(v, list):
            if len(v) != n:
                raise ValueError(f"CNNBlock 参数列表长度 {len(v)} 与 channels 长度 {n} 不一致")
            return v
        return [v] * n

    ks_list = _expand(params.get("kernel_size"), 3)
    stride_list = _expand(params.get("stride"), 1)
    pad_list = _expand(params.get("padding"), None)
    act_list = _expand(params.get("activation"), "relu")
    pool_every = int(params.get("pool_every", 1))
    pool_type = str(params.get("pool_type", "max")).lower()
    pool_size = int(params.get("pool_size", 2))

    cfgs: list[dict[str, Any]] = []
    for i, c_out in enumerate(channels):
        k = int(ks_list[i])
        pad = k // 2 if pad_list[i] is None else int(pad_list[i])
        use_pool = pool_every > 0 and (i + 1) % pool_every == 0
        cfgs.append(
            {
                "out_channels": c_out,
                "kernel_size": k,
                "stride": int(stride_list[i]),
                "padding": pad,
                "activation": act_list[i],
                "pool": pool_type if use_pool else "none",
                "pool_size": pool_size,
            }
        )
    return cfgs


class _CNNModule(nn.Module):
    """通过 PyTorch Forward Hook 支持任意层 Tap 抽取的卷积主干。"""

    def __init__(
        self,
        in_channels: int,
        layer_cfgs: list[dict[str, Any]],
        weight_init: Any = None,
        bias_init: Any = None,
    ) -> None:
        super().__init__()
        layers = []
        c_in = in_channels
        for cfg in layer_cfgs:
            c_out = int(cfg["out_channels"])
            k = int(cfg.get("kernel_size", 3))
            stride = int(cfg.get("stride", 1))
            pad = int(cfg.get("padding", k // 2))
            conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=pad)
            apply_init(
                conv.weight,
                conv.bias,
                cfg.get("init", weight_init),
                cfg.get("bias_init", bias_init),
            )
            block_mods: list[nn.Module] = [
                conv,
                _make_activation(str(cfg.get("activation", "relu"))),
            ]
            pool_kind = str(cfg.get("pool", "none")).lower()
            pool_size = int(cfg.get("pool_size", 2))
            pool_stride = int(cfg.get("pool_stride", pool_size))
            if pool_kind == "max":
                block_mods.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride))
            elif pool_kind == "avg":
                block_mods.append(nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride))
            elif pool_kind != "none":
                raise ValueError(f"Unsupported pool kind in CNNBlock: {pool_kind}")

            layers.append(nn.Sequential(*block_mods))
            c_in = c_out
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        x_sig = inputs["x"]
        h = x_sig.data
        for layer in self.layers:
            h = layer(h)
        return {"out": x_sig.clone_with(data=h, layout="BCHW")}


@register_component("cnn_block")
class CNNBlock(BaseComponent):
    """可配置多层 2D CNN 宏元件，支持逐层核/池化配置与初始化，暴露每层 Tap。"""

    category = "backbone"
    display_name = "CNN Block"
    description = "多层 2D CNN 宏模块（每层暴露 Hook Tap）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("channels", "json", [16, 32], description="各层输出通道列表"),
            ParamSpec("kernel_size", "json", 3, description="卷积核尺寸（标量或列表）"),
            ParamSpec("stride", "json", 1, description="步长（标量或列表）"),
            ParamSpec("activation", "select", "relu", options=["relu", "gelu", "silu", "tanh", "none"], description="激活函数"),
            ParamSpec("pool_every", "int", 1, min=0, max=8, step=1, description="每隔多少层池化（0关闭）"),
            ParamSpec("pool_type", "select", "max", options=["max", "avg"], description="池化类型"),
            ParamSpec("pool_size", "int", 2, min=1, max=4, step=1, description="池化窗口"),
            ParamSpec("in_channels", "int", None, min=1, max=2048, step=1, description="可选：强制输入通道校验"),
            ParamSpec("init", "select", None, options=["", "kaiming_normal", "xavier_normal", "orthogonal"], description="权重初始化"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        num_layers = len(params["layers"]) if "layers" in params else len(params.get("channels", [16, 32]))
        taps = [
            TapSpec(
                name=f"layer_{i}",
                source=f"layers.{i}",
                spec=PortSpec(name=f"layer_{i}", kind="tensor", rank=[4], layout="BCHW"),
            )
            for i in range(num_layers)
        ]
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[4], layout="BCHW")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[4], layout="BCHW")],
            taps=taps,
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
        layer_cfgs = _normalize_cnn_layer_cfgs(params)
        return _CNNModule(
            in_channels=in_channels,
            layer_cfgs=layer_cfgs,
            weight_init=params.get("init") or None,
            bias_init=params.get("bias_init") or None,
        )


class _MLPModule(nn.Module):
    """通过 forward_with_taps 约定返回支持任意层 Tap 抽取的 MLP。"""

    def __init__(
        self,
        in_dim: int,
        dims: list[int],
        activation: str,
        activate_last: bool,
        dropout: float,
        weight_init: Any = None,
        bias_init: Any = None,
    ) -> None:
        super().__init__()
        blocks = []
        d_in = in_dim
        for i, d_out in enumerate(dims):
            is_last = i == len(dims) - 1
            linear = nn.Linear(d_in, d_out)
            apply_init(linear.weight, linear.bias, weight_init, bias_init)
            sub: list[nn.Module] = [linear]
            if not is_last or activate_last:
                sub.append(_make_activation(activation))
                if dropout > 0.0:
                    sub.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*sub))
            d_in = d_out
        self.layers = nn.ModuleList(blocks)

    def forward_with_taps(
        self, inputs: dict[str, Signal]
    ) -> tuple[dict[str, Signal], dict[str, Signal]]:
        x_sig = inputs["x"]
        h = x_sig.data
        taps: dict[str, Signal] = {}
        for i, layer in enumerate(self.layers):
            h = layer(h)
            taps[f"layer_{i}"] = x_sig.clone_with(data=h)
        return {"out": x_sig.clone_with(data=h)}, taps

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        out, _ = self.forward_with_taps(inputs)
        return out


@register_component("mlp_block")
class MLPBlock(BaseComponent):
    """可配置多层 MLP 宏元件（支持 BD 与 BLD 布局、统一初始化与 forward_with_taps）。"""

    category = "backbone"
    display_name = "MLP Block"
    description = "多层全连接宏模块（支持 BD / BLD 与每层 Tap 抽头）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("dims", "json", [64, 10], description="各层输出维度列表"),
            ParamSpec("activation", "select", "relu", options=["relu", "gelu", "silu", "tanh", "none"], description="激活函数"),
            ParamSpec("activate_last", "bool", False, description="最后一层是否加激活"),
            ParamSpec("dropout", "float", 0.0, min=0.0, max=0.9, step=0.05, description="层间 Dropout"),
            ParamSpec("in_dim", "int", None, min=1, max=8192, step=1, description="可选：强制输入维度校验"),
            ParamSpec("init", "select", None, options=["", "kaiming_normal", "xavier_normal", "orthogonal"], description="权重初始化"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        dims = params.get("dims", [64, 10])
        taps = [
            TapSpec(
                name=f"layer_{i}",
                source=f"layers.{i}",
                spec=PortSpec(name=f"layer_{i}", kind="tensor", rank=[2, 3], layout="ANY"),
            )
            for i in range(len(dims))
        ]
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[2, 3], layout="ANY")],
            outputs=[PortSpec(name="out", kind="tensor", rank=[2, 3], layout="ANY")],
            taps=taps,
        )

    def expected_input_spec(
        self,
        params: dict[str, Any],
        port_name: str,
        actual: ResolvedSpec,
        resolved_in_specs: dict[str, ResolvedSpec],
    ) -> Optional[ResolvedSpec]:
        if port_name == "x" and params.get("in_dim") is not None:
            d = int(params["in_dim"])
            new_shape = actual.shape[:-1] + (d,)
            return ResolvedSpec(shape=new_shape, dtype=actual.dtype, layout=actual.layout, has_mask=actual.has_mask)
        return None

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        raw_in = params.get("in_dim")
        in_dim = int(raw_in if raw_in is not None else in_specs["x"].shape[-1])
        dims = [int(d) for d in params.get("dims", [64, 10])]
        activation = str(params.get("activation", "relu"))
        activate_last = bool(params.get("activate_last", False))
        dropout = float(params.get("dropout", 0.0))
        return _MLPModule(
            in_dim=in_dim,
            dims=dims,
            activation=activation,
            activate_last=activate_last,
            dropout=dropout,
            weight_init=params.get("init") or None,
            bias_init=params.get("bias_init") or None,
        )
