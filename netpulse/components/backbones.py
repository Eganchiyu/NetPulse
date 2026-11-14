"""主干网络宏元件：增强版 CNNBlock 与 MLPBlock（支持逐层参数与统一初始化）。"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class _LSTMModule(nn.Module):
    """支持变长 Mask 与细粒度门控 Tap 抽头的多层步进 LSTM。"""

    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        forget_bias: float = 1.0,
        weight_init: Any = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forget_bias = forget_bias

        cells: list[nn.LSTMCell] = []
        for layer_idx in range(num_layers):
            d_in = in_dim if layer_idx == 0 else hidden_size
            cell = nn.LSTMCell(d_in, hidden_size)
            if weight_init:
                apply_init(cell.weight_ih, cell.bias_ih, weight_init, None)
                apply_init(cell.weight_hh, cell.bias_hh, "orthogonal" if weight_init == "orthogonal" else weight_init, None)
            if forget_bias != 0.0 and cell.bias_ih is not None and cell.bias_hh is not None:
                with torch.no_grad():
                    # LSTMCell 门控顺序为 i, f, g, o
                    cell.bias_ih[hidden_size : 2 * hidden_size].fill_(forget_bias * 0.5)
                    cell.bias_hh[hidden_size : 2 * hidden_size].fill_(forget_bias * 0.5)
            cells.append(cell)

        self.cells = nn.ModuleList(cells)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 and num_layers > 1 else nn.Identity()

    def _run(
        self,
        inputs: dict[str, Signal],
        collect_taps: bool,
    ) -> tuple[dict[str, Signal], dict[str, Signal]]:
        x_sig = inputs["x"]
        x = x_sig.data
        mask = x_sig.mask.bool() if x_sig.mask is not None else None
        b, seq_len, _ = x.shape

        seq_input = x
        last_h = torch.zeros((b, self.hidden_size), device=x.device, dtype=x.dtype)
        last_c = torch.zeros((b, self.hidden_size), device=x.device, dtype=x.dtype)

        tap_i: list[torch.Tensor] = []
        tap_f: list[torch.Tensor] = []
        tap_o: list[torch.Tensor] = []
        tap_c: list[torch.Tensor] = []
        tap_h: list[torch.Tensor] = []

        for layer_idx, cell in enumerate(self.cells):
            is_last_layer = layer_idx == self.num_layers - 1
            h = torch.zeros((b, self.hidden_size), device=x.device, dtype=x.dtype)
            c = torch.zeros((b, self.hidden_size), device=x.device, dtype=x.dtype)
            layer_outs: list[torch.Tensor] = []

            for t in range(seq_len):
                x_t = seq_input[:, t, :]
                gates = F.linear(x_t, cell.weight_ih, cell.bias_ih) + F.linear(h, cell.weight_hh, cell.bias_hh)
                i_g, f_g, g_g, o_g = gates.chunk(4, dim=-1)
                i_t = torch.sigmoid(i_g)
                f_t = torch.sigmoid(f_g)
                g_t = torch.tanh(g_g)
                o_t = torch.sigmoid(o_g)

                c_new = f_t * c + i_t * g_t
                h_new = o_t * torch.tanh(c_new)

                if mask is not None:
                    m_t = mask[:, t].unsqueeze(-1)
                    c = torch.where(m_t, c_new, c)
                    h = torch.where(m_t, h_new, h)
                else:
                    c, h = c_new, h_new

                layer_outs.append(h)

                if collect_taps and is_last_layer:
                    tap_i.append(i_t)
                    tap_f.append(f_t)
                    tap_o.append(o_t)
                    tap_c.append(c)
                    tap_h.append(h)

            seq_out = torch.stack(layer_outs, dim=1)
            if not is_last_layer:
                seq_input = self.drop(seq_out)
            else:
                seq_input = seq_out
                last_h = h
                last_c = c

        out_meta = {**x_sig.meta, "layout": "BLD"}
        bd_meta = {**x_sig.meta, "layout": "BD"}
        outputs = {
            "out": Signal(data=seq_input, mask=x_sig.mask, meta=out_meta),
            "last_h": Signal(data=last_h, mask=None, meta=bd_meta),
            "last_c": Signal(data=last_c, mask=None, meta=bd_meta),
        }
        taps: dict[str, Signal] = {}
        if collect_taps:
            taps = {
                "gate_i": Signal(data=torch.stack(tap_i, dim=1), mask=x_sig.mask, meta=out_meta),
                "gate_f": Signal(data=torch.stack(tap_f, dim=1), mask=x_sig.mask, meta=out_meta),
                "gate_o": Signal(data=torch.stack(tap_o, dim=1), mask=x_sig.mask, meta=out_meta),
                "cell_state": Signal(data=torch.stack(tap_c, dim=1), mask=x_sig.mask, meta=out_meta),
                "hidden_states": Signal(data=torch.stack(tap_h, dim=1), mask=x_sig.mask, meta=out_meta),
            }
        return outputs, taps

    def forward_with_taps(
        self, inputs: dict[str, Signal]
    ) -> tuple[dict[str, Signal], dict[str, Signal]]:
        return self._run(inputs, collect_taps=True)

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        outputs, _ = self._run(inputs, collect_taps=False)
        return outputs


@register_component("lstm")
class LSTMLayer(BaseComponent):
    """可观测门控 LSTM 模块（支持 Mask 感知与 i/f/o/c/h 细粒度 Tap 抽头）。"""

    category = "backbone"
    display_name = "LSTM"
    description = "门控长短期记忆网络（支持 Mask、last_h/c 输出与门控序列抽头）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("hidden_size", "int", 32, required=True, min=1, max=1024, step=1, description="隐藏状态维度 H"),
            ParamSpec("num_layers", "int", 1, min=1, max=6, step=1, description="堆叠层数"),
            ParamSpec("dropout", "float", 0.0, min=0.0, max=0.9, step=0.05, description="多层间 Dropout"),
            ParamSpec("forget_bias", "float", 1.0, min=-5.0, max=5.0, step=0.5, description="遗忘门初始偏置（建议 >0 利长程记忆）"),
            ParamSpec("in_dim", "int", None, min=1, max=8192, step=1, description="可选：强制输入特征维校验"),
            ParamSpec(
                "init",
                "select",
                "orthogonal",
                options=["", "orthogonal", "xavier_normal", "kaiming_normal"],
                description="权重初始化",
            ),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        taps = [
            TapSpec(name="gate_i", source="cells", spec=PortSpec(name="gate_i", kind="tensor", rank=[3], layout="BLD")),
            TapSpec(name="gate_f", source="cells", spec=PortSpec(name="gate_f", kind="tensor", rank=[3], layout="BLD")),
            TapSpec(name="gate_o", source="cells", spec=PortSpec(name="gate_o", kind="tensor", rank=[3], layout="BLD")),
            TapSpec(name="cell_state", source="cells", spec=PortSpec(name="cell_state", kind="tensor", rank=[3], layout="BLD")),
            TapSpec(name="hidden_states", source="cells", spec=PortSpec(name="hidden_states", kind="tensor", rank=[3], layout="BLD")),
        ]
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="tensor", rank=[3], layout="BLD")],
            outputs=[
                PortSpec(name="out", kind="tensor", rank=[3], layout="BLD"),
                PortSpec(name="last_h", kind="tensor", rank=[2], layout="BD"),
                PortSpec(name="last_c", kind="tensor", rank=[2], layout="BD"),
            ],
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
        return _LSTMModule(
            in_dim=in_dim,
            hidden_size=int(params.get("hidden_size", 32)),
            num_layers=int(params.get("num_layers", 1)),
            dropout=float(params.get("dropout", 0.0)),
            forget_bias=float(params.get("forget_bias", 1.0)),
            weight_init=params.get("init") or None,
        )
