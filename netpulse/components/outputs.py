"""输出与监视工具节点（零梯度旁路探针，支持标量曲线、热力图、图像与原始张量快照）。"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from netpulse.core.component import BaseComponent
from netpulse.core.registry import register_component
from netpulse.core.types import ComponentSignature, ParamSpec, PortSpec, ResolvedSpec, Signal


def _safe_float(val: float) -> float:
    if val != val:  # NaN
        return 0.0
    if val == float("inf"):
        return 1e9
    if val == float("-inf"):
        return -1e9
    return float(val)


def _reduce_scalar(tf: torch.Tensor, reduce_mode: str) -> float:
    if tf.numel() == 0:
        return 0.0
    if tf.ndim == 0 or tf.numel() == 1:
        return round(_safe_float(float(tf.reshape(-1)[0].item())), 6)
    if reduce_mode == "norm":
        val = float(tf.norm().item())
    elif reduce_mode == "max":
        val = float(tf.max().item())
    elif reduce_mode == "min":
        val = float(tf.min().item())
    elif reduce_mode == "std":
        val = float(tf.std(unbiased=False).item()) if tf.numel() > 1 else 0.0
    else:
        val = float(tf.mean().item())
    return round(_safe_float(val), 6)


def _downsample_2d(mat: torch.Tensor, max_size: int) -> torch.Tensor:
    """将 2D 矩阵等距采样至边长不超过 max_size，保留全局结构。"""
    h, w = mat.shape
    if h <= max_size and w <= max_size:
        return mat
    if h > max_size:
        idx_h = torch.linspace(0, h - 1, steps=max_size, device=mat.device).long()
        mat = mat.index_select(0, idx_h)
    if w > max_size:
        idx_w = torch.linspace(0, w - 1, steps=max_size, device=mat.device).long()
        mat = mat.index_select(1, idx_w)
    return mat


def _to_2d_matrix(
    tf: torch.Tensor,
    layout: str,
    sample_idx: int = 0,
    channel: int = 0,
) -> torch.Tensor:
    """将任意维度张量投影为适合热力图绘制的 2D 张量。"""
    if tf.numel() == 0:
        return torch.zeros((1, 1), dtype=torch.float32)
    if tf.ndim == 0:
        return tf.reshape(1, 1)
    if tf.ndim == 1:
        return tf.unsqueeze(0)
    if tf.ndim == 2:
        return tf
    if tf.ndim == 3:
        b = max(0, min(tf.shape[0] - 1, sample_idx))
        return tf[b]
    if tf.ndim == 4:
        b = max(0, min(tf.shape[0] - 1, sample_idx))
        if channel < 0:
            return tf[b].mean(dim=0)
        c = max(0, min(tf.shape[1] - 1, channel))
        return tf[b, c]
    # 5D 及以上：锁定首维 sample_idx 后展平为 2D
    b = max(0, min(tf.shape[0] - 1, sample_idx))
    sub = tf[b]
    return sub.reshape(sub.shape[0], -1)


def _quantize_2d(mat: torch.Tensor, symmetric: bool = False) -> dict[str, Any]:
    finite_mask = torch.isfinite(mat)
    if finite_mask.any():
        vals = mat[finite_mask]
        vmin = float(vals.min().item())
        vmax = float(vals.max().item())
    else:
        vmin, vmax = 0.0, 0.0

    if symmetric:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    span = vmax - vmin
    if span <= 1e-12:
        q = torch.full(mat.shape, 128 if symmetric else 0, dtype=torch.uint8)
    else:
        clean = torch.where(finite_mask, mat, torch.tensor(vmin, dtype=mat.dtype))
        scaled = ((clean - vmin) / span).clamp(0.0, 1.0) * 255.0
        q = scaled.round().to(torch.uint8)

    return {
        "rows": int(mat.shape[0]),
        "cols": int(mat.shape[1]),
        "vmin": round(_safe_float(vmin), 5),
        "vmax": round(_safe_float(vmax), 5),
        "data": q.reshape(-1).tolist(),
    }


def _extract_raw_slice(
    tf: torch.Tensor,
    sample_idx: int = 0,
    max_rows: int = 8,
    max_cols: int = 8,
    precision: int = 4,
) -> dict[str, Any]:
    mat = _to_2d_matrix(tf, layout="ANY", sample_idx=sample_idx, channel=0)
    r = min(mat.shape[0], max_rows)
    c = min(mat.shape[1], max_cols)
    sub = mat[:r, :c].tolist()
    rounded = [[round(_safe_float(float(v)), precision) for v in row] for row in sub]
    return {
        "total_rows": int(mat.shape[0]),
        "total_cols": int(mat.shape[1]),
        "rows": r,
        "cols": c,
        "values": rounded,
    }


def _extract_histogram(tf: torch.Tensor, bins: int = 16) -> dict[str, Any]:
    flat = tf.reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"counts": [0] * bins, "vmin": 0.0, "vmax": 0.0}
    vmin = float(flat.min().item())
    vmax = float(flat.max().item())
    if vmax - vmin <= 1e-12:
        counts = [0] * bins
        counts[bins // 2] = int(flat.numel())
        return {"counts": counts, "vmin": round(vmin, 5), "vmax": round(vmax, 5)}
    hist = torch.histc(flat, bins=bins, min=vmin, max=vmax)
    return {
        "counts": [int(x) for x in hist.tolist()],
        "vmin": round(_safe_float(vmin), 5),
        "vmax": round(_safe_float(vmax), 5),
    }


def _extract_images(
    tf: torch.Tensor,
    sample_idx: int = 0,
    max_samples: int = 4,
    max_hw: int = 64,
) -> Optional[dict[str, Any]]:
    """从 4D (B, C, H, W) 或 3D (C, H, W) 提取可视化图像网格。"""
    if tf.ndim == 3 and tf.shape[0] in (1, 3):
        batch = tf.unsqueeze(0)
    elif tf.ndim == 4:
        batch = tf
    else:
        return None

    b_total, c, h, w = batch.shape
    start = max(0, min(b_total - 1, sample_idx))
    end = min(b_total, start + max_samples)
    sub = batch[start:end]

    if c not in (1, 3):
        sub = sub[:, :1]
        c = 1

    if h > max_hw or w > max_hw:
        scale_h = max(1, h // max_hw)
        scale_w = max(1, w // max_hw)
        sub = sub[:, :, ::scale_h, ::scale_w]
        h, w = sub.shape[-2], sub.shape[-1]

    samples = []
    for img in sub:
        finite = img[torch.isfinite(img)]
        if finite.numel() == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(finite.min().item())
            vmax = float(finite.max().item())
        span = vmax - vmin
        if span <= 1e-12:
            q = torch.zeros_like(img, dtype=torch.uint8)
        else:
            q = (((img - vmin) / span).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        # CHW -> HWC flat
        hwc = q.permute(1, 2, 0).contiguous().reshape(-1).tolist()
        samples.append(hwc)

    return {
        "count": len(samples),
        "channels": int(c),
        "height": int(h),
        "width": int(w),
        "start_idx": int(start),
        "samples": samples,
    }


class _OutputProbeModule(nn.Module):
    """零梯度旁路输出探针模块基类。"""

    def __init__(
        self,
        probe_kind: str,
        mode: str = "auto",
        colormap: str = "viridis",
        max_size: int = 64,
        channel: int = 0,
        sample_idx: int = 0,
        symmetric: bool = False,
        reduce_mode: str = "mean",
        ema_alpha: float = 0.1,
        history_len: int = 120,
        max_rows: int = 8,
        max_cols: int = 8,
        precision: int = 4,
    ) -> None:
        super().__init__()
        self.probe_kind = probe_kind
        self.mode = mode
        self.colormap = colormap
        self.max_size = max_size
        self.channel = channel
        self.sample_idx = sample_idx
        self.symmetric = symmetric
        self.reduce_mode = reduce_mode
        self.ema_alpha = ema_alpha
        self.history_len = history_len
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.precision = precision

        self._last_tensor: Optional[torch.Tensor] = None
        self._last_layout: str = "ANY"

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        if sig.data.device.type != "meta":
            self._last_tensor = sig.data.detach()
            self._last_layout = sig.layout
        return {}

    def capture_snapshot(self) -> Optional[dict[str, Any]]:
        if self._last_tensor is None:
            return None

        with torch.no_grad():
            raw = self._last_tensor.cpu()
            tf = raw.float()
            shape = list(raw.shape)
            dtype_str = str(raw.dtype).replace("torch.", "")

            finite = tf[torch.isfinite(tf)]
            if finite.numel() == 0:
                stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "norm": 0.0}
            else:
                stats = {
                    "min": round(_safe_float(float(finite.min().item())), 5),
                    "max": round(_safe_float(float(finite.max().item())), 5),
                    "mean": round(_safe_float(float(finite.mean().item())), 5),
                    "std": round(_safe_float(float(finite.std(unbiased=False).item())), 5) if finite.numel() > 1 else 0.0,
                    "norm": round(_safe_float(float(finite.norm().item())), 5),
                }

            resolved_view = self.mode
            if resolved_view == "auto":
                if self.probe_kind == "tensor_peek":
                    if tf.ndim == 4 and self._last_layout == "BCHW" and tf.shape[1] in (1, 3):
                        resolved_view = "image"
                    else:
                        resolved_view = "raw"
                elif tf.ndim == 0 or tf.numel() == 1:
                    resolved_view = "scalar"
                elif tf.ndim == 4 and self._last_layout == "BCHW" and tf.shape[1] in (1, 3):
                    resolved_view = "image"
                elif tf.ndim >= 2:
                    resolved_view = "heatmap"
                else:
                    resolved_view = "raw"

            scalar_val = _reduce_scalar(tf, self.reduce_mode)

            # 按需附带热力图、图像、直方图与原始切片
            mat = _to_2d_matrix(tf, self._last_layout, sample_idx=self.sample_idx, channel=self.channel)
            mat_ds = _downsample_2d(mat, self.max_size)
            heatmap = _quantize_2d(mat_ds, symmetric=self.symmetric)
            heatmap["colormap"] = self.colormap
            heatmap["symmetric"] = self.symmetric
            heatmap["orig_shape"] = [int(mat.shape[0]), int(mat.shape[1])]

            images = _extract_images(tf, sample_idx=self.sample_idx) if tf.ndim in (3, 4) else None
            raw_slice = _extract_raw_slice(
                tf,
                sample_idx=self.sample_idx,
                max_rows=self.max_rows,
                max_cols=self.max_cols,
                precision=self.precision,
            )
            hist = _extract_histogram(tf)

            return {
                "probe_kind": self.probe_kind,
                "view": resolved_view,
                "shape": shape,
                "dtype": dtype_str,
                "layout": self._last_layout,
                "stats": stats,
                "scalar": scalar_val,
                "reduce": self.reduce_mode,
                "ema_alpha": self.ema_alpha,
                "history_len": self.history_len,
                "heatmap": heatmap,
                "images": images,
                "raw": raw_slice,
                "hist": hist,
            }


@register_component("monitor")
class Monitor(BaseComponent):
    """通用数据监视器：自适应或手动切换标量曲线、热力图、图像与原始数据视图。"""

    category = "output"
    display_name = "Monitor"
    description = "通用数据监视器（自适应或切换 标量/热力图/图像/原始切片）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "mode",
                "select",
                "auto",
                options=["auto", "scalar", "heatmap", "image", "raw"],
                description="显示模式（auto 按张量维度自动选择）",
            ),
            ParamSpec(
                "colormap",
                "select",
                "viridis",
                options=["viridis", "coolwarm", "magma", "grayscale"],
                description="热力图配色方案",
            ),
            ParamSpec("max_size", "int", 64, min=8, max=128, step=8, description="热力图最大边长"),
            ParamSpec("channel", "int", 0, min=-1, max=512, step=1, description="4D 通道索引（-1 为通道均值）"),
            ParamSpec("sample_idx", "int", 0, min=0, max=256, step=1, description="Batch 样本切片索引"),
            ParamSpec("symmetric", "bool", False, description="热力图是否以 0 为中心对称"),
            ParamSpec(
                "reduce",
                "select",
                "mean",
                options=["mean", "norm", "max", "min", "std"],
                description="多维张量转标量的聚合方式",
            ),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="any", rank=None, layout="ANY")],
            outputs=[],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _OutputProbeModule(
            probe_kind="monitor",
            mode=str(params.get("mode", "auto")),
            colormap=str(params.get("colormap", "viridis")),
            max_size=int(params.get("max_size", 64)),
            channel=int(params.get("channel", 0)),
            sample_idx=int(params.get("sample_idx", 0)),
            symmetric=bool(params.get("symmetric", False)),
            reduce_mode=str(params.get("reduce", "mean")),
        )


@register_component("scalar_watch")
class ScalarWatch(BaseComponent):
    """专用标量/曲线监视器：实时追踪 Loss、标量或聚合统计量变化曲线。"""

    category = "output"
    display_name = "ScalarWatch"
    description = "标量数值与历史折线图（适合 Loss 或任意端口聚合趋势）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "reduce",
                "select",
                "mean",
                options=["mean", "norm", "max", "min", "std"],
                description="非标量输入的降维聚合方式",
            ),
            ParamSpec("ema_alpha", "float", 0.1, min=0.0, max=0.99, step=0.05, description="EMA 平滑系数（0 关闭）"),
            ParamSpec("history_len", "int", 120, min=20, max=300, step=10, description="曲线保留步数"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="any", rank=None, layout="ANY")],
            outputs=[],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _OutputProbeModule(
            probe_kind="scalar_watch",
            mode="scalar",
            reduce_mode=str(params.get("reduce", "mean")),
            ema_alpha=float(params.get("ema_alpha", 0.1)),
            history_len=int(params.get("history_len", 120)),
        )


@register_component("heatmap_preview")
class HeatmapPreview(BaseComponent):
    """专用 2D 矩阵 / 隐层激活 / 特征图热力图预览节点。"""

    category = "output"
    display_name = "HeatmapPreview"
    description = "2D 矩阵 / 激活输出 / 特征图热力图实时渲染"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "colormap",
                "select",
                "viridis",
                options=["viridis", "coolwarm", "magma", "grayscale"],
                description="热力图色标",
            ),
            ParamSpec("max_size", "int", 64, min=8, max=128, step=8, description="热力图最大采样边长"),
            ParamSpec("channel", "int", 0, min=-1, max=512, step=1, description="4D 通道索引（-1 为通道均值）"),
            ParamSpec("sample_idx", "int", 0, min=0, max=256, step=1, description="3D/4D 样本索引"),
            ParamSpec("symmetric", "bool", False, description="是否以 0 为中心对称映射"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="any", rank=None, layout="ANY")],
            outputs=[],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _OutputProbeModule(
            probe_kind="heatmap_preview",
            mode="heatmap",
            colormap=str(params.get("colormap", "viridis")),
            max_size=int(params.get("max_size", 64)),
            channel=int(params.get("channel", 0)),
            sample_idx=int(params.get("sample_idx", 0)),
            symmetric=bool(params.get("symmetric", False)),
        )


@register_component("tensor_peek")
class TensorPeek(BaseComponent):
    """原始数据切片、直方图分布与 4D 图像预览节点。"""

    category = "output"
    display_name = "TensorPeek"
    description = "原始浮点数值切片、直方图与 4D 图像（如 MNIST）预览"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "view",
                "select",
                "auto",
                options=["auto", "raw", "image", "hist"],
                description="默认视图（auto 对 4D BCHW 自动切图像预览）",
            ),
            ParamSpec("sample_idx", "int", 0, min=0, max=256, step=1, description="样本起始索引"),
            ParamSpec("max_rows", "int", 8, min=1, max=16, step=1, description="原始矩阵最大显示行数"),
            ParamSpec("max_cols", "int", 8, min=1, max=16, step=1, description="原始矩阵最大显示列数"),
            ParamSpec("precision", "int", 4, min=1, max=6, step=1, description="浮点显示小数位数"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[PortSpec(name="x", kind="any", rank=None, layout="ANY")],
            outputs=[],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        view = str(params.get("view", "auto"))
        return _OutputProbeModule(
            probe_kind="tensor_peek",
            mode=view,
            sample_idx=int(params.get("sample_idx", 0)),
            max_rows=int(params.get("max_rows", 8)),
            max_cols=int(params.get("max_cols", 8)),
            precision=int(params.get("precision", 4)),
        )
