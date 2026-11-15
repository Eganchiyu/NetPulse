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
    title = "通用监视器"
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
            inputs=[
                PortSpec(name="x", title="被观察数据", description="任意张量，按 mode 自动选择可视化方式", kind="any", rank=None, layout="ANY"),
            ],
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
    title = "标量曲线"
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
            inputs=[
                PortSpec(name="x", title="标量或张量", description="标量，或按 reduce 归约成标量的任意张量", kind="any", rank=None, layout="ANY"),
            ],
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
    title = "热力图预览"
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
            inputs=[
                PortSpec(name="x", title="热力图数据", description="2D 矩阵、BCHW 特征图或可展平张量", kind="any", rank=None, layout="ANY"),
            ],
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
    title = "张量切片查看"
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
            inputs=[
                PortSpec(name="x", title="待查看张量", description="任意张量，按行/列切片并绘制数值直方图", kind="any", rank=None, layout="ANY"),
            ],
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


def _tensor_basic_stats(tf: torch.Tensor) -> dict[str, float]:
    finite = tf[torch.isfinite(tf)]
    if finite.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "norm": 0.0}
    return {
        "min": round(_safe_float(float(finite.min().item())), 5),
        "max": round(_safe_float(float(finite.max().item())), 5),
        "mean": round(_safe_float(float(finite.mean().item())), 5),
        "std": round(_safe_float(float(finite.std(unbiased=False).item())), 5) if finite.numel() > 1 else 0.0,
        "norm": round(_safe_float(float(finite.norm().item())), 5),
    }


def _valid_seq_slice(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    sample_idx: int,
    max_steps: int,
) -> tuple[torch.Tensor, int, int]:
    """抽取单样本序列 (L_eff, D)，感知 Mask 截断并按需等距下采样。"""
    b_idx = max(0, min(tensor.shape[0] - 1, sample_idx))
    seq = tensor[b_idx].float()
    total_len = int(seq.shape[0])
    valid_len = total_len
    if mask is not None and mask.ndim >= 2:
        m = mask[b_idx].bool()
        cnt = int(m.sum().item())
        if 0 < cnt <= total_len:
            valid_len = cnt
            seq = seq[:valid_len]

    if seq.shape[0] > max_steps and max_steps >= 2:
        idx = torch.linspace(0, seq.shape[0] - 1, steps=max_steps, device=seq.device).long()
        seq = seq.index_select(0, idx)
    return seq, valid_len, total_len


class _GateDynamicsModule(nn.Module):
    """LSTM 多门控时序演化与饱和度分析探针。"""

    def __init__(
        self,
        sample_idx: int = 0,
        max_steps: int = 64,
        saturation_thresh: float = 0.1,
    ) -> None:
        super().__init__()
        self.sample_idx = sample_idx
        self.max_steps = max_steps
        self.saturation_thresh = saturation_thresh
        self._last_gates: dict[str, torch.Tensor] = {}
        self._last_mask: Optional[torch.Tensor] = None

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        captured: dict[str, torch.Tensor] = {}
        found_mask: Optional[torch.Tensor] = None
        for key in ("i", "f", "o", "c"):
            sig = inputs.get(key)
            if sig is not None and sig.data.device.type != "meta":
                captured[key] = sig.data.detach()
                if found_mask is None and sig.mask is not None:
                    found_mask = sig.mask.detach()
        if captured:
            self._last_gates = captured
            self._last_mask = found_mask
        return {}

    def capture_snapshot(self) -> Optional[dict[str, Any]]:
        if not self._last_gates:
            return None

        with torch.no_grad():
            mask = self._last_mask.cpu() if self._last_mask is not None else None
            first_key = next(iter(self._last_gates))
            first_raw = self._last_gates[first_key].cpu()
            tf_first = first_raw.float()
            stats = _tensor_basic_stats(tf_first)

            gates_payload: dict[str, Any] = {}
            valid_len_out = int(tf_first.shape[1]) if tf_first.ndim >= 2 else 1
            total_len_out = valid_len_out
            steps_out = 0

            strip_rows: list[list[int]] = []
            strip_labels: list[str] = []

            for key in ("i", "f", "o", "c"):
                if key not in self._last_gates:
                    continue
                t = self._last_gates[key].cpu()
                seq, valid_len, total_len = _valid_seq_slice(t, mask, self.sample_idx, self.max_steps)
                valid_len_out = valid_len
                total_len_out = total_len
                steps_out = int(seq.shape[0])

                mean_c = seq.mean(dim=-1)
                std_c = seq.std(dim=-1, unbiased=False) if seq.shape[-1] > 1 else torch.zeros_like(mean_c)
                sat_low = float((seq < self.saturation_thresh).float().mean().item()) if key != "c" else 0.0
                sat_high = float((seq > (1.0 - self.saturation_thresh)).float().mean().item()) if key != "c" else 0.0

                # 压缩出最多 4 条代表通道的时序条带 (steps,) -> [0..255]
                sub_h = seq[:, : min(4, seq.shape[1])].T  # (K, L)
                q = _quantize_2d(sub_h, symmetric=(key == "c"))
                cols = q["cols"]
                for r in range(q["rows"]):
                    strip_rows.append(q["data"][r * cols : (r + 1) * cols])
                    strip_labels.append(key if r == 0 else "")

                gates_payload[key] = {
                    "mean": [round(_safe_float(float(v)), 4) for v in mean_c.tolist()],
                    "std": [round(_safe_float(float(v)), 4) for v in std_c.tolist()],
                    "sat_low": round(sat_low, 4),
                    "sat_high": round(sat_high, 4),
                    "abs_mean": round(_safe_float(float(seq.abs().mean().item())), 4),
                    "min": round(_safe_float(float(seq.min().item())), 4),
                    "max": round(_safe_float(float(seq.max().item())), 4),
                }

            return {
                "probe_kind": "gate_dynamics",
                "view": "gate_dynamics",
                "shape": list(first_raw.shape),
                "dtype": str(first_raw.dtype).replace("torch.", ""),
                "layout": "BLD",
                "stats": stats,
                "scalar": stats["mean"],
                "gate_dynamics": {
                    "gates": gates_payload,
                    "steps": steps_out,
                    "valid_len": valid_len_out,
                    "total_len": total_len_out,
                    "strip_rows": strip_rows,
                    "strip_labels": strip_labels,
                },
            }


@register_component("gate_dynamics")
class GateDynamics(BaseComponent):
    """LSTM 门控时序演化分析器：同屏对比 i/f/o/c 门控时间曲线、神经元条带热图与饱和度。"""

    category = "output"
    display_name = "GateDynamics"
    title = "门控动态分析"
    description = "LSTM 多门控时序曲线、条带热图与饱和度分析（可同时接 gate_i/f/o/cell_state）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("sample_idx", "int", 0, min=0, max=256, step=1, description="观察的 Batch 样本索引"),
            ParamSpec("max_steps", "int", 64, min=8, max=256, step=8, description="最大显示时间步数"),
            ParamSpec("saturation_thresh", "float", 0.1, min=0.01, max=0.45, step=0.02, description="饱和判断阈值（<t 或 >1-t）"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="i", title="输入门序列", description="接 LSTM 的 tap:gate_i，BLD", kind="tensor", rank=[3], layout="BLD", optional=True),
                PortSpec(name="f", title="遗忘门序列", description="接 LSTM 的 tap:gate_f，BLD", kind="tensor", rank=[3], layout="BLD", optional=True),
                PortSpec(name="o", title="输出门序列", description="接 LSTM 的 tap:gate_o，BLD", kind="tensor", rank=[3], layout="BLD", optional=True),
                PortSpec(name="c", title="细胞状态序列", description="接 LSTM 的 tap:cell_state，BLD", kind="tensor", rank=[3], layout="BLD", optional=True),
            ],
            outputs=[],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _GateDynamicsModule(
            sample_idx=int(params.get("sample_idx", 0)),
            max_steps=int(params.get("max_steps", 64)),
            saturation_thresh=float(params.get("saturation_thresh", 0.1)),
        )


class _HiddenTrajectoryModule(nn.Module):
    """隐状态 2D 相空间轨迹分析探针（支持 BLD 时间轨迹与 BD Batch 分布）。"""

    def __init__(
        self,
        method: str = "pca",
        sample_idx: int = 0,
        max_points: int = 64,
    ) -> None:
        super().__init__()
        self.method = method
        self.sample_idx = sample_idx
        self.max_points = max_points
        self._last_tensor: Optional[torch.Tensor] = None
        self._last_mask: Optional[torch.Tensor] = None
        self._last_layout: str = "ANY"

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        if sig.data.device.type != "meta":
            self._last_tensor = sig.data.detach()
            self._last_mask = sig.mask.detach() if sig.mask is not None else None
            self._last_layout = sig.layout
        return {}

    def capture_snapshot(self) -> Optional[dict[str, Any]]:
        if self._last_tensor is None:
            return None

        with torch.no_grad():
            raw = self._last_tensor.cpu()
            tf = raw.float()
            stats = _tensor_basic_stats(tf)
            mask = self._last_mask.cpu() if self._last_mask is not None else None
            is_seq = tf.ndim == 3

            if is_seq:
                mat, valid_len, total_len = _valid_seq_slice(tf, mask, self.sample_idx, self.max_points)
            elif tf.ndim == 2:
                n = min(tf.shape[0], self.max_points)
                mat = tf[:n]
                valid_len = int(n)
                total_len = int(tf.shape[0])
            else:
                mat = tf.reshape(tf.shape[0], -1)[: self.max_points] if tf.ndim >= 1 else tf.reshape(1, 1)
                valid_len = int(mat.shape[0])
                total_len = valid_len

            n_pts, dim = mat.shape
            explained_var = 0.0
            if dim == 0 or n_pts == 0:
                coords = torch.zeros((1, 2), dtype=torch.float32)
            elif dim == 1:
                t_axis = torch.linspace(-1.0, 1.0, n_pts).unsqueeze(-1)
                coords = torch.cat([mat, t_axis], dim=-1)
            elif self.method == "pca" and n_pts >= 2 and dim >= 2:
                centered = mat - mat.mean(dim=0, keepdim=True)
                try:
                    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
                    if vh.shape[0] >= 2:
                        coords = centered @ vh[:2].T
                        total_e = float((s ** 2).sum().item())
                        if total_e > 1e-12:
                            explained_var = float(((s[:2] ** 2).sum() / total_e).item())
                    else:
                        coords = torch.cat([centered[:, :1], torch.zeros((n_pts, 1))], dim=-1)
                except Exception:
                    coords = mat[:, :2]
            else:
                coords = mat[:, :2]

            xs = coords[:, 0]
            ys = coords[:, 1]
            xmin, xmax = float(xs.min().item()), float(xs.max().item())
            ymin, ymax = float(ys.min().item()), float(ys.max().item())

            if n_pts >= 2:
                diffs = coords[1:] - coords[:-1]
                step_dists = diffs.norm(dim=-1)
                path_len = float(step_dists.sum().item())
                net_disp = float((coords[-1] - coords[0]).norm().item())
            else:
                path_len = 0.0
                net_disp = 0.0

            points = [
                [round(_safe_float(float(px)), 4), round(_safe_float(float(py)), 4)]
                for px, py in coords.tolist()
            ]

            return {
                "probe_kind": "hidden_trajectory",
                "view": "trajectory",
                "shape": list(raw.shape),
                "dtype": str(raw.dtype).replace("torch.", ""),
                "layout": self._last_layout,
                "stats": stats,
                "scalar": round(_safe_float(path_len), 5),
                "trajectory": {
                    "points": points,
                    "method": self.method,
                    "is_sequence": is_seq,
                    "valid_len": valid_len,
                    "total_len": total_len,
                    "explained_var": round(explained_var, 4),
                    "path_len": round(_safe_float(path_len), 4),
                    "net_disp": round(_safe_float(net_disp), 4),
                    "xmin": round(_safe_float(xmin), 4),
                    "xmax": round(_safe_float(xmax), 4),
                    "ymin": round(_safe_float(ymin), 4),
                    "ymax": round(_safe_float(ymax), 4),
                },
            }


@register_component("hidden_trajectory")
class HiddenTrajectory(BaseComponent):
    """隐状态 2D 相空间轨迹：将 BLD 序列（或 BD 隐向量）投影至 2D 绘制状态迁移轨迹。"""

    category = "output"
    display_name = "HiddenTrajectory"
    title = "隐状态轨迹"
    description = "隐状态 2D 相空间轨迹（PCA 投影展示状态从 t=0 到终点的演化路径）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("method", "select", "pca", options=["pca", "first2"], description="2D 投影方式"),
            ParamSpec("sample_idx", "int", 0, min=0, max=256, step=1, description="3D 序列的样本索引"),
            ParamSpec("max_points", "int", 64, min=8, max=256, step=8, description="最大轨迹点数"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="隐状态序列", description="BLD/BD 隐状态，PCA 投影到 2D 相空间", kind="tensor", rank=[2, 3], layout="ANY"),
            ],
            outputs=[],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _HiddenTrajectoryModule(
            method=str(params.get("method", "pca")),
            sample_idx=int(params.get("sample_idx", 0)),
            max_points=int(params.get("max_points", 64)),
        )


class _StepEnergyModule(nn.Module):
    """时间步响应能量与步间状态跃迁分析探针（BLD）。"""

    def __init__(
        self,
        sample_idx: int = 0,
        max_steps: int = 64,
        norm_mode: str = "l2",
    ) -> None:
        super().__init__()
        self.sample_idx = sample_idx
        self.max_steps = max_steps
        self.norm_mode = norm_mode
        self._last_tensor: Optional[torch.Tensor] = None
        self._last_mask: Optional[torch.Tensor] = None

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        sig = inputs["x"]
        if sig.data.device.type != "meta":
            self._last_tensor = sig.data.detach()
            self._last_mask = sig.mask.detach() if sig.mask is not None else None
        return {}

    def capture_snapshot(self) -> Optional[dict[str, Any]]:
        if self._last_tensor is None:
            return None

        with torch.no_grad():
            raw = self._last_tensor.cpu()
            tf = raw.float()
            stats = _tensor_basic_stats(tf)
            mask = self._last_mask.cpu() if self._last_mask is not None else None
            seq, valid_len, total_len = _valid_seq_slice(tf, mask, self.sample_idx, self.max_steps)

            if self.norm_mode == "l2_mean":
                energy = seq.pow(2).mean(dim=-1).sqrt()
            elif self.norm_mode == "linf":
                energy = seq.abs().max(dim=-1).values
            else:
                energy = seq.norm(dim=-1)

            if seq.shape[0] >= 2:
                delta_tail = (seq[1:] - seq[:-1]).norm(dim=-1)
                delta = torch.cat([torch.zeros(1, dtype=seq.dtype), delta_tail], dim=0)
            else:
                delta = torch.zeros_like(energy)

            peak_idx = int(energy.argmax().item()) if energy.numel() > 0 else 0
            peak_val = float(energy[peak_idx].item()) if energy.numel() > 0 else 0.0
            first_e = float(energy[0].item()) if energy.numel() > 0 else 0.0
            last_e = float(energy[-1].item()) if energy.numel() > 0 else 0.0
            decay_ratio = last_e / max(1e-6, first_e) if first_e > 1e-6 else 1.0

            return {
                "probe_kind": "step_energy",
                "view": "step_energy",
                "shape": list(raw.shape),
                "dtype": str(raw.dtype).replace("torch.", ""),
                "layout": "BLD",
                "stats": stats,
                "scalar": round(_safe_float(peak_val), 5),
                "step_energy": {
                    "energy": [round(_safe_float(float(v)), 4) for v in energy.tolist()],
                    "delta": [round(_safe_float(float(v)), 4) for v in delta.tolist()],
                    "peak_step": peak_idx,
                    "peak_val": round(_safe_float(peak_val), 4),
                    "decay_ratio": round(_safe_float(decay_ratio), 4),
                    "valid_len": valid_len,
                    "total_len": total_len,
                    "norm_mode": self.norm_mode,
                },
            }


@register_component("step_energy")
class StepEnergy(BaseComponent):
    """时间步响应与记忆衰减分析：展示各时间步 L2 能量与步间状态跃迁 ||h_t - h_{t-1}||。"""

    category = "output"
    display_name = "StepEnergy"
    title = "时间步能量分析"
    description = "时间步响应能量与步间跃迁分析（识别关键记忆写入步与衰减）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("sample_idx", "int", 0, min=0, max=256, step=1, description="观察的样本索引"),
            ParamSpec("max_steps", "int", 64, min=8, max=256, step=8, description="最大显示时间步数"),
            ParamSpec(
                "norm_mode",
                "select",
                "l2",
                options=["l2", "l2_mean", "linf"],
                description="每步特征能量度量方式",
            ),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[
                PortSpec(name="x", title="序列输入", description="BLD 序列，逐时间步计算响应能量", kind="tensor", rank=[3], layout="BLD"),
            ],
            outputs=[],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        return _StepEnergyModule(
            sample_idx=int(params.get("sample_idx", 0)),
            max_steps=int(params.get("max_steps", 64)),
            norm_mode=str(params.get("norm_mode", "l2")),
        )
