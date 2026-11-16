"""内置数据源元件：极速合成数据与纯 PyTorch MNIST 加载器。"""

from __future__ import annotations

import gzip
import math
import struct
import urllib.request
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from netpulse.core.component import BaseComponent
from netpulse.core.registry import register_component
from netpulse.core.types import ComponentSignature, ParamSpec, PortSpec, ResolvedSpec, Signal


class _SourceModule(nn.Module):
    """数据源运行时占位模块：从 batch 中透传信号，支持 Meta 推导与 DataLoader 构造。"""

    def __init__(
        self,
        sample_x_shape: tuple[int, ...],
        sample_y_shape: tuple[int, ...],
        x_layout: str,
        loader_factory,
        sample_has_mask: bool = False,
        y_dtype: torch.dtype = torch.int64,
        y_layout: str = "B",
    ) -> None:
        super().__init__()
        self.sample_x_shape = sample_x_shape
        self.sample_y_shape = sample_y_shape
        self.x_layout = x_layout
        self._loader_factory = loader_factory
        self.sample_has_mask = sample_has_mask
        self.y_dtype = y_dtype
        self.y_layout = y_layout

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        if "x" in inputs and "y" in inputs:
            return inputs
        # Meta 推导或空输入时返回虚拟符号张量
        dev = "meta"
        x = torch.empty(self.sample_x_shape, dtype=torch.float32, device=dev)
        y = torch.empty(self.sample_y_shape, dtype=self.y_dtype, device=dev)
        mask = (
            torch.empty(self.sample_x_shape[:2], dtype=torch.bool, device=dev)
            if self.sample_has_mask and len(self.sample_x_shape) >= 2
            else None
        )
        return {
            "x": Signal(data=x, mask=mask, meta={"layout": self.x_layout}),
            "y": Signal(data=y, meta={"layout": self.y_layout}),
        }

    def build_dataloaders(self, batch_size: int, seed: int) -> tuple[DataLoader, Optional[DataLoader]]:
        return self._loader_factory(batch_size=batch_size, seed=seed)


@register_component("spiral_data")
class SpiralData(BaseComponent):
    """内置秒开 2D 螺旋分类合成数据源。"""

    category = "source"
    display_name = "Spiral Data"
    title = "螺旋数据源"
    description = "2D 螺旋多分类合成数据源（秒开，适合快速实验）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("num_classes", "int", 3, min=2, max=20, step=1, description="分类数"),
            ParamSpec("points_per_class", "int", 200, min=20, max=5000, step=20, description="每类样本数"),
            ParamSpec("noise", "float", 0.15, min=0.0, max=1.0, step=0.01, description="角度高斯噪声"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[],
            outputs=[
                PortSpec(name="x", title="特征坐标", description="每行一个二维点 (x1, x2)，形状 BD", kind="tensor", rank=[2], layout="BD", dtype="float32"),
                PortSpec(name="y", title="类别标签", description="整型类别索引，形状 B", kind="tensor", rank=[1], layout="B", dtype="int64"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        num_classes = int(params.get("num_classes", 3))
        points_per_class = int(params.get("points_per_class", 200))
        noise = float(params.get("noise", 0.15))
        batch_hint = int(params.get("meta_batch_size", 64))

        def _factory(batch_size: int, seed: int):
            g = torch.Generator().manual_seed(seed)
            xs, ys = [], []
            for c in range(num_classes):
                r = torch.linspace(0.1, 1.0, points_per_class)
                t = (
                    torch.linspace(c * 4.0, (c + 1) * 4.0, points_per_class)
                    + torch.randn(points_per_class, generator=g) * noise
                )
                x_c = torch.stack([r * torch.sin(t), r * torch.cos(t)], dim=1)
                y_c = torch.full((points_per_class,), c, dtype=torch.int64)
                xs.append(x_c)
                ys.append(y_c)
            all_x = torch.cat(xs, dim=0)
            all_y = torch.cat(ys, dim=0)
            perm = torch.randperm(all_x.shape[0], generator=g)
            all_x, all_y = all_x[perm], all_y[perm]

            split = int(all_x.shape[0] * 0.8)
            train_ds = TensorDataset(all_x[:split], all_y[:split])
            val_ds = TensorDataset(all_x[split:], all_y[split:])
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            )

        return _SourceModule(
            sample_x_shape=(batch_hint, 2),
            sample_y_shape=(batch_hint,),
            x_layout="BD",
            loader_factory=_factory,
        )


_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]


def _download_file(filename: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    last_err: Optional[Exception] = None
    for base in _MNIST_MIRRORS:
        url = base + filename
        try:
            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"无法下载 MNIST 文件 {filename}: {last_err}")


def _read_idx_images(gz_path: Path) -> torch.Tensor:
    with gzip.open(gz_path, "rb") as f:
        raw = f.read()
    magic, num, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Invalid MNIST image magic {magic} in {gz_path}")
    arr = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).clone()
    return arr.view(num, 1, rows, cols).to(torch.float32).div_(255.0)


def _read_idx_labels(gz_path: Path) -> torch.Tensor:
    with gzip.open(gz_path, "rb") as f:
        raw = f.read()
    magic, num = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Invalid MNIST label magic {magic} in {gz_path}")
    arr = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).clone()
    return arr.view(num).to(torch.int64)


@register_component("mnist_data")
class MNISTData(BaseComponent):
    """内置纯 PyTorch 轻量 MNIST 数据源（零 torchvision 依赖，自动缓存）。"""

    category = "source"
    display_name = "MNIST Data"
    title = "MNIST 手写数字"
    description = "手写数字数据集（1×28×28 灰度图，10 分类）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec("subset_train", "int", 1024, min=32, max=60000, step=64, description="训练子集样本数（留空全量）"),
            ParamSpec("subset_val", "int", 256, min=32, max=10000, step=32, description="验证子集样本数（留空全量）"),
            ParamSpec("normalize", "bool", True, description="是否做均值方差标准化"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[],
            outputs=[
                PortSpec(name="x", title="图像批", description="28×28 灰度图，形状 BCHW，已归一化", kind="tensor", rank=[4], layout="BCHW", dtype="float32"),
                PortSpec(name="y", title="数字标签", description="0~9 整型标签，形状 B", kind="tensor", rank=[1], layout="B", dtype="int64"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        data_dir = Path(params.get("data_dir", ".cache/mnist"))
        subset_train = params.get("subset_train")
        subset_val = params.get("subset_val")
        normalize = bool(params.get("normalize", True))
        batch_hint = int(params.get("meta_batch_size", 64))

        def _factory(batch_size: int, seed: int):
            paths = {k: _download_file(v, data_dir) for k, v in _MNIST_FILES.items()}
            x_train = _read_idx_images(paths["train_images"])
            y_train = _read_idx_labels(paths["train_labels"])
            x_val = _read_idx_images(paths["test_images"])
            y_val = _read_idx_labels(paths["test_labels"])

            if normalize:
                x_train = (x_train - 0.1307) / 0.3081
                x_val = (x_val - 0.1307) / 0.3081

            g = torch.Generator().manual_seed(seed)
            if subset_train is not None:
                n = min(int(subset_train), x_train.shape[0])
                idx = torch.randperm(x_train.shape[0], generator=g)[:n]
                x_train, y_train = x_train[idx], y_train[idx]
            if subset_val is not None:
                n = min(int(subset_val), x_val.shape[0])
                idx = torch.randperm(x_val.shape[0], generator=g)[:n]
                x_val, y_val = x_val[idx], y_val[idx]

            return (
                DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
                DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False),
            )

        return _SourceModule(
            sample_x_shape=(batch_hint, 1, 28, 28),
            sample_y_shape=(batch_hint,),
            x_layout="BCHW",
            loader_factory=_factory,
        )


@register_component("seq_wave_data")
class SeqWaveData(BaseComponent):
    """内置秒开合成时序数据源：支持波形频率分类与长程记忆线索（含可选变长 Mask）。"""

    category = "source"
    display_name = "SeqWave Data"
    title = "序列波形数据源"
    description = "合成多维序列分类数据源（波形模式 / 记忆线索，支持变长 Mask）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "task",
                "select",
                "memory_cue",
                options=["memory_cue", "freq_pattern"],
                description="任务类型（memory_cue 突出长程记忆门控；freq_pattern 多频波形分类）",
            ),
            ParamSpec("seq_len", "int", 28, min=8, max=256, step=4, description="最大序列长度 L"),
            ParamSpec("feat_dim", "int", 4, min=1, max=64, step=1, description="每步输入特征维 D"),
            ParamSpec("num_classes", "int", 3, min=2, max=10, step=1, description="分类数"),
            ParamSpec("num_samples", "int", 600, min=64, max=5000, step=50, description="总样本数"),
            ParamSpec("variable_length", "bool", True, description="是否生成随机有效长度并附带 Mask"),
            ParamSpec("noise", "float", 0.12, min=0.0, max=1.0, step=0.02, description="观测高斯噪声"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[],
            outputs=[
                PortSpec(name="x", title="序列输入", description="波形/记忆线索输入，形状 BLD，附带变长 mask", kind="tensor", rank=[3], layout="BLD", dtype="float32"),
                PortSpec(name="y", title="类别标签", description="整型类别索引，形状 B", kind="tensor", rank=[1], layout="B", dtype="int64"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        task = str(params.get("task", "memory_cue"))
        seq_len = int(params.get("seq_len", 28))
        feat_dim = int(params.get("feat_dim", 4))
        num_classes = int(params.get("num_classes", 3))
        num_samples = int(params.get("num_samples", 600))
        variable_length = bool(params.get("variable_length", True))
        noise = float(params.get("noise", 0.12))
        batch_hint = int(params.get("meta_batch_size", 64))

        def _factory(batch_size: int, seed: int):
            g = torch.Generator().manual_seed(seed)
            x = torch.randn((num_samples, seq_len, feat_dim), generator=g) * noise
            y = torch.randint(0, num_classes, (num_samples,), generator=g, dtype=torch.int64)
            mask = torch.ones((num_samples, seq_len), dtype=torch.bool)

            t = torch.linspace(0.0, 2.0 * math.pi, seq_len)
            min_valid = max(6, seq_len // 2)

            for i in range(num_samples):
                cls_id = int(y[i].item())
                valid_len = (
                    int(torch.randint(min_valid, seq_len + 1, (1,), generator=g).item())
                    if variable_length
                    else seq_len
                )
                if task == "memory_cue":
                    # 开头 3~5 步注入类别原型脉冲，中间主要为噪声，结尾给出查询触发脉冲
                    cue_len = min(4, max(2, seq_len // 6))
                    angle = 2.0 * math.pi * cls_id / num_classes
                    x[i, :cue_len, 0] += 1.8 * math.cos(angle)
                    if feat_dim > 1:
                        x[i, :cue_len, 1] += 1.8 * math.sin(angle)
                    if feat_dim > 2:
                        # 第 2 维放门控提示脉冲：开头写提示=+1，结尾读提示=-1
                        x[i, :cue_len, 2] += 1.2
                        x[i, valid_len - 2 : valid_len, 2] -= 1.2
                else:
                    # 多频正余弦与斜坡包络波形
                    freq = 1.0 + cls_id * 1.15
                    phase = float(torch.rand(1, generator=g).item()) * 0.5
                    wave = torch.sin(freq * t + phase)
                    cos_w = torch.cos(freq * t + phase)
                    x[i, :, 0] += wave
                    if feat_dim > 1:
                        x[i, :, 1] += cos_w
                    if feat_dim > 2:
                        x[i, :, 2] += (cls_id / max(1, num_classes - 1) - 0.5) * torch.linspace(0.2, 1.0, seq_len)

                if valid_len < seq_len:
                    x[i, valid_len:] = 0.0
                    mask[i, valid_len:] = False

            split = int(num_samples * 0.8)
            if variable_length:
                train_ds = TensorDataset(x[:split], y[:split], mask[:split])
                val_ds = TensorDataset(x[split:], y[split:], mask[split:])
            else:
                train_ds = TensorDataset(x[:split], y[:split])
                val_ds = TensorDataset(x[split:], y[split:])
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            )

        return _SourceModule(
            sample_x_shape=(batch_hint, seq_len, feat_dim),
            sample_y_shape=(batch_hint,),
            x_layout="BLD",
            loader_factory=_factory,
            sample_has_mask=variable_length,
        )


@register_component("seq_forecast_data")
class SeqForecastData(BaseComponent):
    """经典时间序列滑窗回归预测数据源（多谐波叠加 / 阻尼振荡 / 自回归趋势）。"""

    category = "source"
    display_name = "SeqForecast Data"
    title = "时序预测数据源"
    description = "时间序列滑窗预测数据源（x 为历史窗口 BLD，y 为未来 horizon 步真值 BD）"

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                "pattern",
                "select",
                "sine_mix",
                options=["sine_mix", "damped_osc", "ar_trend"],
                description="信号生成模式（sine_mix 多谐波；damped_osc 阻尼振荡；ar_trend 自回归趋势）",
            ),
            ParamSpec("window_len", "int", 24, min=6, max=128, step=2, description="输入历史窗口长度 L"),
            ParamSpec("horizon", "int", 1, min=1, max=16, step=1, description="预测未来步数（y 维度）"),
            ParamSpec("feat_dim", "int", 3, min=1, max=16, step=1, description="每步通道数 D（第 0 维为目标序列）"),
            ParamSpec("num_samples", "int", 800, min=64, max=5000, step=50, description="样本总数"),
            ParamSpec("noise", "float", 0.05, min=0.0, max=0.5, step=0.01, description="观测高斯噪声"),
        ]

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[],
            outputs=[
                PortSpec(name="x", title="历史窗口", description="过去若干步观测值，形状 BLD", kind="tensor", rank=[3], layout="BLD", dtype="float32"),
                PortSpec(name="y", title="未来真值", description="未来 horizon 步目标值，形状 BD", kind="tensor", rank=[2], layout="BD", dtype="float32"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        pattern = str(params.get("pattern", "sine_mix"))
        window_len = int(params.get("window_len", 24))
        horizon = int(params.get("horizon", 1))
        feat_dim = int(params.get("feat_dim", 3))
        num_samples = int(params.get("num_samples", 800))
        noise = float(params.get("noise", 0.05))
        batch_hint = int(params.get("meta_batch_size", 64))

        def _factory(batch_size: int, seed: int):
            g = torch.Generator().manual_seed(seed)
            total_steps = window_len + horizon
            x = torch.zeros((num_samples, window_len, feat_dim), dtype=torch.float32)
            y = torch.zeros((num_samples, horizon), dtype=torch.float32)

            t_full = torch.linspace(0.0, 4.0 * math.pi, total_steps)
            for i in range(num_samples):
                phase = float(torch.rand(1, generator=g).item()) * 2.0 * math.pi
                freq = 0.8 + 0.8 * float(torch.rand(1, generator=g).item())
                amp = 0.7 + 0.5 * float(torch.rand(1, generator=g).item())

                if pattern == "damped_osc":
                    decay = torch.exp(-0.18 * t_full)
                    base = amp * decay * torch.sin(freq * t_full + phase) + 0.3 * torch.cos(0.5 * t_full)
                elif pattern == "ar_trend":
                    trend = 0.08 * t_full - 0.4
                    base = trend + 0.65 * amp * torch.sin(freq * t_full + phase) + 0.25 * torch.sin(2.1 * freq * t_full)
                else:
                    # sine_mix: 主频 + 二次谐波 + 慢包络
                    base = (
                        amp * torch.sin(freq * t_full + phase)
                        + 0.4 * torch.cos(2.0 * freq * t_full + 0.5 * phase)
                        + 0.2 * torch.sin(0.35 * t_full)
                    )

                series = base + torch.randn(total_steps, generator=g) * noise
                x[i, :, 0] = series[:window_len]
                y[i, :] = base[window_len:]  # 预测干净未来值，稳定收敛

                if feat_dim > 1:
                    # 一阶差分特征（局部斜率）
                    diff = torch.zeros_like(series[:window_len])
                    diff[1:] = series[1:window_len] - series[: window_len - 1]
                    x[i, :, 1] = diff
                if feat_dim > 2:
                    # 归一化绝对位置编码 / 相位线索
                    x[i, :, 2] = torch.linspace(-1.0, 1.0, window_len)
                for d in range(3, feat_dim):
                    x[i, :, d] = torch.cos((d - 1) * freq * t_full[:window_len] + phase)

            # 随机划分训练/验证集
            perm = torch.randperm(num_samples, generator=g)
            x, y = x[perm], y[perm]
            split = int(num_samples * 0.8)
            train_ds = TensorDataset(x[:split], y[:split])
            val_ds = TensorDataset(x[split:], y[split:])
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            )

        return _SourceModule(
            sample_x_shape=(batch_hint, window_len, feat_dim),
            sample_y_shape=(batch_hint, horizon),
            x_layout="BLD",
            loader_factory=_factory,
            sample_has_mask=False,
            y_dtype=torch.float32,
            y_layout="BD",
        )
