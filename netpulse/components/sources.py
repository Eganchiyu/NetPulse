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
from netpulse.core.types import ComponentSignature, PortSpec, ResolvedSpec, Signal


class _SourceModule(nn.Module):
    """数据源运行时占位模块：从 batch 中透传信号，支持 Meta 推导与 DataLoader 构造。"""

    def __init__(
        self,
        sample_x_shape: tuple[int, ...],
        sample_y_shape: tuple[int, ...],
        x_layout: str,
        loader_factory,
    ) -> None:
        super().__init__()
        self.sample_x_shape = sample_x_shape
        self.sample_y_shape = sample_y_shape
        self.x_layout = x_layout
        self._loader_factory = loader_factory

    def forward(self, inputs: dict[str, Signal]) -> dict[str, Signal]:
        if "x" in inputs and "y" in inputs:
            return inputs
        # Meta 推导或空输入时返回虚拟符号张量
        dev = "meta" if any(True for _ in []) else "meta"
        x = torch.empty(self.sample_x_shape, dtype=torch.float32, device=dev)
        y = torch.empty(self.sample_y_shape, dtype=torch.int64, device=dev)
        return {
            "x": Signal(data=x, meta={"layout": self.x_layout}),
            "y": Signal(data=y, meta={"layout": "B"}),
        }

    def build_dataloaders(self, batch_size: int, seed: int) -> tuple[DataLoader, Optional[DataLoader]]:
        return self._loader_factory(batch_size=batch_size, seed=seed)


@register_component("spiral_data")
class SpiralData(BaseComponent):
    """内置秒开 2D 螺旋分类合成数据源。"""

    category = "source"

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[],
            outputs=[
                PortSpec(name="x", kind="tensor", rank=[2], layout="BD", dtype="float32"),
                PortSpec(name="y", kind="tensor", rank=[1], layout="B", dtype="int64"),
            ],
        )

    def build(self, params: dict[str, Any], in_specs: dict[str, ResolvedSpec]) -> nn.Module:
        num_classes = int(params.get("num_classes", 3))
        points_per_class = int(params.get("points_per_class", 200))
        noise = float(params.get("noise", 0.15))
        batch_hint = int(params.get("meta_batch_size", 32))

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

    @classmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        return ComponentSignature(
            inputs=[],
            outputs=[
                PortSpec(name="x", kind="tensor", rank=[4], layout="BCHW", dtype="float32"),
                PortSpec(name="y", kind="tensor", rank=[1], layout="B", dtype="int64"),
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
