"""核心协议与运行时数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import torch

PortKind = Literal["tensor", "loss", "data_batch", "any"]
CategoryKind = Literal["source", "backbone", "op", "bridge", "loss", "output"]
ParamType = Literal["int", "float", "bool", "string", "select", "json"]


@dataclass(frozen=True)
class ParamSpec:
    """元件参数契约（用于 WebUI 动态表单渲染与 Schema 校验）。"""

    name: str
    type: ParamType = "string"
    default: Any = None
    required: bool = False
    options: Optional[list[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "required": self.required,
            "options": list(self.options) if self.options is not None else None,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "description": self.description,
        }


@dataclass
class Signal:
    """统一边载荷，封装张量本体、可选 Mask 与语义元数据。"""

    data: torch.Tensor
    mask: Optional[torch.Tensor] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def layout(self) -> str:
        return str(self.meta.get("layout", "ANY"))

    def clone_with(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **meta_updates: Any,
    ) -> "Signal":
        new_meta = dict(self.meta)
        new_meta.update(meta_updates)
        return Signal(
            data=data,
            mask=self.mask if mask is None else mask,
            meta=new_meta,
        )

    def detached(self) -> "Signal":
        """切断反向传播：数据 detach，同时保留 shape / layout / mask 语义。"""
        mask = self.mask.detach() if self.mask is not None else None
        return Signal(data=self.data.detach(), mask=mask, meta=dict(self.meta))


@dataclass(frozen=True)
class PortSpec:
    """元件静态端口契约。"""

    name: str
    kind: PortKind = "tensor"
    rank: Optional[list[int]] = None
    layout: str = "ANY"
    require_mask: bool = False
    optional: bool = False
    dtype: str = "float32"
    # 中文含义（如「输入特征图」）与简单介绍，供 WebUI 展示
    title: str = ""
    description: str = ""


@dataclass(frozen=True)
class TapSpec:
    """隐层抽头契约。"""

    name: str
    source: str
    spec: PortSpec


@dataclass
class ComponentSignature:
    """元件静态签名（仅由 params 决定，不依赖运行时输入张量）。"""

    inputs: list[PortSpec] = field(default_factory=list)
    outputs: list[PortSpec] = field(default_factory=list)
    taps: list[TapSpec] = field(default_factory=list)

    def input_map(self) -> dict[str, PortSpec]:
        return {p.name: p for p in self.inputs}

    def output_map(self) -> dict[str, PortSpec]:
        return {p.name: p for p in self.outputs}

    def tap_map(self) -> dict[str, TapSpec]:
        return {t.name: t for t in self.taps}


@dataclass(frozen=True)
class ResolvedSpec:
    """经 Meta 推导后解析出的具体端口形状与语义规格。"""

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    layout: str = "ANY"
    has_mask: bool = False

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass
class Diagnostic:
    """ERC 静态检查诊断条目。"""

    level: Literal["error", "warning", "info"]
    code: str
    message: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None


@dataclass
class GraphOutput:
    """CompiledGraph 单次前向输出。"""

    outputs: dict[str, dict[str, Signal]]
    tap_bus: dict[str, Signal]
    loss_dict: dict[str, torch.Tensor]
    total_loss: Optional[torch.Tensor] = None
    # 反向梯度总线：key 为边 id，由 backward 期间触发的 Hook 填充（前向刚返回时为空）
    grad_bus: dict[str, torch.Tensor] = field(default_factory=dict)
