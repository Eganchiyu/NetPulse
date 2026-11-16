"""GraphIR：计算图 JSON / YAML 唯一事实源。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class NodeIR(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    position: Optional[dict[str, float]] = None
    # 用户自定义显示标签（留空则用元件中文名）
    label: Optional[str] = None


class EdgeIR(BaseModel):
    id: str
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str
    edge_kind: Literal["signal", "tap", "loss"] = "signal"
    bridge: Literal["auto", "none"] = "none"
    # 梯度阻断：该边传递的信号 detach，反向传播不穿过此边（特征干预 / 无侵入表征评估）
    stop_grad: bool = False
    # 梯度观测：编译期在该边注册反向 Hook，梯度张量写入 CompiledGraph.grad_bus
    watch_grad: bool = False
    # AutoBridge 权重与偏置初始化策略（传给 init_utils.apply_init）
    bridge_init: Optional[str | dict[str, Any]] = None
    bridge_bias_init: Optional[str | dict[str, Any]] = None


class SubgraphPortMapping(BaseModel):
    """子图边界端口映射：将外层端口绑定到内部节点端口或抽头。"""

    port_name: str
    internal_node: str
    internal_port: str
    edge_kind: Literal["signal", "tap"] = "signal"
    title: Optional[str] = None
    description: Optional[str] = None


class SubgraphDef(BaseModel):
    """可复用的嵌套子图定义（随 GraphIR 保存）。"""

    id: str
    name: str = "Subgraph"
    title: Optional[str] = None
    description: Optional[str] = None
    nodes: list[NodeIR] = Field(default_factory=list)
    edges: list[EdgeIR] = Field(default_factory=list)
    losses: list[LossConfig] = Field(default_factory=list)
    inputs: list[SubgraphPortMapping] = Field(default_factory=list)
    outputs: list[SubgraphPortMapping] = Field(default_factory=list)
    taps: list[SubgraphPortMapping] = Field(default_factory=list)


class LossConfig(BaseModel):
    node_id: str
    weight: float = 1.0
    enabled: bool = True


class TrainerConfig(BaseModel):
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 3
    max_steps: Optional[int] = None
    grad_clip: Optional[float] = 1.0
    device: str = "cpu"
    seed: int = 42
    eval_every_steps: int = 50
    infer_outputs: list[str] = Field(default_factory=list)


class GraphIR(BaseModel):
    name: str = "untitled_graph"
    nodes: list[NodeIR]
    edges: list[EdgeIR]
    losses: list[LossConfig] = Field(default_factory=list)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    subgraphs: dict[str, SubgraphDef] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "GraphIR":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.model_validate(data)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_file(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in {".yaml", ".yml"}:
            text = yaml.safe_dump(self.model_dump(), allow_unicode=True, sort_keys=False)
        else:
            text = self.to_json(indent=2) + "\n"
        p.write_text(text, encoding="utf-8")
