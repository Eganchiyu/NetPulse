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


class EdgeIR(BaseModel):
    id: str
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str
    edge_kind: Literal["signal", "tap", "loss"] = "signal"
    bridge: Literal["auto", "none"] = "none"


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
