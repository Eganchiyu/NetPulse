"""Trainer：基于 CompiledGraph 的 PyTorch 训练与评估闭环。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from netpulse.core.compiler import CompiledGraph
from netpulse.core.types import Signal


@dataclass
class TrainResult:
    """训练完成后的摘要指标与历史记录。"""

    steps: int
    history: list[dict[str, float]] = field(default_factory=list)
    final_train_loss: float = 0.0
    final_val_loss: Optional[float] = None
    final_val_acc: Optional[float] = None


def _build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("CompiledGraph 中没有可训练参数")
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _to_signal_batch(
    source_id: str,
    raw_batch: Any,
    device: torch.device,
) -> dict[str, dict[str, Signal]]:
    if isinstance(raw_batch, (tuple, list)) and len(raw_batch) == 2:
        x, y = raw_batch
        x = x.to(device)
        y = y.to(device)
        x_layout = "BCHW" if x.ndim == 4 else ("BD" if x.ndim == 2 else "ANY")
        return {
            source_id: {
                "x": Signal(data=x, meta={"layout": x_layout}),
                "y": Signal(data=y, meta={"layout": "B"}),
            }
        }
    if isinstance(raw_batch, dict):
        moved = {
            k: Signal(data=v.to(device), meta={"layout": "BCHW" if v.ndim == 4 else "ANY"})
            for k, v in raw_batch.items()
        }
        return {source_id: moved}
    raise TypeError(f"Unsupported batch format from source '{source_id}': {type(raw_batch)}")


def _extract_acc(
    compiled: CompiledGraph,
    out_signals: dict[str, dict[str, Signal]],
    batch: dict[str, dict[str, Signal]],
    source_id: str,
) -> Optional[tuple[int, int]]:
    if source_id not in batch or "y" not in batch[source_id]:
        return None
    y = batch[source_id]["y"].data
    if y.dtype not in (torch.int64, torch.int32):
        return None

    node_types = {n.id: n.type for n in compiled.graph_ir.nodes}
    for edge in compiled.graph_ir.edges:
        if node_types.get(edge.dst_node) == "cross_entropy_loss" and edge.dst_port == "logits":
            src_ports = out_signals.get(edge.src_node, {})
            if edge.src_port in src_ports:
                logits = src_ports[edge.src_port].data
                if logits.ndim == 2 and logits.shape[0] == y.shape[0]:
                    preds = logits.argmax(dim=-1)
                    return int((preds == y).sum().item()), int(y.shape[0])
    return None


class Trainer:
    """驱动 CompiledGraph 完成训练、验证与权重导出。"""

    def __init__(self, compiled: CompiledGraph) -> None:
        self.compiled = compiled
        self.cfg = compiled.graph_ir.trainer

    def _find_source_id(self) -> str:
        sources = [nid for nid, cat in self.compiled.categories.items() if cat == "source"]
        if len(sources) != 1:
            raise ValueError(f"Phase 0 Trainer 要求恰好 1 个 source 节点，当前发现: {sources}")
        return sources[0]

    @torch.no_grad()
    def evaluate(self, val_loader, source_id: str, device: torch.device) -> tuple[float, Optional[float]]:
        self.compiled.eval()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total_samples = 0

        for raw_batch in val_loader:
            batch = _to_signal_batch(source_id, raw_batch, device)
            out = self.compiled(batch, mode="eval", compute_losses=True)
            if out.total_loss is not None:
                total_loss += float(out.total_loss.item())
                n_batches += 1
            acc_pair = _extract_acc(self.compiled, out.outputs, batch, source_id)
            if acc_pair is not None:
                correct += acc_pair[0]
                total_samples += acc_pair[1]

        avg_loss = total_loss / max(1, n_batches)
        acc = (correct / total_samples) if total_samples > 0 else None
        return avg_loss, acc

    def fit(self, verbose: bool = True) -> TrainResult:
        torch.manual_seed(self.cfg.seed)
        device = torch.device(self.cfg.device)
        self.compiled.to(device)

        source_id = self._find_source_id()
        source_mod = self.compiled.nodes_mod[source_id]
        train_loader, val_loader = source_mod.build_dataloaders(
            batch_size=self.cfg.batch_size,
            seed=self.cfg.seed,
        )

        optimizer = _build_optimizer(
            self.compiled,
            name=self.cfg.optimizer,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        history: list[dict[str, float]] = []
        step = 0
        last_train_loss = 0.0

        for epoch in range(self.cfg.epochs):
            self.compiled.train()
            for raw_batch in train_loader:
                batch = _to_signal_batch(source_id, raw_batch, device)
                optimizer.zero_grad()
                out = self.compiled(batch, mode="train")
                if out.total_loss is None:
                    raise RuntimeError("训练图未产生任何 Loss，请检查 LossNode 连接与配置")
                out.total_loss.backward()
                if self.cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.compiled.parameters(), self.cfg.grad_clip)
                optimizer.step()

                step += 1
                last_train_loss = float(out.total_loss.item())
                rec: dict[str, float] = {
                    "step": float(step),
                    "epoch": float(epoch + 1),
                    "total_loss": last_train_loss,
                }
                for lname, lval in out.loss_dict.items():
                    rec[f"loss/{lname}"] = float(lval.item())
                history.append(rec)

                if verbose and (step == 1 or step % self.cfg.eval_every_steps == 0):
                    parts = " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k.startswith("loss/"))
                    print(f"[epoch {epoch+1} step {step}] total_loss={last_train_loss:.4f} {parts}")

                if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                    break

            if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                break

        val_loss: Optional[float] = None
        val_acc: Optional[float] = None
        if val_loader is not None:
            val_loss, val_acc = self.evaluate(val_loader, source_id, device)
            if verbose:
                acc_str = f" val_acc={val_acc:.4f}" if val_acc is not None else ""
                print(f"[eval] val_loss={val_loss:.4f}{acc_str}")

        return TrainResult(
            steps=step,
            history=history,
            final_train_loss=last_train_loss,
            final_val_loss=val_loss,
            final_val_acc=val_acc,
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.compiled.state_dict(),
            "graph_ir": self.compiled.graph_ir.model_dump(),
        }
        torch.save(payload, p)
        return p
