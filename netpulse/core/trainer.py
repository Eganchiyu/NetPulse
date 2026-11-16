"""Trainer：基于 CompiledGraph 的 PyTorch 训练与评估闭环。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from netpulse.core.compiler import CompiledGraph
from netpulse.core.types import GraphOutput, Signal


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


def _infer_x_layout(x: torch.Tensor) -> str:
    if x.ndim == 4:
        return "BCHW"
    if x.ndim == 3:
        return "BLD"
    if x.ndim == 2:
        return "BD"
    return "ANY"


def _to_signal_batch(
    source_id: str,
    raw_batch: Any,
    device: torch.device,
) -> dict[str, dict[str, Signal]]:
    if isinstance(raw_batch, (tuple, list)) and len(raw_batch) in (2, 3):
        x = raw_batch[0].to(device)
        y = raw_batch[1].to(device)
        mask = raw_batch[2].to(device).bool() if len(raw_batch) == 3 and raw_batch[2] is not None else None
        x_layout = _infer_x_layout(x)
        return {
            source_id: {
                "x": Signal(data=x, mask=mask, meta={"layout": x_layout}),
                "y": Signal(data=y, meta={"layout": _infer_x_layout(y) if y.ndim >= 2 else "B"}),
            }
        }
    if isinstance(raw_batch, dict):
        mask = raw_batch.get("mask")
        mask_dev = mask.to(device).bool() if isinstance(mask, torch.Tensor) else None
        moved: dict[str, Signal] = {}
        for k, v in raw_batch.items():
            if k == "mask":
                continue
            vt = v.to(device)
            moved[k] = Signal(
                data=vt,
                mask=mask_dev if k == "x" else None,
                meta={"layout": _infer_x_layout(vt) if (k != "y" or vt.ndim >= 2) else "B"},
            )
        return {source_id: moved}
    raise TypeError(f"Unsupported batch format from source '{source_id}': {type(raw_batch)}")


def find_source_ids(compiled: CompiledGraph) -> list[str]:
    """查找计算图中的所有 Source 节点 ID（支持单源与多源协同训练）。"""
    sources = [nid for nid, cat in compiled.categories.items() if cat == "source"]
    if not sources:
        raise ValueError("Trainer 要求至少 1 个 source 节点，当前未发现任何 source")
    return sources


def build_source_loaders(
    compiled: CompiledGraph,
    batch_size: int,
    seed: int,
) -> tuple[list[str], list[Any], Optional[list[Any]]]:
    """为所有 Source 节点构建训练与验证 DataLoader 列表。"""
    source_ids = find_source_ids(compiled)
    train_loaders: list[Any] = []
    val_loaders: list[Any] = []
    for sid in source_ids:
        tr, va = compiled.nodes_mod[sid].build_dataloaders(batch_size=batch_size, seed=seed)
        train_loaders.append(tr)
        val_loaders.append(va)
    has_all_val = all(va is not None for va in val_loaders)
    return source_ids, train_loaders, (val_loaders if has_all_val else None)


def merge_source_batches(
    source_ids: list[str],
    raw_batches: tuple[Any, ...] | list[Any],
    device: torch.device,
) -> dict[str, dict[str, Signal]]:
    """将多路 Source 的原始批数据合并为 CompiledGraph.forward 所需的 batch 字典。"""
    merged: dict[str, dict[str, Signal]] = {}
    for sid, raw_batch in zip(source_ids, raw_batches):
        merged.update(_to_signal_batch(sid, raw_batch, device))
    return merged


def _extract_acc(
    compiled: CompiledGraph,
    out_signals: dict[str, dict[str, Signal]],
    batch: dict[str, dict[str, Signal]],
    source_id: Optional[str] = None,
    graph_output: Optional[GraphOutput] = None,
) -> Optional[tuple[int, int]]:
    """从 GraphOutput.metrics 聚合准确率（由 Loss 组件 compute_metrics 钩子提供）。"""
    if graph_output is not None and "acc" in graph_output.metrics:
        correct, total = graph_output.metrics["acc"]
        if total > 0:
            return int(round(correct)), int(total)

    if source_id and source_id in batch and "y" in batch[source_id]:
        y = batch[source_id]["y"].data
        if y.dtype in (torch.int64, torch.int32):
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

    def _find_source_ids(self) -> list[str]:
        return find_source_ids(self.compiled)

    @torch.no_grad()
    def evaluate(
        self,
        val_loaders: list[Any],
        source_ids: list[str] | str,
        device: torch.device,
    ) -> tuple[float, Optional[float]]:
        self.compiled.eval()
        sids = [source_ids] if isinstance(source_ids, str) else list(source_ids)
        loaders = val_loaders if isinstance(val_loaders, list) else [val_loaders]
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total_samples = 0

        for raw_batches in zip(*loaders):
            batch = merge_source_batches(sids, raw_batches, device)
            out = self.compiled(batch, mode="eval", compute_losses=True)
            if out.total_loss is not None:
                total_loss += float(out.total_loss.item())
                n_batches += 1
            acc_pair = _extract_acc(self.compiled, out.outputs, batch, graph_output=out)
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

        source_ids, train_loaders, val_loaders = build_source_loaders(
            self.compiled,
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
            for raw_batches in zip(*train_loaders):
                batch = merge_source_batches(source_ids, raw_batches, device)
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
        if val_loaders is not None:
            val_loss, val_acc = self.evaluate(val_loaders, source_ids, device)
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
            "graph_ir": self.compiled.raw_graph_ir.model_dump(),
        }
        torch.save(payload, p)
        return p
