"""可控速度的后台训练会话，用于向 WebUI 实时推送 Loss 与节点张量流状态。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from netpulse.core.compiler import CompiledGraph, GraphCompiler
from netpulse.core.ir import GraphIR
from netpulse.core.registry import ComponentRegistry
from netpulse.core.trainer import (
    _build_optimizer,
    _extract_acc,
    build_source_loaders,
    merge_source_batches,
)


def _tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        tf = t.detach().float()
        if tf.numel() == 0:
            return {"shape": list(t.shape), "mean": 0.0, "std": 0.0, "norm": 0.0}
        return {
            "shape": list(t.shape),
            "mean": round(float(tf.mean().item()), 4),
            "std": round(float(tf.std(unbiased=False).item()), 4) if tf.numel() > 1 else 0.0,
            "norm": round(float(tf.norm().item()), 4),
        }


@dataclass
class LiveSessionConfig:
    step_delay_ms: int = 150
    epochs: Optional[int] = None
    lr: Optional[float] = None


class LiveTrainRunner:
    """单会话后台训练控制器：支持实时调速、暂停/继续、停止与逐节点张量流遥测。"""

    def __init__(self, emit_cb: Callable[[dict[str, Any]], None]) -> None:
        self._emit_cb = emit_cb
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()  # set = paused
        self.step_delay_ms: int = 150
        self.running: bool = False
        self.paused: bool = False
        self.graph_name: Optional[str] = None

    def set_speed(self, step_delay_ms: int, paused: Optional[bool] = None) -> dict[str, Any]:
        self.step_delay_ms = max(0, min(3000, int(step_delay_ms)))
        if paused is not None:
            self.paused = bool(paused)
            if self.paused:
                self._pause_flag.set()
            else:
                self._pause_flag.clear()
        return {
            "running": self.running,
            "paused": self.paused,
            "step_delay_ms": self.step_delay_ms,
        }

    def stop(self) -> None:
        self._stop_flag.set()
        self._pause_flag.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.running = False
        self.paused = False

    def start(
        self,
        graph_name: str,
        graph_ir: GraphIR,
        registry: ComponentRegistry,
        cfg: LiveSessionConfig,
    ) -> None:
        self.stop()
        with self._lock:
            self._stop_flag.clear()
            self._pause_flag.clear()
            self.paused = False
            self.step_delay_ms = max(0, min(3000, int(cfg.step_delay_ms)))
            self.graph_name = graph_name
            self.running = True

            # 允许 UI 临时覆盖 epochs / lr
            run_ir = graph_ir.model_copy(deep=True)
            if cfg.epochs is not None and cfg.epochs > 0:
                run_ir.trainer.epochs = int(cfg.epochs)
            if cfg.lr is not None and cfg.lr > 0:
                run_ir.trainer.lr = float(cfg.lr)

            self._thread = threading.Thread(
                target=self._run_loop,
                args=(graph_name, run_ir, registry),
                daemon=True,
            )
            self._thread.start()

    def _run_loop(
        self,
        graph_name: str,
        graph_ir: GraphIR,
        registry: ComponentRegistry,
    ) -> None:
        try:
            compiled: CompiledGraph = GraphCompiler.compile(graph_ir, registry)
            tcfg = graph_ir.trainer
            torch.manual_seed(tcfg.seed)
            device = torch.device(tcfg.device)
            compiled.to(device)

            source_ids, train_loaders, val_loaders = build_source_loaders(
                compiled,
                batch_size=tcfg.batch_size,
                seed=tcfg.seed,
            )
            optimizer = _build_optimizer(compiled, tcfg.optimizer, tcfg.lr, tcfg.weight_decay)

            self._emit_cb({
                "type": "train_started",
                "graph_name": graph_name,
                "epochs": tcfg.epochs,
                "step_delay_ms": self.step_delay_ms,
            })

            step = 0
            last_emit_ts = 0.0
            loader_lens = [len(ld) for ld in train_loaders if hasattr(ld, "__len__")]
            steps_per_epoch = min(loader_lens) if loader_lens else 1
            total_steps_est = tcfg.epochs * max(1, steps_per_epoch)

            for epoch in range(tcfg.epochs):
                if self._stop_flag.is_set():
                    break
                compiled.train()
                for raw_batches in zip(*train_loaders):
                    while self._pause_flag.is_set() and not self._stop_flag.is_set():
                        time.sleep(0.05)
                    if self._stop_flag.is_set():
                        break

                    batch = merge_source_batches(source_ids, raw_batches, device)
                    optimizer.zero_grad()
                    out = compiled(batch, mode="train")
                    if out.total_loss is None:
                        raise RuntimeError("图未产生任何 Loss")
                    out.total_loss.backward()
                    if tcfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(compiled.parameters(), tcfg.grad_clip)
                    optimizer.step()
                    step += 1

                    now = time.monotonic()
                    # 当全速模式(0ms)时节流至 ~20fps 以免淹没浏览器 WS；有延迟时每步都推送
                    should_emit = (
                        self.step_delay_ms > 0
                        or step == 1
                        or (now - last_emit_ts) >= 0.05
                    )
                    if should_emit:
                        last_emit_ts = now
                        node_stats: dict[str, Any] = {}
                        for nid, ports in out.outputs.items():
                            port_stats = {}
                            for pname, sig in ports.items():
                                port_stats[pname] = _tensor_summary(sig.data)
                            node_stats[nid] = port_stats

                        tap_stats: dict[str, Any] = {}
                        for tkey, tsig in out.tap_bus.items():
                            tap_stats[tkey] = _tensor_summary(tsig.data)

                        # 若有折叠子图，将展开后的内部端口遥测映射回外层子图节点
                        exp_info = compiled.erc_result.expanded_info
                        if exp_info is not None:
                            for sg_id, port_groups in exp_info.outer_port_targets.items():
                                sg_ports: dict[str, Any] = {}
                                for pname, tgt in port_groups.get("outputs", {}).items():
                                    flat_nid, flat_port, kind = tgt[0], tgt[1], tgt[2]
                                    if kind == "tap":
                                        st = tap_stats.get(f"{flat_nid}.{flat_port}")
                                    else:
                                        st = node_stats.get(flat_nid, {}).get(flat_port)
                                    if st is not None:
                                        sg_ports[pname] = st
                                if not sg_ports:
                                    for member_id in reversed(exp_info.outer_members.get(sg_id, [])):
                                        if member_id in node_stats and node_stats[member_id]:
                                            sg_ports = dict(node_stats[member_id])
                                            break
                                if sg_ports:
                                    node_stats[sg_id] = sg_ports
                                for tname, tgt in port_groups.get("taps", {}).items():
                                    flat_nid, flat_port, kind = tgt[0], tgt[1], tgt[2]
                                    st = (
                                        tap_stats.get(f"{flat_nid}.{flat_port}")
                                        if kind == "tap"
                                        else node_stats.get(flat_nid, {}).get(flat_port)
                                    )
                                    if st is not None:
                                        tap_stats[f"{sg_id}.{tname}"] = st

                        # 反向梯度总线（backward 已执行，Hook 已回填）
                        grad_stats: dict[str, Any] = {
                            eid: _tensor_summary(gt) for eid, gt in out.grad_bus.items()
                        }

                        # AutoBridge 参数与梯度统计
                        bridge_stats: dict[str, Any] = {}
                        for eid, bmod in compiled.bridges_mod.items():
                            proj = getattr(bmod, "proj", None)
                            w = getattr(proj, "weight", None)
                            if w is not None:
                                w_sum = _tensor_summary(w)
                                g_norm = (
                                    round(float(w.grad.detach().float().norm().item()), 4)
                                    if w.grad is not None
                                    else None
                                )
                                bridge_stats[eid] = {
                                    "norm": w_sum["norm"],
                                    "mean": w_sum["mean"],
                                    "std": w_sum["std"],
                                    "grad_norm": g_norm,
                                    "params": sum(p.numel() for p in bmod.parameters()),
                                }

                        probes: dict[str, Any] = {}
                        for nid, cat in compiled.categories.items():
                            if cat == "output":
                                mod = compiled.nodes_mod[nid]
                                cap = getattr(mod, "capture_snapshot", None)
                                if callable(cap):
                                    snap = cap()
                                    if snap is not None:
                                        probes[nid] = snap

                        loss_dict = {k: round(float(v.item()), 5) for k, v in out.loss_dict.items()}
                        acc_pair = _extract_acc(compiled, out.outputs, batch, graph_output=out)
                        batch_acc = round(acc_pair[0] / acc_pair[1], 4) if acc_pair else None

                        self._emit_cb({
                            "type": "step",
                            "step": step,
                            "total_steps_est": total_steps_est,
                            "epoch": epoch + 1,
                            "epochs": tcfg.epochs,
                            "total_loss": round(float(out.total_loss.item()), 5),
                            "loss_dict": loss_dict,
                            "batch_acc": batch_acc,
                            "node_stats": node_stats,
                            "tap_stats": tap_stats,
                            "grad_stats": grad_stats,
                            "bridge_stats": bridge_stats,
                            "probes": probes,
                            "step_delay_ms": self.step_delay_ms,
                        })

                    if self.step_delay_ms > 0:
                        # 分段休眠以便快速响应停/调速
                        rem = self.step_delay_ms / 1000.0
                        while rem > 0 and not self._stop_flag.is_set():
                            sl = min(0.05, rem)
                            time.sleep(sl)
                            rem -= sl
                            # 动态读取最新 step_delay_ms
                            if self.step_delay_ms == 0:
                                break

                if self._stop_flag.is_set():
                    break

                # 每个 epoch 结束做一次快速验证集评估
                if val_loaders is not None:
                    val_loss, val_acc = self._eval_val(compiled, val_loaders, source_ids, device)
                    self._emit_cb({
                        "type": "eval",
                        "step": step,
                        "epoch": epoch + 1,
                        "val_loss": round(val_loss, 5),
                        "val_acc": round(val_acc, 4) if val_acc is not None else None,
                    })

            self.running = False
            self._emit_cb({
                "type": "train_finished",
                "stopped": self._stop_flag.is_set(),
                "steps": step,
            })
        except Exception as exc:
            self.running = False
            self._emit_cb({"type": "train_error", "message": str(exc)})

    @torch.no_grad()
    def _eval_val(
        self,
        compiled: CompiledGraph,
        val_loaders: list[Any],
        source_ids: list[str],
        device: torch.device,
    ) -> tuple[float, Optional[float]]:
        compiled.eval()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total_samples = 0
        for raw_batches in zip(*val_loaders):
            if self._stop_flag.is_set():
                break
            batch = merge_source_batches(source_ids, raw_batches, device)
            out = compiled(batch, mode="eval", compute_losses=True)
            if out.total_loss is not None:
                total_loss += float(out.total_loss.item())
                n_batches += 1
            acc_pair = _extract_acc(compiled, out.outputs, batch, graph_output=out)
            if acc_pair is not None:
                correct += acc_pair[0]
                total_samples += acc_pair[1]
        compiled.train()
        avg_loss = total_loss / max(1, n_batches)
        acc = (correct / total_samples) if total_samples > 0 else None
        return avg_loss, acc
