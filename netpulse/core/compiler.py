"""GraphCompiler：将校验后的 GraphIR 编译为标准 PyTorch nn.Module。"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional

import torch
import torch.nn as nn

from netpulse.core.erc import AutoBridgePlan, ERCResult, run_erc
from netpulse.core.ir import EdgeIR, GraphIR, LossConfig, NodeIR
from netpulse.core.registry import ComponentRegistry, get_default_registry
from netpulse.core.types import GraphOutput, Signal


class _AutoBridgeModule(nn.Module):
    """由 ERC 自动插入的矩阵投影桥（Linear 或 1x1 Conv）。"""

    def __init__(self, plan: AutoBridgePlan) -> None:
        super().__init__()
        src = plan.src_spec
        dst = plan.dst_spec
        self.layout = src.layout
        if src.layout == "BCHW" and src.rank == 4:
            self.proj: nn.Module = nn.Conv2d(src.shape[1], dst.shape[1], kernel_size=1)
        else:
            self.proj = nn.Linear(src.shape[-1], dst.shape[-1])

    def forward(self, sig: Signal) -> Signal:
        return sig.clone_with(data=self.proj(sig.data))


class CompiledGraph(nn.Module):
    """编译产物：标准 PyTorch nn.Module，支持 Signal 数据流、Tap 总线与多 Loss 汇集。"""

    def __init__(
        self,
        graph_ir: GraphIR,
        erc_result: ERCResult,
        nodes_mod: nn.ModuleDict,
        bridges_mod: nn.ModuleDict,
        categories: dict[str, str],
        tap_hook_targets: dict[str, dict[str, tuple[str, str]]],
    ) -> None:
        super().__init__()
        self.graph_ir = graph_ir
        self.erc_result = erc_result
        self.nodes_mod = nodes_mod
        self.bridges_mod = bridges_mod
        self.categories = categories

        self._node_map: dict[str, NodeIR] = {n.id: n for n in graph_ir.nodes}
        self._incoming_edges: dict[str, list[EdgeIR]] = defaultdict(list)
        for edge in graph_ir.edges:
            self._incoming_edges[edge.dst_node].append(edge)

        self._used_taps_by_node: dict[str, set[str]] = defaultdict(set)
        for edge in graph_ir.edges:
            if edge.edge_kind == "tap":
                self._used_taps_by_node[edge.src_node].add(edge.src_port)

        self._loss_configs: dict[str, LossConfig] = {
            lc.node_id: lc for lc in graph_ir.losses
        }

        # 持久化注册前向 Hook（用于非 forward_with_taps 模块的隐层抽头）
        self._hook_buffer: dict[str, dict[str, Signal]] = defaultdict(dict)
        self._hook_handles = []
        for nid, taps in tap_hook_targets.items():
            mod = self.nodes_mod[nid]
            for tap_name, (sub_path, layout) in taps.items():
                submod = mod.get_submodule(sub_path)

                def _make_hook(node_id: str, tname: str, tlayout: str):
                    def _hook(_m, _inp, out):
                        t = out.data if isinstance(out, Signal) else out
                        self._hook_buffer[node_id][tname] = Signal(
                            data=t, meta={"layout": tlayout}
                        )

                    return _hook

                self._hook_handles.append(
                    submod.register_forward_hook(_make_hook(nid, tap_name, layout))
                )

    def forward(
        self,
        batch: Optional[dict[str, dict[str, Signal]]] = None,
        mode: Literal["train", "eval", "infer"] = "train",
        compute_losses: Optional[bool] = None,
    ) -> GraphOutput:
        if compute_losses is None:
            compute_losses = mode == "train"

        batch = batch or {}
        outputs: dict[str, dict[str, Signal]] = {}
        tap_bus: dict[str, Signal] = {}
        loss_dict: dict[str, torch.Tensor] = {}

        for nid in self.erc_result.topo_order:
            cat = self.categories[nid]
            if cat == "loss" and not compute_losses:
                continue

            mod = self.nodes_mod[nid]

            if cat == "source":
                src_inputs = batch.get(nid, {})
                outputs[nid] = mod(src_inputs)
                continue

            # 组装当前节点输入
            node_inputs: dict[str, Signal] = {}
            for edge in self._incoming_edges[nid]:
                if edge.edge_kind == "tap":
                    sig = tap_bus[f"{edge.src_node}.{edge.src_port}"]
                else:
                    sig = outputs[edge.src_node][edge.src_port]

                if edge.id in self.bridges_mod:
                    sig = self.bridges_mod[edge.id](sig)

                node_inputs[edge.dst_port] = sig

            used_taps = self._used_taps_by_node.get(nid)
            if used_taps and hasattr(mod, "forward_with_taps"):
                node_out, raw_taps = mod.forward_with_taps(node_inputs)
                for tname in used_taps:
                    if tname in raw_taps:
                        tap_bus[f"{nid}.{tname}"] = raw_taps[tname]
            else:
                if nid in self._hook_buffer:
                    self._hook_buffer[nid].clear()
                node_out = mod(node_inputs)
                if nid in self._hook_buffer:
                    for tname, tsig in self._hook_buffer[nid].items():
                        tap_bus[f"{nid}.{tname}"] = tsig

            outputs[nid] = node_out

            if cat == "loss":
                loss_sig = node_out["loss"]
                loss_dict[nid] = loss_sig.data

        total_loss: Optional[torch.Tensor] = None
        if compute_losses and loss_dict:
            for nid, raw_loss in loss_dict.items():
                lc = self._loss_configs.get(nid)
                if lc is not None and not lc.enabled:
                    continue
                weight = lc.weight if lc is not None else 1.0
                term = raw_loss * weight
                total_loss = term if total_loss is None else (total_loss + term)

        return GraphOutput(
            outputs=outputs,
            tap_bus=tap_bus,
            loss_dict=loss_dict,
            total_loss=total_loss,
        )


class GraphCompiler:
    """将 GraphIR 经 ERC 校验后编译为 CompiledGraph。"""

    @staticmethod
    def compile(
        graph_ir: GraphIR,
        registry: Optional[ComponentRegistry] = None,
    ) -> CompiledGraph:
        reg = registry or get_default_registry()
        erc = run_erc(graph_ir, reg)
        if not erc.passed:
            raise ValueError(f"ERC 校验未通过:\n{erc.format_report()}")

        nodes_mod = nn.ModuleDict()
        categories: dict[str, str] = {}
        node_map = {n.id: n for n in graph_ir.nodes}

        used_taps_by_node: dict[str, set[str]] = defaultdict(set)
        for edge in graph_ir.edges:
            if edge.edge_kind == "tap":
                used_taps_by_node[edge.src_node].add(edge.src_port)

        tap_hook_targets: dict[str, dict[str, tuple[str, str]]] = {}

        for nid in erc.topo_order:
            node = node_map[nid]
            cls = reg.get(node.type)
            comp = cls()
            categories[nid] = cls.category
            mod = comp.build(node.params, erc.node_in_specs[nid])
            nodes_mod[nid] = mod

            used_taps = used_taps_by_node.get(nid, set())
            if used_taps and not hasattr(mod, "forward_with_taps"):
                tap_map = erc.signatures[nid].tap_map()
                tap_hook_targets[nid] = {
                    tname: (tap_map[tname].source, tap_map[tname].spec.layout)
                    for tname in used_taps
                }

        bridges_mod = nn.ModuleDict()
        for eid, plan in erc.auto_bridges.items():
            bridges_mod[eid] = _AutoBridgeModule(plan)

        return CompiledGraph(
            graph_ir=graph_ir,
            erc_result=erc,
            nodes_mod=nodes_mod,
            bridges_mod=bridges_mod,
            categories=categories,
            tap_hook_targets=tap_hook_targets,
        )
