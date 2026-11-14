"""EDA 式电气规则检查（ERC）与 Meta Device 形状推导。"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from netpulse.core.component import BaseComponent
from netpulse.core.ir import EdgeIR, GraphIR, NodeIR
from netpulse.core.registry import ComponentRegistry
from netpulse.core.types import (
    ComponentSignature,
    Diagnostic,
    PortSpec,
    ResolvedSpec,
    Signal,
)


@dataclass
class AutoBridgePlan:
    """记录需要由编译器自动插入的 MatrixBridge 信息。"""

    edge_id: str
    src_spec: ResolvedSpec
    dst_spec: ResolvedSpec


@dataclass
class ERCResult:
    """静态检查与 Meta 推导完整结果。"""

    passed: bool
    topo_order: list[str]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    signatures: dict[str, ComponentSignature] = field(default_factory=dict)
    node_in_specs: dict[str, dict[str, ResolvedSpec]] = field(default_factory=dict)
    node_out_specs: dict[str, dict[str, ResolvedSpec]] = field(default_factory=dict)
    node_tap_specs: dict[str, dict[str, ResolvedSpec]] = field(default_factory=dict)
    auto_bridges: dict[str, AutoBridgePlan] = field(default_factory=dict)

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.level == "error"]

    def format_report(self) -> str:
        lines = [f"ERC passed={self.passed}, nodes={len(self.topo_order)}"]
        for d in self.diagnostics:
            loc = f" node={d.node_id}" if d.node_id else ""
            loc += f" edge={d.edge_id}" if d.edge_id else ""
            lines.append(f"[{d.level.upper()}][{d.code}]{loc}: {d.message}")
        if self.auto_bridges:
            lines.append(f"AutoBridges ({len(self.auto_bridges)}):")
            for eid, plan in self.auto_bridges.items():
                lines.append(f"  - {eid}: {plan.src_spec.shape} -> {plan.dst_spec.shape}")
        return "\n".join(lines)


def _dtype_from_str(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "bool": torch.bool,
    }
    return mapping.get(name, torch.float32)


def _feature_dim_index(layout: str, rank: int) -> Optional[int]:
    if layout == "BD" and rank == 2:
        return 1
    if layout == "BLD" and rank == 3:
        return 2
    if layout == "BCHW" and rank == 4:
        return 1
    if rank >= 2:
        return -1
    return None


def _can_autobridge(src: ResolvedSpec, dst_expected: ResolvedSpec) -> bool:
    if src.rank != dst_expected.rank:
        return False
    if src.layout != "ANY" and dst_expected.layout != "ANY" and src.layout != dst_expected.layout:
        return False
    feat_idx = _feature_dim_index(src.layout, src.rank)
    if feat_idx is None:
        return False
    norm_idx = feat_idx if feat_idx >= 0 else src.rank + feat_idx
    for i, (s_dim, d_dim) in enumerate(zip(src.shape, dst_expected.shape)):
        if i == norm_idx:
            continue
        if s_dim != d_dim:
            return False
    return src.shape[norm_idx] != dst_expected.shape[norm_idx]


def _check_port_compat(
    edge: EdgeIR,
    src_spec: ResolvedSpec,
    dst_port: PortSpec,
    diagnostics: list[Diagnostic],
) -> bool:
    if dst_port.rank is not None and src_spec.rank not in dst_port.rank:
        hint = ""
        if src_spec.rank == 4 and 2 in dst_port.rank:
            hint = "（提示：可插入 Flatten 或 GlobalAvgPool 转换 4D->2D）"
        diagnostics.append(
            Diagnostic(
                level="error",
                code="RANK_MISMATCH",
                edge_id=edge.id,
                node_id=edge.dst_node,
                message=(
                    f"端口 '{edge.dst_node}.{dst_port.name}' 期望 Rank {dst_port.rank}，"
                    f"但上游 '{edge.src_node}.{edge.src_port}' 输出为 {src_spec.shape} (Rank {src_spec.rank}){hint}"
                ),
            )
        )
        return False

    if dst_port.layout != "ANY" and src_spec.layout != "ANY" and src_spec.layout != dst_port.layout:
        diagnostics.append(
            Diagnostic(
                level="error",
                code="LAYOUT_MISMATCH",
                edge_id=edge.id,
                node_id=edge.dst_node,
                message=(
                    f"端口 '{edge.dst_node}.{dst_port.name}' 期望布局 '{dst_port.layout}'，"
                    f"但上游输出布局为 '{src_spec.layout}'"
                ),
            )
        )
        return False

    if dst_port.require_mask and not src_spec.has_mask:
        diagnostics.append(
            Diagnostic(
                level="error",
                code="MASK_REQUIRED",
                edge_id=edge.id,
                node_id=edge.dst_node,
                message=f"端口 '{edge.dst_node}.{dst_port.name}' 要求携带 Mask，但上游 Signal 无 Mask",
            )
        )
        return False

    return True


def _resolve_expected_dim(
    comp: BaseComponent,
    node: NodeIR,
    port_name: str,
    actual: ResolvedSpec,
    resolved_in_specs: dict[str, ResolvedSpec],
) -> Optional[ResolvedSpec]:
    """检查元件是否对某端口有固定特征维约束（可触发 AutoBridge 或报错）。"""
    hook = getattr(comp, "expected_input_spec", None)
    if callable(hook):
        return hook(node.params, port_name, actual, resolved_in_specs)
    return None


def _run_meta_node(
    comp: BaseComponent,
    node: NodeIR,
    in_specs: dict[str, ResolvedSpec],
    used_taps: set[str],
    sig: ComponentSignature,
) -> tuple[dict[str, ResolvedSpec], dict[str, ResolvedSpec]]:
    """在 PyTorch meta device 上零显存执行单节点前向，提取输出与 Tap 形状。"""
    with torch.device("meta"):
        module: nn.Module = comp.build(node.params, in_specs)

    meta_inputs: dict[str, Signal] = {}
    for pname, rspec in in_specs.items():
        tensor = torch.empty(rspec.shape, dtype=rspec.dtype, device="meta")
        mask = (
            torch.empty(rspec.shape[:2], dtype=torch.bool, device="meta")
            if rspec.has_mask and len(rspec.shape) >= 2
            else None
        )
        meta_inputs[pname] = Signal(data=tensor, mask=mask, meta={"layout": rspec.layout})

    captured_taps: dict[str, Signal] = {}
    hooks = []
    tap_map = sig.tap_map()

    use_forward_with_taps = hasattr(module, "forward_with_taps") and callable(
        getattr(module, "forward_with_taps")
    )

    if used_taps and not use_forward_with_taps:
        for tap_name in used_taps:
            tspec = tap_map[tap_name]
            submod = module.get_submodule(tspec.source)

            def _make_hook(tname: str, layout: str):
                def _hook(_mod, _inp, out):
                    t = out.data if isinstance(out, Signal) else out
                    captured_taps[tname] = Signal(data=t, meta={"layout": layout})

                return _hook

            hooks.append(submod.register_forward_hook(_make_hook(tap_name, tspec.spec.layout)))

    try:
        with torch.no_grad():
            if use_forward_with_taps and used_taps:
                out_signals, raw_taps = module.forward_with_taps(meta_inputs)
                for tap_name in used_taps:
                    if tap_name in raw_taps:
                        captured_taps[tap_name] = raw_taps[tap_name]
            else:
                out_signals = module(meta_inputs)
    finally:
        for h in hooks:
            h.remove()

    out_specs: dict[str, ResolvedSpec] = {}
    sig_out_map = sig.output_map()
    for oname, osig in out_signals.items():
        layout = osig.layout
        if layout == "ANY" and oname in sig_out_map:
            layout = sig_out_map[oname].layout
        out_specs[oname] = ResolvedSpec(
            shape=tuple(osig.data.shape),
            dtype=osig.data.dtype,
            layout=layout,
            has_mask=osig.mask is not None,
        )

    tap_specs: dict[str, ResolvedSpec] = {}
    for tname, tsig in captured_taps.items():
        layout = tsig.layout
        if layout == "ANY" and tname in tap_map:
            layout = tap_map[tname].spec.layout
        tap_specs[tname] = ResolvedSpec(
            shape=tuple(tsig.data.shape),
            dtype=tsig.data.dtype,
            layout=layout,
            has_mask=tsig.mask is not None,
        )

    return out_specs, tap_specs


def run_erc(graph: GraphIR, registry: ComponentRegistry) -> ERCResult:
    """对 GraphIR 执行完整 ERC 检查与 Meta 形状推导。"""
    diagnostics: list[Diagnostic] = []
    node_map: dict[str, NodeIR] = {}

    for n in graph.nodes:
        if n.id in node_map:
            diagnostics.append(
                Diagnostic(level="error", code="DUPLICATE_NODE_ID", node_id=n.id, message=f"重复节点 ID: {n.id}")
            )
            continue
        if not registry.has(n.type):
            diagnostics.append(
                Diagnostic(level="error", code="UNKNOWN_COMPONENT", node_id=n.id, message=f"未注册元件: {n.type}")
            )
            continue
        node_map[n.id] = n

    if any(d.level == "error" for d in diagnostics):
        return ERCResult(passed=False, topo_order=[], diagnostics=diagnostics)

    signatures: dict[str, ComponentSignature] = {}
    components: dict[str, BaseComponent] = {}
    for nid, node in node_map.items():
        cls = registry.get(node.type)
        comp = cls()
        components[nid] = comp
        try:
            signatures[nid] = cls.declare_ports(node.params)
        except Exception as exc:
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="DECLARE_PORTS_FAILED",
                    node_id=nid,
                    message=f"声明端口失败: {exc}",
                )
            )

    if any(d.level == "error" for d in diagnostics):
        return ERCResult(passed=False, topo_order=[], diagnostics=diagnostics)

    # 校验边与端口合法性
    incoming_by_port: dict[tuple[str, str], list[EdgeIR]] = defaultdict(list)
    adj: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {nid: 0 for nid in node_map}
    used_taps_by_node: dict[str, set[str]] = defaultdict(set)
    connected_nodes: set[str] = set()

    for edge in graph.edges:
        if edge.src_node not in node_map or edge.dst_node not in node_map:
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="DANGLING_EDGE",
                    edge_id=edge.id,
                    message=f"边引用了不存在的节点: {edge.src_node} -> {edge.dst_node}",
                )
            )
            continue

        src_sig = signatures[edge.src_node]
        dst_sig = signatures[edge.dst_node]

        if edge.edge_kind == "tap":
            if edge.src_port not in src_sig.tap_map():
                diagnostics.append(
                    Diagnostic(
                        level="error",
                        code="UNKNOWN_TAP",
                        edge_id=edge.id,
                        node_id=edge.src_node,
                        message=f"节点 '{edge.src_node}' 不存在抽头 '{edge.src_port}'",
                    )
                )
                continue
            used_taps_by_node[edge.src_node].add(edge.src_port)
        else:
            if edge.src_port not in src_sig.output_map():
                diagnostics.append(
                    Diagnostic(
                        level="error",
                        code="UNKNOWN_OUTPUT_PORT",
                        edge_id=edge.id,
                        node_id=edge.src_node,
                        message=f"节点 '{edge.src_node}' 不存在输出端口 '{edge.src_port}'",
                    )
                )
                continue

        if edge.dst_port not in dst_sig.input_map():
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="UNKNOWN_INPUT_PORT",
                    edge_id=edge.id,
                    node_id=edge.dst_node,
                    message=f"节点 '{edge.dst_node}' 不存在输入端口 '{edge.dst_port}'",
                )
            )
            continue

        incoming_by_port[(edge.dst_node, edge.dst_port)].append(edge)
        adj[edge.src_node].append(edge.dst_node)
        indegree[edge.dst_node] += 1
        connected_nodes.add(edge.src_node)
        connected_nodes.add(edge.dst_node)

    # 检查必填端口和重复连线
    for nid, sig in signatures.items():
        for p in sig.inputs:
            edges = incoming_by_port.get((nid, p.name), [])
            if not edges and not p.optional:
                diagnostics.append(
                    Diagnostic(
                        level="error",
                        code="UNCONNECTED_REQUIRED_PORT",
                        node_id=nid,
                        message=f"节点 '{nid}' 的必填输入端口 '{p.name}' 未连接",
                    )
                )
            elif len(edges) > 1:
                diagnostics.append(
                    Diagnostic(
                        level="error",
                        code="MULTIPLE_INPUT_EDGES",
                        node_id=nid,
                        message=f"节点 '{nid}' 的输入端口 '{p.name}' 存在多条连线（请使用 TensorAdd 等显式融合节点）",
                    )
                )

    for nid in node_map:
        if nid not in connected_nodes and len(node_map) > 1:
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="ISOLATED_NODE",
                    node_id=nid,
                    message=f"节点 '{nid}' 未连接到任何其他节点",
                )
            )

    # 拓扑排序与环检测
    q = deque([nid for nid, deg in indegree.items() if deg == 0])
    topo_order: list[str] = []
    while q:
        curr = q.popleft()
        topo_order.append(curr)
        for nxt in adj[curr]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    if len(topo_order) != len(node_map):
        cyclic = [nid for nid, deg in indegree.items() if deg > 0]
        diagnostics.append(
            Diagnostic(
                level="error",
                code="CYCLE_DETECTED",
                message=f"计算图存在有向环，涉及节点: {cyclic}",
            )
        )

    if any(d.level == "error" for d in diagnostics):
        return ERCResult(passed=False, topo_order=topo_order, diagnostics=diagnostics, signatures=signatures)

    # Meta 形状推导
    node_in_specs: dict[str, dict[str, ResolvedSpec]] = {}
    node_out_specs: dict[str, dict[str, ResolvedSpec]] = {}
    node_tap_specs: dict[str, dict[str, ResolvedSpec]] = {}
    auto_bridges: dict[str, AutoBridgePlan] = {}

    for nid in topo_order:
        node = node_map[nid]
        comp = components[nid]
        sig = signatures[nid]
        in_map = sig.input_map()

        resolved_in: dict[str, ResolvedSpec] = {}
        node_failed = False

        # 按签名输入顺序解析，保证 TensorAdd 等多端口对比稳定
        for port_spec in sig.inputs:
            edges = incoming_by_port.get((nid, port_spec.name), [])
            if not edges:
                continue
            edge = edges[0]
            if edge.edge_kind == "tap":
                tap_dict = node_tap_specs.get(edge.src_node, {})
                if edge.src_port not in tap_dict:
                    node_failed = True
                    continue
                src_spec = tap_dict[edge.src_port]
            else:
                out_dict = node_out_specs.get(edge.src_node, {})
                if edge.src_port not in out_dict:
                    node_failed = True
                    continue
                src_spec = out_dict[edge.src_port]

            if not _check_port_compat(edge, src_spec, port_spec, diagnostics):
                node_failed = True
                continue

            expected = _resolve_expected_dim(comp, node, port_spec.name, src_spec, resolved_in)
            if expected is not None and expected.shape != src_spec.shape:
                if _can_autobridge(src_spec, expected):
                    if edge.bridge == "auto":
                        auto_bridges[edge.id] = AutoBridgePlan(
                            edge_id=edge.id,
                            src_spec=src_spec,
                            dst_spec=expected,
                        )
                        diagnostics.append(
                            Diagnostic(
                                level="info",
                                code="AUTO_BRIDGE_INSERTED",
                                edge_id=edge.id,
                                node_id=nid,
                                message=f"边 '{edge.id}' 自动插入 MatrixBridge: {src_spec.shape} -> {expected.shape}",
                            )
                        )
                        src_spec = expected
                    else:
                        diagnostics.append(
                            Diagnostic(
                                level="error",
                                code="DIM_MISMATCH",
                                edge_id=edge.id,
                                node_id=nid,
                                message=(
                                    f"维度不匹配: 上游 {src_spec.shape} vs 期望 {expected.shape}。"
                                    f"请设置 bridge='auto' 或手动插入 MatrixBridge"
                                ),
                            )
                        )
                        node_failed = True
                        continue
                else:
                    diagnostics.append(
                        Diagnostic(
                            level="error",
                            code="SHAPE_MISMATCH",
                            edge_id=edge.id,
                            node_id=nid,
                            message=f"形状不可桥接: 上游 {src_spec.shape} vs 期望 {expected.shape}",
                        )
                    )
                    node_failed = True
                    continue

            resolved_in[port_spec.name] = src_spec

        if node_failed:
            continue

        node_in_specs[nid] = resolved_in
        try:
            out_specs, tap_specs = _run_meta_node(
                comp=comp,
                node=node,
                in_specs=resolved_in,
                used_taps=used_taps_by_node.get(nid, set()),
                sig=sig,
            )
            node_out_specs[nid] = out_specs
            node_tap_specs[nid] = tap_specs
        except Exception as exc:
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="META_INFER_FAILED",
                    node_id=nid,
                    message=f"Meta 形状推导失败: {exc}",
                )
            )

    passed = not any(d.level == "error" for d in diagnostics)
    return ERCResult(
        passed=passed,
        topo_order=topo_order,
        diagnostics=diagnostics,
        signatures=signatures,
        node_in_specs=node_in_specs,
        node_out_specs=node_out_specs,
        node_tap_specs=node_tap_specs,
        auto_bridges=auto_bridges,
    )
