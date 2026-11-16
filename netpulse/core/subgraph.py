"""Subgraph 展开引擎：借鉴 ComfyUI add_graph_prefix 机制，将嵌套子图展平为标准 GraphIR。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from netpulse.core.ir import EdgeIR, GraphIR, LossConfig, NodeIR, SubgraphDef
from netpulse.core.types import Diagnostic


@dataclass
class ExpandedGraph:
    """子图展开结果，包含展平图、诊断及外层子图节点回映表。"""

    graph: GraphIR
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # 展平节点 ID -> 所属最外层（画布当前层）子图节点 ID
    flat_to_outer: dict[str, str] = field(default_factory=dict)
    # 最外层子图节点 ID -> {"inputs": {p: (flat_nid, flat_p)}, "outputs": {p: (flat_nid, flat_p, kind)}, "taps": ...}
    outer_port_targets: dict[str, dict[str, dict[str, tuple[str, ...]]]] = field(default_factory=dict)
    # 最外层子图节点 ID -> 包含的展平叶子节点 ID 列表
    outer_members: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(d.level == "error" for d in self.diagnostics)


def has_subgraphs(graph: GraphIR) -> bool:
    """判断计算图是否包含子图节点或子图定义。"""
    return bool(graph.subgraphs) or any(n.type == "subgraph" for n in graph.nodes)


def expand_graph(graph: GraphIR, max_depth: int = 8) -> ExpandedGraph:
    """递归展开 GraphIR 中的所有 type=='subgraph' 节点，返回展平图与双向映射。"""
    diagnostics: list[Diagnostic] = []
    subgraphs: dict[str, SubgraphDef] = graph.subgraphs or {}

    if not any(n.type == "subgraph" for n in graph.nodes):
        return ExpandedGraph(graph=graph)

    # 追踪最外层（当前画布层）子图节点到展平节点/端口的指针
    flat_to_outer: dict[str, str] = {}
    in_ptrs: dict[tuple[str, str], tuple[str, str]] = {}
    out_ptrs: dict[tuple[str, str], tuple[str, str, str]] = {}
    tap_ptrs: dict[tuple[str, str], tuple[str, str, str]] = {}

    for node in graph.nodes:
        if node.type != "subgraph":
            continue
        sg_id = str((node.params or {}).get("subgraph_id", "")).strip()
        sdef = subgraphs.get(sg_id)
        if sdef is None:
            continue
        for m in sdef.inputs:
            in_ptrs[(node.id, m.port_name)] = (node.id, m.port_name)
        for m in sdef.outputs:
            out_ptrs[(node.id, m.port_name)] = (node.id, m.port_name, "signal")
        for m in sdef.taps:
            tap_ptrs[(node.id, m.port_name)] = (node.id, m.port_name, "tap")

    work_nodes: list[NodeIR] = list(graph.nodes)
    work_edges: list[EdgeIR] = list(graph.edges)
    work_losses: list[LossConfig] = list(graph.losses)
    # node_id -> (subgraph_id 调用栈, 最外层所属节点 ID)
    lineage_map: dict[str, tuple[str, ...]] = {n.id: () for n in work_nodes}
    outer_owner: dict[str, Optional[str]] = {
        n.id: (n.id if n.type == "subgraph" else None) for n in work_nodes
    }

    depth = 0
    while any(n.type == "subgraph" for n in work_nodes):
        depth += 1
        if depth > max_depth:
            bad = next(n for n in work_nodes if n.type == "subgraph")
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="SUBGRAPH_DEPTH_EXCEEDED",
                    node_id=outer_owner.get(bad.id) or bad.id,
                    message=f"子图嵌套层级超过上限 ({max_depth})，节点 '{bad.id}' 展开中止",
                )
            )
            break

        next_nodes: list[NodeIR] = []
        expanded_defs: dict[str, tuple[SubgraphDef, str]] = {}

        for node in work_nodes:
            if node.type != "subgraph":
                next_nodes.append(node)
                continue

            top_outer = outer_owner.get(node.id) or node.id
            sg_id = str((node.params or {}).get("subgraph_id", "")).strip()
            if not sg_id or sg_id not in subgraphs:
                diagnostics.append(
                    Diagnostic(
                        level="error",
                        code="UNKNOWN_SUBGRAPH",
                        node_id=top_outer,
                        message=f"节点 '{node.id}' 引用了不存在的子图定义 '{sg_id}'",
                    )
                )
                continue

            lineage = lineage_map.get(node.id, ())
            if sg_id in lineage:
                diagnostics.append(
                    Diagnostic(
                        level="error",
                        code="SUBGRAPH_CYCLE",
                        node_id=top_outer,
                        message=f"检测到子图循环嵌套: {' -> '.join(lineage)} -> {sg_id}",
                    )
                )
                continue

            sdef = subgraphs[sg_id]
            inner_ids = {in_n.id for in_n in sdef.nodes}
            mapping_ok = True
            for m in (*sdef.inputs, *sdef.outputs, *sdef.taps):
                if m.internal_node not in inner_ids:
                    diagnostics.append(
                        Diagnostic(
                            level="error",
                            code="INVALID_SUBGRAPH_MAPPING",
                            node_id=top_outer,
                            message=(
                                f"子图 '{sg_id}' 的端口映射 '{m.port_name}' 指向了"
                                f"不存在的内部节点 '{m.internal_node}'"
                            ),
                        )
                    )
                    mapping_ok = False
            if not mapping_ok:
                continue

            prefix = f"{node.id}/"
            expanded_defs[node.id] = (sdef, prefix)
            new_lineage = lineage + (sg_id,)

            for in_n in sdef.nodes:
                pref_id = f"{prefix}{in_n.id}"
                next_nodes.append(
                    in_n.model_copy(
                        update={
                            "id": pref_id,
                            "params": dict(in_n.params),
                        }
                    )
                )
                lineage_map[pref_id] = new_lineage
                outer_owner[pref_id] = top_outer
                if in_n.type != "subgraph":
                    flat_to_outer[pref_id] = top_outer

            for lc in sdef.losses:
                work_losses.append(
                    lc.model_copy(update={"node_id": f"{prefix}{lc.node_id}"})
                )

        if not expanded_defs:
            break

        # 构造当前层子图节点的端口映射速查表
        in_map_by_node: dict[str, dict[str, tuple[str, str]]] = {}
        out_map_by_node: dict[str, dict[str, tuple[str, str, str]]] = {}
        tap_map_by_node: dict[str, dict[str, tuple[str, str, str]]] = {}

        for sg_nid, (sdef, prefix) in expanded_defs.items():
            in_map_by_node[sg_nid] = {
                m.port_name: (f"{prefix}{m.internal_node}", m.internal_port)
                for m in sdef.inputs
            }
            out_map_by_node[sg_nid] = {
                m.port_name: (f"{prefix}{m.internal_node}", m.internal_port, m.edge_kind)
                for m in sdef.outputs
            }
            tap_map_by_node[sg_nid] = {
                m.port_name: (f"{prefix}{m.internal_node}", m.internal_port, m.edge_kind)
                for m in sdef.taps
            }

        # 更新外层指针映射（多层嵌套时逐步追踪到最深层）
        for k, (tgt_nid, tgt_port) in list(in_ptrs.items()):
            if tgt_nid in in_map_by_node:
                mapped = in_map_by_node[tgt_nid].get(tgt_port)
                if mapped is not None:
                    in_ptrs[k] = mapped

        for k, (src_nid, src_port, kind) in list(out_ptrs.items()):
            lookup = tap_map_by_node.get(src_nid) if kind == "tap" else out_map_by_node.get(src_nid)
            if lookup and src_port in lookup:
                out_ptrs[k] = lookup[src_port]

        for k, (src_nid, src_port, kind) in list(tap_ptrs.items()):
            lookup = tap_map_by_node.get(src_nid) if kind == "tap" else out_map_by_node.get(src_nid)
            if lookup and src_port in lookup:
                tap_ptrs[k] = lookup[src_port]

        # 重写现有边并加入子图内部边
        rewritten_edges: list[EdgeIR] = []
        for edge in work_edges:
            new_src_node = edge.src_node
            new_src_port = edge.src_port
            new_kind = edge.edge_kind
            new_dst_node = edge.dst_node
            new_dst_port = edge.dst_port

            if new_src_node in expanded_defs:
                if new_kind == "tap":
                    mapped = tap_map_by_node[new_src_node].get(new_src_port)
                else:
                    mapped = out_map_by_node[new_src_node].get(new_src_port)
                if mapped is None:
                    diagnostics.append(
                        Diagnostic(
                            level="error",
                            code="UNKNOWN_SUBGRAPH_PORT",
                            edge_id=edge.id,
                            node_id=outer_owner.get(new_src_node) or new_src_node,
                            message=f"子图节点 '{new_src_node}' 不存在导出端口 '{new_src_port}'",
                        )
                    )
                    continue
                new_src_node, new_src_port, mapped_kind = mapped
                new_kind = mapped_kind

            if new_dst_node in expanded_defs:
                mapped_in = in_map_by_node[new_dst_node].get(new_dst_port)
                if mapped_in is None:
                    diagnostics.append(
                        Diagnostic(
                            level="error",
                            code="UNKNOWN_SUBGRAPH_PORT",
                            edge_id=edge.id,
                            node_id=outer_owner.get(new_dst_node) or new_dst_node,
                            message=f"子图节点 '{new_dst_node}' 不存在输入端口 '{new_dst_port}'",
                        )
                    )
                    continue
                new_dst_node, new_dst_port = mapped_in

            rewritten_edges.append(
                edge.model_copy(
                    update={
                        "src_node": new_src_node,
                        "src_port": new_src_port,
                        "edge_kind": new_kind,
                        "dst_node": new_dst_node,
                        "dst_port": new_dst_port,
                    }
                )
            )

        for sg_nid, (sdef, prefix) in expanded_defs.items():
            for in_e in sdef.edges:
                rewritten_edges.append(
                    in_e.model_copy(
                        update={
                            "id": f"{prefix}{in_e.id}",
                            "src_node": f"{prefix}{in_e.src_node}",
                            "dst_node": f"{prefix}{in_e.dst_node}",
                        }
                    )
                )

        work_nodes = next_nodes
        work_edges = rewritten_edges

    # 汇总外层子图端口映射与成员节点表
    outer_port_targets: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    outer_members: dict[str, list[str]] = {}
    for flat_nid, top_outer in flat_to_outer.items():
        outer_members.setdefault(top_outer, []).append(flat_nid)

    for (outer_id, pname), target in in_ptrs.items():
        outer_port_targets.setdefault(outer_id, {"inputs": {}, "outputs": {}, "taps": {}})
        outer_port_targets[outer_id]["inputs"][pname] = target
    for (outer_id, pname), target in out_ptrs.items():
        outer_port_targets.setdefault(outer_id, {"inputs": {}, "outputs": {}, "taps": {}})
        outer_port_targets[outer_id]["outputs"][pname] = target
    for (outer_id, pname), target in tap_ptrs.items():
        outer_port_targets.setdefault(outer_id, {"inputs": {}, "outputs": {}, "taps": {}})
        outer_port_targets[outer_id]["taps"][pname] = target

    flat_graph = GraphIR(
        name=graph.name,
        nodes=work_nodes,
        edges=work_edges,
        losses=work_losses,
        trainer=graph.trainer,
        subgraphs={},
    )
    return ExpandedGraph(
        graph=flat_graph,
        diagnostics=diagnostics,
        flat_to_outer=flat_to_outer,
        outer_port_targets=outer_port_targets,
        outer_members=outer_members,
    )
