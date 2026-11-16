"""外挂 WebUI 服务端（基于 aiohttp + 原生 WebSocket）。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import urllib.parse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import yaml
from aiohttp import web
from pydantic import ValidationError

from netpulse.core.erc import run_erc
from netpulse.core.ir import GraphIR
from netpulse.core.registry import ComponentRegistry, get_default_registry
from netpulse.webui.runner import LiveSessionConfig, LiveTrainRunner

logger = logging.getLogger(__name__)

_RUNNER_KEY = web.AppKey("runner", LiveTrainRunner)
_BROADCASTER_KEY = web.AppKey("broadcaster_task", asyncio.Task)


def _is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _create_origin_only_middleware():
    """借鉴 ComfyUI：本地环回服务防跨站 Origin 伪造请求。"""

    @web.middleware
    async def origin_only_middleware(request: web.Request, handler):
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            return web.json_response({"error": "Cross-site request blocked"}, status=403)

        if "Host" in request.headers and "Origin" in request.headers:
            host_raw = request.headers["Host"].lower()
            origin_parsed = urllib.parse.urlparse(request.headers["Origin"])
            host_parsed = urllib.parse.urlsplit("//" + host_raw)
            host_name = host_parsed.hostname
            origin_host = (origin_parsed.hostname or "").lower()

            if _is_loopback(host_name):
                host_cmp = host_raw if origin_parsed.port is not None else host_name
                origin_cmp = (origin_parsed.netloc or "").lower() if host_parsed.port is not None else origin_host
                if host_cmp and origin_cmp and host_cmp != origin_cmp:
                    logger.warning("Blocked mismatched Host/Origin: %s != %s", host_cmp, origin_cmp)
                    return web.json_response({"error": "Origin mismatch"}, status=403)

        if request.method == "OPTIONS":
            return web.Response()
        return await handler(request)

    return origin_only_middleware


def _compute_auto_layout(graph: GraphIR, topo_order: list[str], force_auto: bool = False) -> dict[str, dict[str, float]]:
    """根据拓扑层级计算清晰的左->右 DAG 节点坐标。"""
    node_map = {n.id: n for n in graph.nodes}
    if len(topo_order) == len(graph.nodes) and all(nid in node_map for nid in topo_order):
        order = topo_order
    else:
        # 存在子图节点时，针对外层视图图单独计算拓扑顺序
        indeg = {n.id: 0 for n in graph.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for e in graph.edges:
            if e.src_node in node_map and e.dst_node in node_map:
                children[e.src_node].append(e.dst_node)
                indeg[e.dst_node] += 1
        q = deque([nid for nid, d in indeg.items() if d == 0])
        order = []
        while q:
            curr = q.popleft()
            order.append(curr)
            for nxt in children[curr]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        for n in graph.nodes:
            if n.id not in order:
                order.append(n.id)

    parents: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges:
        parents[e.dst_node].append(e.src_node)

    layer_of: dict[str, int] = {}
    for nid in order:
        ps = parents.get(nid, [])
        layer_of[nid] = (max(layer_of.get(p, 0) for p in ps) + 1) if ps else 0

    by_layer: dict[int, list[str]] = defaultdict(list)
    for nid in order:
        by_layer[layer_of[nid]].append(nid)

    positions: dict[str, dict[str, float]] = {}
    col_gap = 270.0
    row_gap = 155.0
    for layer_idx, nids in sorted(by_layer.items()):
        total_h = (len(nids) - 1) * row_gap
        start_y = 80.0 + max(0.0, (420.0 - total_h) / 2.0)
        for row_idx, nid in enumerate(nids):
            node = node_map[nid]
            if not force_auto and node.position and "x" in node.position and "y" in node.position:
                positions[nid] = {"x": float(node.position["x"]), "y": float(node.position["y"])}
            else:
                positions[nid] = {
                    "x": 60.0 + layer_idx * col_gap,
                    "y": start_y + row_idx * row_gap,
                }
    return positions


def serialize_graph_ir(
    graph: GraphIR,
    reg: ComponentRegistry,
    file_name: str = "unsaved.json",
    force_auto_layout: bool = False,
    view_subgraph_path: Optional[list[str]] = None,
) -> dict[str, Any]:
    """对任意 GraphIR 运行 ERC 并序列化为 WebUI 载荷（支持按路径下钻序列化子图视图）。"""
    erc = run_erc(graph, reg)
    exp_info = erc.expanded_info
    flat_to_outer = exp_info.flat_to_outer if exp_info else {}
    outer_targets = exp_info.outer_port_targets if exp_info else {}

    view_graph = graph
    prefix = ""
    active_sub_id: Optional[str] = None
    if view_subgraph_path:
        curr_nodes = graph.nodes
        for seg in view_subgraph_path:
            found = next((n for n in curr_nodes if n.id == seg and n.type == "subgraph"), None)
            if found is None:
                break
            sg_id = str((found.params or {}).get("subgraph_id", "")).strip()
            sdef = graph.subgraphs.get(sg_id)
            if sdef is None:
                break
            prefix += f"{seg}/"
            active_sub_id = sg_id
            curr_nodes = sdef.nodes
            view_graph = GraphIR(
                name=sdef.name,
                nodes=sdef.nodes,
                edges=sdef.edges,
                losses=sdef.losses,
                trainer=graph.trainer,
                subgraphs=graph.subgraphs,
            )

    positions = _compute_auto_layout(view_graph, erc.topo_order, force_auto=force_auto_layout)

    node_out_specs_view: dict[str, dict[str, dict[str, Any]]] = {}
    node_tap_specs_view: dict[str, dict[str, dict[str, Any]]] = {}

    nodes_payload = []
    for node in view_graph.nodes:
        flat_nid = f"{prefix}{node.id}"
        if node.type == "subgraph":
            sg_id = str((node.params or {}).get("subgraph_id", "")).strip()
            sdef = graph.subgraphs.get(sg_id)
            targets = outer_targets.get(flat_nid, {"inputs": {}, "outputs": {}, "taps": {}}) if not prefix else {"inputs": {}, "outputs": {}, "taps": {}}

            in_specs: dict[str, dict[str, Any]] = {}
            for pname, tgt in targets.get("inputs", {}).items():
                t_nid, t_port = tgt[0], tgt[1]
                sp = erc.node_in_specs.get(t_nid, {}).get(t_port)
                if sp:
                    in_specs[pname] = {"shape": list(sp.shape), "layout": sp.layout}

            out_specs: dict[str, dict[str, Any]] = {}
            for pname, tgt in targets.get("outputs", {}).items():
                t_nid, t_port, kind = tgt[0], tgt[1], tgt[2]
                src_dict = erc.node_tap_specs if kind == "tap" else erc.node_out_specs
                sp = src_dict.get(t_nid, {}).get(t_port)
                if sp:
                    out_specs[pname] = {"shape": list(sp.shape), "layout": sp.layout}

            tap_specs: dict[str, dict[str, Any]] = {}
            for pname, tgt in targets.get("taps", {}).items():
                t_nid, t_port, kind = tgt[0], tgt[1], tgt[2]
                src_dict = erc.node_tap_specs if kind == "tap" else erc.node_out_specs
                sp = src_dict.get(t_nid, {}).get(t_port)
                if sp:
                    tap_specs[pname] = {"shape": list(sp.shape), "layout": sp.layout}

            node_out_specs_view[node.id] = out_specs
            node_tap_specs_view[node.id] = tap_specs

            nodes_payload.append({
                "id": node.id,
                "type": "subgraph",
                "display_name": (sdef.name if sdef else "Subgraph"),
                "title": ((sdef.title or sdef.name) if sdef else "子图封装"),
                "label": node.label,
                "description": (sdef.description or f"封装子图 {sg_id}") if sdef else f"缺失子图 {sg_id}",
                "category": "group",
                "params": node.params,
                "position": positions[node.id],
                "inputs": [m.port_name for m in sdef.inputs] if sdef else [],
                "outputs": [m.port_name for m in sdef.outputs] if sdef else [],
                "taps": [m.port_name for m in sdef.taps] if sdef else [],
                "in_specs": in_specs,
                "out_specs": out_specs,
                "tap_specs": tap_specs,
                "subgraph_info": (
                    {
                        "subgraph_id": sg_id,
                        "node_count": len(sdef.nodes),
                        "edge_count": len(sdef.edges),
                        "inputs": [m.model_dump() for m in sdef.inputs],
                        "outputs": [m.model_dump() for m in sdef.outputs],
                        "taps": [m.model_dump() for m in sdef.taps],
                    }
                    if sdef
                    else None
                ),
            })
            continue

        cls = reg.get(node.type) if reg.has(node.type) else None
        sig = erc.signatures.get(flat_nid)
        if sig is None and cls is not None:
            try:
                sig = cls.declare_ports(node.params)
            except Exception:
                sig = None

        in_specs = {
            k: {"shape": list(v.shape), "layout": v.layout}
            for k, v in erc.node_in_specs.get(flat_nid, {}).items()
        }
        out_specs = {
            k: {"shape": list(v.shape), "layout": v.layout}
            for k, v in erc.node_out_specs.get(flat_nid, {}).items()
        }
        tap_specs = {
            k: {"shape": list(v.shape), "layout": v.layout}
            for k, v in erc.node_tap_specs.get(flat_nid, {}).items()
        }
        node_out_specs_view[node.id] = out_specs
        node_tap_specs_view[node.id] = tap_specs
        nodes_payload.append({
            "id": node.id,
            "type": node.type,
            "display_name": (cls.display_name or node.type) if cls else node.type,
            "title": ((cls.title or cls.display_name) if cls else node.type),
            "label": node.label,
            "category": cls.category if cls else "unknown",
            "params": node.params,
            "position": positions[node.id],
            "inputs": [p.name for p in sig.inputs] if sig else [],
            "outputs": [p.name for p in sig.outputs] if sig else [],
            "taps": [t.name for t in sig.taps] if sig else [],
            "in_specs": in_specs,
            "out_specs": out_specs,
            "tap_specs": tap_specs,
        })

    edges_payload = []
    for edge in view_graph.edges:
        flat_eid = f"{prefix}{edge.id}"
        plan = erc.auto_bridges.get(flat_eid)
        shape_info: Optional[list[int]] = None
        if edge.edge_kind == "tap":
            sp_dict = node_tap_specs_view.get(edge.src_node, {}).get(edge.src_port)
            if sp_dict:
                shape_info = list(sp_dict["shape"])
        else:
            sp_dict = node_out_specs_view.get(edge.src_node, {}).get(edge.src_port)
            if sp_dict:
                shape_info = list(sp_dict["shape"])
        edges_payload.append({
            "id": edge.id,
            "src_node": edge.src_node,
            "src_port": edge.src_port,
            "dst_node": edge.dst_node,
            "dst_port": edge.dst_port,
            "edge_kind": edge.edge_kind,
            "bridge": edge.bridge,
            "bridge_init": edge.bridge_init,
            "bridge_bias_init": edge.bridge_bias_init,
            "stop_grad": edge.stop_grad,
            "watch_grad": edge.watch_grad,
            "shape": shape_info,
            "autobridge": (
                {
                    "from": list(plan.src_spec.shape),
                    "to": list(plan.dst_spec.shape),
                    "module_type": plan.module_type,
                    "in_dim": plan.in_dim,
                    "out_dim": plan.out_dim,
                    "param_count": plan.param_count,
                    "weight_init": plan.weight_init,
                    "bias_init": plan.bias_init,
                }
                if plan
                else None
            ),
        })

    def _map_diag_nid(raw_nid: Optional[str]) -> Optional[str]:
        if not raw_nid:
            return None
        if prefix:
            if raw_nid.startswith(prefix):
                rest = raw_nid[len(prefix):]
                return rest.split("/", 1)[0]
            return None
        return flat_to_outer.get(raw_nid, raw_nid)

    return {
        "name": file_name,
        "graph_name": graph.name,
        "active_subgraph_id": active_sub_id,
        "nodes": nodes_payload,
        "edges": edges_payload,
        "losses": [lc.model_dump() for lc in graph.losses],
        "trainer": graph.trainer.model_dump(),
        "subgraphs": {k: v.model_dump() for k, v in graph.subgraphs.items()},
        "erc": {
            "passed": erc.passed,
            "topo_order": erc.topo_order,
            "diagnostics": [
                {
                    "level": d.level,
                    "code": d.code,
                    "message": d.message,
                    "node_id": _map_diag_nid(d.node_id),
                    "raw_node_id": d.node_id,
                    "edge_id": d.edge_id,
                }
                for d in erc.diagnostics
            ],
        },
    }


def _serialize_graph_payload(graph_path: Path, plugins_dir: Optional[str]) -> dict[str, Any]:
    reg = get_default_registry(plugins_dir)
    graph = GraphIR.from_file(graph_path)
    return serialize_graph_ir(graph, reg, file_name=graph_path.name)


def create_app(
    graphs_dir: str | Path = "examples",
    plugins_dir: Optional[str] = "plugins",
    templates_dir: str | Path = "templates",
) -> web.Application:
    graphs_root = Path(graphs_dir)
    templates_root = Path(templates_dir)
    html_path = Path(__file__).parent / "index.html"
    clients: set[web.WebSocketResponse] = set()
    event_loop_holder: dict[str, Optional[asyncio.AbstractEventLoop]] = {"loop": None}
    msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def _emit_from_worker(payload: dict[str, Any]) -> None:
        loop = event_loop_holder["loop"]
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(msg_queue.put_nowait, payload)

    runner = LiveTrainRunner(_emit_from_worker)

    async def _broadcaster(_app: web.Application) -> None:
        try:
            while True:
                payload = await msg_queue.get()
                text = json.dumps(payload, ensure_ascii=False)
                dead: list[web.WebSocketResponse] = []
                for ws in list(clients):
                    try:
                        await ws.send_str(text)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    clients.discard(ws)
        except asyncio.CancelledError:
            return

    async def on_startup(app: web.Application) -> None:
        event_loop_holder["loop"] = asyncio.get_running_loop()
        app[_BROADCASTER_KEY] = asyncio.create_task(_broadcaster(app))

    async def on_cleanup(app: web.Application) -> None:
        runner.stop()
        task = app.get(_BROADCASTER_KEY)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for ws in list(clients):
            await ws.close()
        clients.clear()

    def _list_dir_graphs(root: Path) -> list[Path]:
        if not root.exists():
            return []
        files = list(root.glob("*.json")) + list(root.glob("*.yaml")) + list(root.glob("*.yml"))
        return sorted(files, key=lambda p: p.name)

    def _list_graph_files() -> list[Path]:
        return _list_dir_graphs(graphs_root)

    def _list_template_files() -> list[Path]:
        return _list_dir_graphs(templates_root)

    def _resolve_safe_save_path(raw_name: str) -> Optional[Path]:
        name = Path(raw_name).name
        if not name or name != raw_name or ".." in raw_name or "/" in raw_name or "\\" in raw_name:
            return None
        if Path(name).suffix.lower() not in {".json", ".yaml", ".yml"}:
            name = f"{name}.json"
        graphs_root.mkdir(parents=True, exist_ok=True)
        root_resolved = graphs_root.resolve()
        candidate = (graphs_root / name).resolve()
        if candidate.parent != root_resolved:
            return None
        return candidate

    def _resolve_safe_existing_path(root: Path, raw_name: str) -> Optional[Path]:
        name = Path(raw_name).name
        if not name or name != raw_name or ".." in raw_name or "/" in raw_name or "\\" in raw_name:
            return None
        if not root.exists():
            return None
        root_resolved = root.resolve()
        candidate = (root / name).resolve()
        if candidate.parent != root_resolved or not candidate.is_file():
            return None
        return candidate

    async def index(_request: web.Request) -> web.Response:
        return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")

    async def api_components(_request: web.Request) -> web.Response:
        reg = get_default_registry(plugins_dir)
        return web.json_response({
            "components": reg.describe_all(),
            "plugin_errors": reg.plugin_errors,
        })

    async def api_graphs(_request: web.Request) -> web.Response:
        return web.json_response({
            "graphs": [p.name for p in _list_graph_files()],
            "status": {
                "running": runner.running,
                "paused": runner.paused,
                "step_delay_ms": runner.step_delay_ms,
                "graph_name": runner.graph_name,
            },
        })

    async def api_graph(request: web.Request) -> web.Response:
        name = request.query.get("name")
        files = {p.name: p for p in _list_graph_files()}
        if not files:
            return web.json_response({"error": "未找到任何计算图文件"}, status=404)
        target = files.get(name) if name else next(iter(files.values()))
        if target is None:
            return web.json_response({"error": f"找不到图文件: {name}"}, status=404)
        return web.json_response(_serialize_graph_payload(target, plugins_dir))

    async def api_erc_check(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "无效的 JSON 请求体"}, status=400)

        raw_graph = body.get("graph", body) if isinstance(body, dict) else body
        file_name = str(body.get("name", "draft.json")) if isinstance(body, dict) else "draft.json"
        force_auto = bool(body.get("auto_layout", False)) if isinstance(body, dict) else False
        view_subgraph_path = body.get("view_subgraph_path") if isinstance(body, dict) else None
        try:
            graph = GraphIR.model_validate(raw_graph)
        except ValidationError as exc:
            return web.json_response({"error": f"GraphIR 校验失败: {exc}"}, status=400)

        reg = get_default_registry(plugins_dir)
        return web.json_response(
            serialize_graph_ir(
                graph,
                reg,
                file_name=file_name,
                force_auto_layout=force_auto,
                view_subgraph_path=view_subgraph_path,
            )
        )

    async def api_graph_save(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "无效的 JSON 请求体"}, status=400)

        raw_name = str(body.get("name", "")).strip()
        target = _resolve_safe_save_path(raw_name)
        if target is None:
            return web.json_response({"error": "非法文件名（仅支持当前目录 .json/.yaml/.yml）"}, status=400)

        try:
            graph = GraphIR.model_validate(body.get("graph"))
        except ValidationError as exc:
            return web.json_response({"error": f"GraphIR 校验失败: {exc}"}, status=400)

        graph.to_file(target)
        reg = get_default_registry(plugins_dir)
        payload = serialize_graph_ir(graph, reg, file_name=target.name)
        return web.json_response({"ok": True, "name": target.name, "graph": payload})

    async def api_graph_delete(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "无效的 JSON 请求体"}, status=400)

        raw_name = str(body.get("name", "")).strip()
        if not raw_name or raw_name != Path(raw_name).name or ".." in raw_name or "/" in raw_name or "\\" in raw_name:
            return web.json_response({"error": "非法文件名"}, status=400)
        target = _resolve_safe_existing_path(graphs_root, raw_name)
        if target is None:
            return web.json_response({"error": f"找不到图文件: {raw_name}"}, status=404)
        target.unlink()
        return web.json_response({"ok": True, "deleted": raw_name})

    async def api_graph_import(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "无效的 JSON 请求体"}, status=400)

        raw_name = str(body.get("name", "imported_graph.json")).strip() or "imported_graph.json"
        safe_name = Path(raw_name).name
        if Path(safe_name).suffix.lower() not in {".json", ".yaml", ".yml"}:
            safe_name = f"{safe_name}.json"

        content = body.get("content")
        raw_graph: Any = None
        if isinstance(content, dict):
            raw_graph = content
        elif isinstance(content, str):
            try:
                if safe_name.lower().endswith((".yaml", ".yml")):
                    raw_graph = yaml.safe_load(content)
                else:
                    try:
                        raw_graph = json.loads(content)
                    except json.JSONDecodeError:
                        raw_graph = yaml.safe_load(content)
            except Exception as exc:
                return web.json_response({"error": f"解析工作流文件失败: {exc}"}, status=400)
        else:
            return web.json_response({"error": "缺少工作流文件内容"}, status=400)

        try:
            graph = GraphIR.model_validate(raw_graph)
        except ValidationError as exc:
            return web.json_response({"error": f"GraphIR 校验失败: {exc}"}, status=400)

        saved_name: Optional[str] = None
        if bool(body.get("save", False)):
            save_target = _resolve_safe_save_path(safe_name)
            if save_target is None:
                return web.json_response({"error": "非法文件名"}, status=400)
            graph.to_file(save_target)
            saved_name = save_target.name

        reg = get_default_registry(plugins_dir)
        payload = serialize_graph_ir(graph, reg, file_name=saved_name or safe_name)
        return web.json_response({"ok": True, "name": saved_name or safe_name, "saved": saved_name is not None, "graph": payload})

    async def api_graph_export(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "无效的 JSON 请求体"}, status=400)

        fmt = str(body.get("format", "json")).lower()
        if fmt not in {"json", "yaml", "yml"}:
            return web.json_response({"error": "导出格式仅支持 json 或 yaml"}, status=400)

        try:
            graph = GraphIR.model_validate(body.get("graph"))
        except ValidationError as exc:
            return web.json_response({"error": f"GraphIR 校验失败: {exc}"}, status=400)

        base_name = Path(str(body.get("name") or graph.name or "workflow")).stem or "workflow"
        ext = "yaml" if fmt in {"yaml", "yml"} else "json"
        if ext == "yaml":
            text = yaml.safe_dump(graph.model_dump(), allow_unicode=True, sort_keys=False)
            mime = "application/x-yaml"
        else:
            text = graph.to_json(indent=2) + "\n"
            mime = "application/json"
        return web.json_response({
            "ok": True,
            "filename": f"{base_name}.{ext}",
            "content": text,
            "mime": mime,
        })

    async def api_templates(_request: web.Request) -> web.Response:
        items = []
        for p in _list_template_files():
            try:
                g = GraphIR.from_file(p)
                items.append({
                    "file": p.name,
                    "name": g.name,
                    "node_count": len(g.nodes),
                    "edge_count": len(g.edges),
                    "node_types": sorted({n.type for n in g.nodes}),
                })
            except Exception:
                continue
        return web.json_response({"templates": items})

    async def api_template(request: web.Request) -> web.Response:
        raw_name = str(request.query.get("name", "")).strip()
        target = _resolve_safe_existing_path(templates_root, raw_name)
        if target is None:
            return web.json_response({"error": f"找不到模板文件: {raw_name}"}, status=404)
        payload = _serialize_graph_payload(target, plugins_dir)
        payload["is_template"] = True
        return web.json_response(payload)

    async def api_train_start(request: web.Request) -> web.Response:
        body = await request.json()
        reg = get_default_registry(plugins_dir)

        if "graph" in body and body["graph"] is not None:
            try:
                graph_ir = GraphIR.model_validate(body["graph"])
            except ValidationError as exc:
                return web.json_response({"error": f"GraphIR 校验失败: {exc}"}, status=400)
            session_name = str(body.get("graph_name") or graph_ir.name or "custom_graph.json")
        else:
            name = body.get("graph_name")
            files = {p.name: p for p in _list_graph_files()}
            target = files.get(name) if name else (next(iter(files.values())) if files else None)
            if target is None:
                return web.json_response({"error": f"找不到图文件: {name}"}, status=404)
            graph_ir = GraphIR.from_file(target)
            session_name = target.name

        cfg = LiveSessionConfig(
            step_delay_ms=int(body.get("step_delay_ms", 150)),
            epochs=int(body["epochs"]) if body.get("epochs") else None,
            lr=float(body["lr"]) if body.get("lr") else None,
        )
        runner.start(session_name, graph_ir, reg, cfg)
        return web.json_response({"ok": True, "graph_name": session_name, "step_delay_ms": cfg.step_delay_ms})

    async def api_train_speed(request: web.Request) -> web.Response:
        body = await request.json()
        delay = int(body.get("step_delay_ms", runner.step_delay_ms))
        paused = body.get("paused")
        state = runner.set_speed(delay, paused)
        return web.json_response(state)

    async def api_train_stop(_request: web.Request) -> web.Response:
        runner.stop()
        return web.json_response({"ok": True, "running": False})

    async def ws_endpoint(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)
        try:
            await ws.send_str(json.dumps({
                "type": "hello",
                "running": runner.running,
                "paused": runner.paused,
                "step_delay_ms": runner.step_delay_ms,
                "graph_name": runner.graph_name,
            }))
            async for _msg in ws:
                pass
        finally:
            clients.discard(ws)
        return ws

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app = web.Application(middlewares=[_create_origin_only_middleware()])
    app.add_routes([
        web.get("/", index),
        web.static("/static", static_dir),
        web.get("/api/components", api_components),
        web.get("/api/graphs", api_graphs),
        web.get("/api/graph", api_graph),
        web.post("/api/erc/check", api_erc_check),
        web.post("/api/graph/save", api_graph_save),
        web.post("/api/graph/delete", api_graph_delete),
        web.post("/api/graph/import", api_graph_import),
        web.post("/api/graph/export", api_graph_export),
        web.get("/api/templates", api_templates),
        web.get("/api/template", api_template),
        web.post("/api/train/start", api_train_start),
        web.post("/api/train/speed", api_train_speed),
        web.post("/api/train/stop", api_train_stop),
        web.get("/ws", ws_endpoint),
    ])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app[_RUNNER_KEY] = runner
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    graphs_dir: str | Path = "examples",
    plugins_dir: Optional[str] = "plugins",
    templates_dir: str | Path = "templates",
) -> None:
    app = create_app(graphs_dir=graphs_dir, plugins_dir=plugins_dir, templates_dir=templates_dir)
    web.run_app(app, host=host, port=port)
