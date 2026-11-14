"""外挂 WebUI 服务端（基于 aiohttp + 原生 WebSocket）。"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from aiohttp import web

from netpulse.core.erc import run_erc
from netpulse.core.ir import GraphIR
from netpulse.core.registry import get_default_registry
from netpulse.webui.runner import LiveSessionConfig, LiveTrainRunner


def _compute_auto_layout(graph: GraphIR, topo_order: list[str]) -> dict[str, dict[str, float]]:
    """根据拓扑层级计算清晰的左->右 DAG 节点坐标。"""
    node_map = {n.id: n for n in graph.nodes}
    order = topo_order if len(topo_order) == len(graph.nodes) else [n.id for n in graph.nodes]

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
    row_gap = 150.0
    for layer_idx, nids in sorted(by_layer.items()):
        total_h = (len(nids) - 1) * row_gap
        start_y = 80.0 + max(0.0, (420.0 - total_h) / 2.0)
        for row_idx, nid in enumerate(nids):
            node = node_map[nid]
            if node.position and "x" in node.position and "y" in node.position:
                positions[nid] = {"x": float(node.position["x"]), "y": float(node.position["y"])}
            else:
                positions[nid] = {
                    "x": 60.0 + layer_idx * col_gap,
                    "y": start_y + row_idx * row_gap,
                }
    return positions


def _serialize_graph_payload(graph_path: Path, plugins_dir: Optional[str]) -> dict[str, Any]:
    reg = get_default_registry(plugins_dir)
    graph = GraphIR.from_file(graph_path)
    erc = run_erc(graph, reg)
    positions = _compute_auto_layout(graph, erc.topo_order)

    nodes_payload = []
    for node in graph.nodes:
        cls = reg.get(node.type) if reg.has(node.type) else None
        sig = erc.signatures.get(node.id)
        out_specs = {
            k: {"shape": list(v.shape), "layout": v.layout}
            for k, v in erc.node_out_specs.get(node.id, {}).items()
        }
        tap_specs = {
            k: {"shape": list(v.shape), "layout": v.layout}
            for k, v in erc.node_tap_specs.get(node.id, {}).items()
        }
        nodes_payload.append({
            "id": node.id,
            "type": node.type,
            "category": cls.category if cls else "unknown",
            "params": node.params,
            "position": positions[node.id],
            "inputs": [p.name for p in sig.inputs] if sig else [],
            "outputs": [p.name for p in sig.outputs] if sig else [],
            "taps": [t.name for t in sig.taps] if sig else [],
            "out_specs": out_specs,
            "tap_specs": tap_specs,
        })

    edges_payload = []
    for edge in graph.edges:
        plan = erc.auto_bridges.get(edge.id)
        shape_info: Optional[list[int]] = None
        if edge.edge_kind == "tap":
            spec = erc.node_tap_specs.get(edge.src_node, {}).get(edge.src_port)
            if spec:
                shape_info = list(spec.shape)
        else:
            spec = erc.node_out_specs.get(edge.src_node, {}).get(edge.src_port)
            if spec:
                shape_info = list(spec.shape)
        edges_payload.append({
            "id": edge.id,
            "src_node": edge.src_node,
            "src_port": edge.src_port,
            "dst_node": edge.dst_node,
            "dst_port": edge.dst_port,
            "edge_kind": edge.edge_kind,
            "bridge": edge.bridge,
            "shape": shape_info,
            "autobridge": (
                {
                    "from": list(plan.src_spec.shape),
                    "to": list(plan.dst_spec.shape),
                }
                if plan
                else None
            ),
        })

    return {
        "name": graph_path.name,
        "graph_name": graph.name,
        "nodes": nodes_payload,
        "edges": edges_payload,
        "losses": [lc.model_dump() for lc in graph.losses],
        "trainer": graph.trainer.model_dump(),
        "erc": {
            "passed": erc.passed,
            "topo_order": erc.topo_order,
            "diagnostics": [
                {
                    "level": d.level,
                    "code": d.code,
                    "message": d.message,
                    "node_id": d.node_id,
                    "edge_id": d.edge_id,
                }
                for d in erc.diagnostics
            ],
        },
    }


def create_app(
    graphs_dir: str | Path = "examples",
    plugins_dir: Optional[str] = "plugins",
) -> web.Application:
    graphs_root = Path(graphs_dir)
    html_path = Path(__file__).parent / "index.html"
    clients: set[web.WebSocketResponse] = set()
    event_loop_holder: dict[str, Optional[asyncio.AbstractEventLoop]] = {"loop": None}
    msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def _emit_from_worker(payload: dict[str, Any]) -> None:
        loop = event_loop_holder["loop"]
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(msg_queue.put_nowait, payload)

    runner = LiveTrainRunner(_emit_from_worker)

    async def _broadcaster(app: web.Application) -> None:
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
        app["broadcaster_task"] = asyncio.create_task(_broadcaster(app))

    async def on_cleanup(app: web.Application) -> None:
        runner.stop()
        task = app.get("broadcaster_task")
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for ws in list(clients):
            await ws.close()
        clients.clear()

    def _list_graph_files() -> list[Path]:
        if not graphs_root.exists():
            return []
        files = list(graphs_root.glob("*.json")) + list(graphs_root.glob("*.yaml")) + list(graphs_root.glob("*.yml"))
        return sorted(files, key=lambda p: p.name)

    async def index(_request: web.Request) -> web.Response:
        return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")

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

    async def api_train_start(request: web.Request) -> web.Response:
        body = await request.json()
        name = body.get("graph_name")
        files = {p.name: p for p in _list_graph_files()}
        target = files.get(name) if name else (next(iter(files.values())) if files else None)
        if target is None:
            return web.json_response({"error": f"找不到图文件: {name}"}, status=404)

        reg = get_default_registry(plugins_dir)
        graph_ir = GraphIR.from_file(target)
        cfg = LiveSessionConfig(
            step_delay_ms=int(body.get("step_delay_ms", 150)),
            epochs=int(body["epochs"]) if body.get("epochs") else None,
            lr=float(body["lr"]) if body.get("lr") else None,
        )
        runner.start(target.name, graph_ir, reg, cfg)
        return web.json_response({"ok": True, "graph_name": target.name, "step_delay_ms": cfg.step_delay_ms})

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

    app = web.Application()
    app.add_routes([
        web.get("/", index),
        web.get("/api/graphs", api_graphs),
        web.get("/api/graph", api_graph),
        web.post("/api/train/start", api_train_start),
        web.post("/api/train/speed", api_train_speed),
        web.post("/api/train/stop", api_train_stop),
        web.get("/ws", ws_endpoint),
    ])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app["runner"] = runner
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    graphs_dir: str | Path = "examples",
    plugins_dir: Optional[str] = "plugins",
) -> None:
    app = create_app(graphs_dir=graphs_dir, plugins_dir=plugins_dir)
    web.run_app(app, host=host, port=port)
