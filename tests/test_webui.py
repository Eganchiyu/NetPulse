"""外挂 WebUI REST API 与 WebSocket 实时数据流测试。"""

from __future__ import annotations

import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer

from netpulse.webui.server import create_app


def test_webui_http_and_ws_flow() -> None:
    async def _run() -> None:
        app = create_app(graphs_dir="examples", plugins_dir=None)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            # 1. HTML 页面与图列表
            r_home = await client.get("/")
            assert r_home.status == 200
            assert "NetPulse WebUI" in await r_home.text()

            r_list = await client.get("/api/graphs")
            assert r_list.status == 200
            graphs = (await r_list.json())["graphs"]
            assert "mnist_atomic.json" in graphs
            assert "spiral_quickstart.json" in graphs

            # 2. 获取图详情 + ERC 与自动排布坐标
            r_graph = await client.get("/api/graph", params={"name": "mnist_atomic.json"})
            assert r_graph.status == 200
            gdata = await r_graph.json()
            assert gdata["erc"]["passed"] is True
            assert len(gdata["nodes"]) == 15
            assert all("position" in n for n in gdata["nodes"])

            # 3. 连接 WebSocket 并启动极速训练 + 调速 + 停止
            async with client.ws_connect("/ws") as ws:
                hello = json.loads(await ws.receive_str(timeout=3))
                assert hello["type"] == "hello"

                r_start = await client.post(
                    "/api/train/start",
                    json={"graph_name": "spiral_quickstart.json", "step_delay_ms": 20, "epochs": 2},
                )
                assert r_start.status == 200

                r_speed = await client.post(
                    "/api/train/speed",
                    json={"step_delay_ms": 10, "paused": False},
                )
                assert r_speed.status == 200
                assert (await r_speed.json())["step_delay_ms"] == 10

                saw_step = False
                for _ in range(30):
                    raw = await ws.receive_str(timeout=5)
                    msg = json.loads(raw)
                    if msg["type"] == "step":
                        saw_step = True
                        assert "total_loss" in msg
                        assert "mlp" in msg["node_stats"]
                        assert "ce" in msg["loss_dict"]
                        break
                assert saw_step, "未收到训练 step 实时推送"

                r_stop = await client.post("/api/train/stop")
                assert r_stop.status == 200
        finally:
            await client.close()

    asyncio.run(_run())
