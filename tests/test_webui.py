"""外挂 WebUI REST API 与 WebSocket 实时数据流测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from netpulse.core.ir import GraphIR
from netpulse.webui.server import create_app


def test_webui_http_and_ws_flow() -> None:
    async def _run() -> None:
        app = create_app(graphs_dir="examples", plugins_dir=None)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            # 1. HTML 页面、全量元件 Schema 与图列表
            r_home = await client.get("/")
            assert r_home.status == 200
            assert "NetPulse WebUI" in await r_home.text()

            r_comps = await client.get("/api/components")
            assert r_comps.status == 200
            comps = (await r_comps.json())["components"]
            assert len(comps) >= 17
            assert "conv2d" in comps
            assert any(p["name"] == "out_channels" for p in comps["conv2d"]["params"])

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
            assert len(gdata["nodes"]) >= 15
            assert all("position" in n for n in gdata["nodes"])

            # 3. 实时 ERC 校验接口（含成功与维度冲突诊断）
            spiral_raw = json.loads(Path("examples/spiral_quickstart.json").read_text(encoding="utf-8"))
            r_erc_ok = await client.post("/api/erc/check", json={"graph": spiral_raw})
            assert r_erc_ok.status == 200
            assert (await r_erc_ok.json())["erc"]["passed"] is True

            bad_raw = json.loads(json.dumps(spiral_raw))
            bad_raw["nodes"][1]["params"]["in_dim"] = 99
            bad_raw["edges"][0]["bridge"] = "none"
            r_erc_bad = await client.post("/api/erc/check", json={"graph": bad_raw})
            assert r_erc_bad.status == 200
            bad_res = await r_erc_bad.json()
            assert bad_res["erc"]["passed"] is False
            assert any(d["code"] == "DIM_MISMATCH" for d in bad_res["erc"]["diagnostics"])

            # 4. 连接 WebSocket 并使用内存修改后的 inline graph 启动极速训练 + 调速 + 停止
            async with client.ws_connect("/ws") as ws:
                hello = json.loads(await ws.receive_str(timeout=3))
                assert hello["type"] == "hello"

                r_start = await client.post(
                    "/api/train/start",
                    json={
                        "graph_name": "inline_spiral.json",
                        "graph": spiral_raw,
                        "step_delay_ms": 20,
                        "epochs": 2,
                    },
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


def test_webui_save_graph_and_path_guard(tmp_path: Path) -> None:
    async def _run() -> None:
        app = create_app(graphs_dir=tmp_path, plugins_dir=None)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            spiral = GraphIR.from_file("examples/spiral_quickstart.json").model_dump()

            # 禁止路径穿越
            r_bad = await client.post("/api/graph/save", json={"name": "../evil.json", "graph": spiral})
            assert r_bad.status == 400

            # 正常保存到 graphs_dir
            r_ok = await client.post("/api/graph/save", json={"name": "saved_spiral.json", "graph": spiral})
            assert r_ok.status == 200
            body = await r_ok.json()
            assert body["ok"] is True
            assert (tmp_path / "saved_spiral.json").exists()

            # 导出为 YAML 与 JSON
            r_exp_yaml = await client.post(
                "/api/graph/export",
                json={"name": "saved_spiral.json", "format": "yaml", "graph": spiral},
            )
            assert r_exp_yaml.status == 200
            exp_yaml = await r_exp_yaml.json()
            assert exp_yaml["filename"] == "saved_spiral.yaml"
            assert "spiral_data" in exp_yaml["content"]

            # 导入 YAML 并可选择保存到 graphs_dir
            r_imp = await client.post(
                "/api/graph/import",
                json={"name": "imported_spiral.yaml", "content": exp_yaml["content"], "save": True},
            )
            assert r_imp.status == 200
            imp_body = await r_imp.json()
            assert imp_body["ok"] is True
            assert imp_body["saved"] is True
            assert imp_body["graph"]["erc"]["passed"] is True
            assert (tmp_path / "imported_spiral.yaml").exists()

            # 加载列表应包含两个工作流
            r_list2 = await client.get("/api/graphs")
            graphs2 = (await r_list2.json())["graphs"]
            assert "saved_spiral.json" in graphs2
            assert "imported_spiral.yaml" in graphs2

            # 删除工作流（含路径穿越防护）
            r_del_bad = await client.post("/api/graph/delete", json={"name": "../evil.json"})
            assert r_del_bad.status == 400
            r_del = await client.post("/api/graph/delete", json={"name": "imported_spiral.yaml"})
            assert r_del.status == 200
            assert not (tmp_path / "imported_spiral.yaml").exists()
        finally:
            await client.close()

    asyncio.run(_run())


def test_webui_templates_api() -> None:
    async def _run() -> None:
        app = create_app(graphs_dir="examples", plugins_dir=None, templates_dir="templates")
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            r_tpls = await client.get("/api/templates")
            assert r_tpls.status == 200
            tpls = (await r_tpls.json())["templates"]
            files = [t["file"] for t in tpls]
            assert "01_spiral_mlp.json" in files
            assert "05_lstm_time_series_forecast.json" in files

            r_tpl = await client.get("/api/template", params={"name": "05_lstm_time_series_forecast.json"})
            assert r_tpl.status == 200
            data = await r_tpl.json()
            assert data["is_template"] is True
            assert data["erc"]["passed"] is True

            r_bad = await client.get("/api/template", params={"name": "../secret.json"})
            assert r_bad.status == 404
        finally:
            await client.close()

    asyncio.run(_run())
