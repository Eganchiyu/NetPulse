"""梯度控制测试：边级 stop_grad（梯度阻断）与 watch_grad / 探针梯度观测。"""

from __future__ import annotations

import copy

import torch

from netpulse.core.compiler import GraphCompiler
from netpulse.core.erc import run_erc
from netpulse.core.ir import EdgeIR, GraphIR, NodeIR
from netpulse.core.registry import get_default_registry
from netpulse.core.types import Signal


def _batch(n: int = 16) -> dict[str, dict[str, Signal]]:
    g = torch.Generator().manual_seed(0)
    return {
        "src": {
            "x": Signal(data=torch.randn(n, 2, generator=g), meta={"layout": "BD"}),
            "y": Signal(data=torch.randint(0, 3, (n,), generator=g), meta={"layout": "B"}),
        }
    }


def _spiral_graph() -> GraphIR:
    return GraphIR.from_file("examples/spiral_quickstart.json")


def test_stop_grad_edge_defaults_false_and_roundtrip() -> None:
    graph = _spiral_graph()
    assert all(e.stop_grad is False and e.watch_grad is False for e in graph.edges)
    graph.edges[0].stop_grad = True
    graph.edges[1].watch_grad = True
    again = GraphIR.model_validate(graph.model_dump())
    assert again.edges[0].stop_grad is True
    assert again.edges[1].watch_grad is True


def test_stop_grad_decouples_regularization_from_backbone() -> None:
    """阻断正则支路梯度后，主干权重梯度应与「完全没有正则节点」时一致。"""
    reg = get_default_registry()
    graph = _spiral_graph()
    # e4: mlp.tap layer_1 -> reg.x  （正则支路）
    reg_edge = next(e for e in graph.edges if e.id == "e4")
    reg_edge.stop_grad = True

    erc = run_erc(graph, reg)
    assert erc.passed, erc.format_report()

    torch.manual_seed(0)
    compiled = GraphCompiler.compile(graph, reg)
    out = compiled(_batch(), mode="train")
    assert out.total_loss is not None
    out.total_loss.backward()
    grad_with_reg = {n: p.grad.clone() for n, p in compiled.named_parameters() if p.grad is not None}

    # 参照：删掉正则节点与其连线，只留交叉熵
    plain = _spiral_graph()
    plain.nodes = [n for n in plain.nodes if n.id != "reg"]
    plain.edges = [e for e in plain.edges if e.dst_node != "reg"]
    plain.losses = [lc for lc in plain.losses if lc.node_id != "reg"]
    torch.manual_seed(0)
    compiled_plain = GraphCompiler.compile(plain, reg)
    out_plain = compiled_plain(_batch(), mode="train")
    out_plain.total_loss.backward()
    grad_plain = {n: p.grad.clone() for n, p in compiled_plain.named_parameters() if p.grad is not None}

    assert set(grad_with_reg) == set(grad_plain)
    for name in grad_plain:
        assert torch.allclose(grad_with_reg[name], grad_plain[name], atol=1e-6), name


def test_without_stop_grad_regularization_does_change_gradients() -> None:
    reg = get_default_registry()
    graph = _spiral_graph()
    torch.manual_seed(0)
    compiled = GraphCompiler.compile(graph, reg)
    out = compiled(_batch(), mode="train")
    out.total_loss.backward()
    grad_reg = {n: p.grad.clone() for n, p in compiled.named_parameters() if p.grad is not None}

    plain = copy.deepcopy(graph)
    plain.nodes = [n for n in plain.nodes if n.id != "reg"]
    plain.edges = [e for e in plain.edges if e.dst_node != "reg"]
    plain.losses = [lc for lc in plain.losses if lc.node_id != "reg"]
    torch.manual_seed(0)
    compiled_plain = GraphCompiler.compile(plain, reg)
    out_plain = compiled_plain(_batch(), mode="train")
    out_plain.total_loss.backward()
    grad_plain = {n: p.grad.clone() for n, p in compiled_plain.named_parameters() if p.grad is not None}

    assert any(not torch.allclose(grad_reg[n], grad_plain[n], atol=1e-6) for n in grad_plain)


def test_erc_flags_blocked_loss_gradient() -> None:
    reg = get_default_registry()
    graph = _spiral_graph()
    next(e for e in graph.edges if e.id == "e2").stop_grad = True  # mlp.out -> ce.logits
    # 仅 ce 被阻断时只是警告（正则分支仍可反传）
    erc = run_erc(graph, reg)
    codes = [d.code for d in erc.diagnostics]
    assert "STOP_GRAD_ON_LOSS_EDGE" in codes
    assert "LOSS_GRAD_BLOCKED" not in codes
    assert erc.passed

    # 全部 Loss 分支都被阻断 => 错误
    next(e for e in graph.edges if e.id == "e4").stop_grad = True
    erc2 = run_erc(graph, reg)
    assert "LOSS_GRAD_BLOCKED" in [d.code for d in erc2.errors()]
    assert not erc2.passed


def test_watch_grad_populates_grad_bus() -> None:
    reg = get_default_registry()
    graph = _spiral_graph()
    next(e for e in graph.edges if e.id == "e2").watch_grad = True
    erc = run_erc(graph, reg)
    assert erc.passed, erc.format_report()

    compiled = GraphCompiler.compile(graph, reg)
    batch = _batch()
    out = compiled(batch, mode="train")
    assert out.grad_bus == {}
    out.total_loss.backward()
    assert "e2" in out.grad_bus
    assert tuple(out.grad_bus["e2"].shape) == (16, 3)
    assert out.grad_bus is compiled.grad_bus


def test_probe_watch_grad_snapshot() -> None:
    reg = get_default_registry()
    graph = _spiral_graph()
    graph.nodes.append(
        NodeIR(id="gp", type="heatmap_preview", params={"watch_grad": True, "max_size": 16})
    )
    graph.edges.append(
        EdgeIR(id="eg", src_node="mlp", src_port="out", dst_node="gp", dst_port="x")
    )
    erc = run_erc(graph, reg)
    assert erc.passed, erc.format_report()

    compiled = GraphCompiler.compile(graph, reg)
    out = compiled(_batch(), mode="train")
    probe = compiled.nodes_mod["gp"]
    assert probe.capture_snapshot()["grad"] is None  # 尚未反传
    out.total_loss.backward()
    snap = probe.capture_snapshot()
    grad = snap["grad"]
    assert grad is not None
    assert grad["shape"] == [16, 3]
    assert len(grad["norm_history"]) == 1
    assert grad["heatmap"]["symmetric"] is True

    # 未开启 watch_grad 的探针不应产生梯度快照
    graph2 = _spiral_graph()
    graph2.nodes.append(
        NodeIR(id="gp2", type="heatmap_preview", params={"max_size": 16})
    )
    graph2.edges.append(
        EdgeIR(id="eg2", src_node="mlp", src_port="out", dst_node="gp2", dst_port="x")
    )
    compiled2 = GraphCompiler.compile(graph2, reg)
    out2 = compiled2(_batch(), mode="train")
    out2.total_loss.backward()
    assert compiled2.nodes_mod["gp2"].capture_snapshot()["grad"] is None
