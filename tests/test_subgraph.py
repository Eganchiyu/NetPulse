"""Subgraph 子图封装、嵌套递归展开、ERC 校验与端到端训练测试。"""

from __future__ import annotations

from netpulse.core.compiler import GraphCompiler
from netpulse.core.erc import run_erc
from netpulse.core.ir import (
    EdgeIR,
    GraphIR,
    LossConfig,
    NodeIR,
    SubgraphDef,
    SubgraphPortMapping,
    TrainerConfig,
)
from netpulse.core.registry import get_default_registry
from netpulse.core.subgraph import expand_graph
from netpulse.core.trainer import Trainer


def _make_residual_block_def() -> SubgraphDef:
    return SubgraphDef(
        id="res_mlp",
        name="ResidualMLP",
        title="残差 MLP 块",
        nodes=[
            NodeIR(id="lin", type="linear", params={"out_features": 16}),
            NodeIR(id="act", type="activation", params={"kind": "relu"}),
            NodeIR(id="add", type="tensor_add", params={"alpha": 1.0, "beta": 1.0}),
        ],
        edges=[
            EdgeIR(id="ie1", src_node="lin", src_port="out", dst_node="act", dst_port="x"),
            EdgeIR(id="ie2", src_node="act", src_port="out", dst_node="add", dst_port="b", bridge="auto"),
        ],
        inputs=[
            SubgraphPortMapping(port_name="x_main", internal_node="lin", internal_port="x"),
            SubgraphPortMapping(port_name="x_skip", internal_node="add", internal_port="a"),
        ],
        outputs=[
            SubgraphPortMapping(port_name="out", internal_node="add", internal_port="out"),
        ],
        taps=[
            SubgraphPortMapping(port_name="mid_act", internal_node="act", internal_port="out"),
        ],
    )


def test_subgraph_expand_reused_and_nested_e2e() -> None:
    reg = get_default_registry(None)
    res_block = _make_residual_block_def()

    # 外层包裹一层嵌套子图 outer_stage -> 内部调用 res_mlp
    outer_stage = SubgraphDef(
        id="stage_wrap",
        name="StageWrap",
        nodes=[
            NodeIR(id="proj", type="linear", params={"out_features": 16}),
            NodeIR(id="inner_res", type="subgraph", params={"subgraph_id": "res_mlp"}),
        ],
        edges=[
            EdgeIR(id="se1", src_node="proj", src_port="out", dst_node="inner_res", dst_port="x_main"),
            EdgeIR(id="se2", src_node="proj", src_port="out", dst_node="inner_res", dst_port="x_skip"),
        ],
        inputs=[SubgraphPortMapping(port_name="x", internal_node="proj", internal_port="x")],
        outputs=[SubgraphPortMapping(port_name="y", internal_node="inner_res", internal_port="out")],
        taps=[SubgraphPortMapping(port_name="mid", internal_node="inner_res", internal_port="mid_act", edge_kind="tap")],
    )

    graph = GraphIR(
        name="nested_subgraph_e2e",
        nodes=[
            NodeIR(id="src", type="spiral_data", params={"points_per_class": 80}),
            NodeIR(id="stage", type="subgraph", params={"subgraph_id": "stage_wrap"}),
            NodeIR(id="block2", type="subgraph", params={"subgraph_id": "res_mlp"}),
            NodeIR(id="head", type="linear", params={"out_features": 3}),
            NodeIR(id="ce", type="cross_entropy_loss"),
            NodeIR(id="reg", type="activation_reg_loss", params={"mode": "l2"}),
        ],
        edges=[
            EdgeIR(id="e1", src_node="src", src_port="x", dst_node="stage", dst_port="x"),
            EdgeIR(id="e2", src_node="stage", src_port="y", dst_node="block2", dst_port="x_main"),
            EdgeIR(id="e3", src_node="stage", src_port="y", dst_node="block2", dst_port="x_skip"),
            EdgeIR(id="e4", src_node="block2", src_port="out", dst_node="head", dst_port="x"),
            EdgeIR(id="e5", src_node="head", src_port="out", dst_node="ce", dst_port="logits"),
            EdgeIR(id="e6", src_node="src", src_port="y", dst_node="ce", dst_port="target"),
            EdgeIR(id="e_tap", src_node="stage", src_port="mid", dst_node="reg", dst_port="x", edge_kind="tap"),
        ],
        losses=[LossConfig(node_id="ce", weight=1.0), LossConfig(node_id="reg", weight=0.01)],
        trainer=TrainerConfig(batch_size=32, epochs=15, lr=0.02),
        subgraphs={"res_mlp": res_block, "stage_wrap": outer_stage},
    )

    exp = expand_graph(graph)
    assert not exp.has_errors
    flat_ids = {n.id for n in exp.graph.nodes}
    assert "stage/proj" in flat_ids
    assert "stage/inner_res/lin" in flat_ids
    assert "block2/lin" in flat_ids
    assert exp.flat_to_outer["stage/inner_res/lin"] == "stage"

    erc = run_erc(graph, reg)
    assert erc.passed, erc.format_report()
    assert erc.expanded_info is not None
    out_tgt = erc.expanded_info.outer_port_targets["stage"]["outputs"]["y"]
    assert out_tgt[0] == "stage/inner_res/add"

    compiled = GraphCompiler.compile(graph, reg)
    res = Trainer(compiled).fit(verbose=False)
    assert res.steps > 0
    assert res.final_train_loss < res.history[0]["total_loss"]
    assert res.final_val_acc is not None and res.final_val_acc >= 0.6


def test_subgraph_error_detection() -> None:
    reg = get_default_registry(None)

    # 1. 缺失子图定义
    g_missing = GraphIR(
        name="missing_sg",
        nodes=[NodeIR(id="g1", type="subgraph", params={"subgraph_id": "nonexistent"})],
        edges=[],
    )
    erc_missing = run_erc(g_missing, reg)
    assert not erc_missing.passed
    assert any(d.code == "UNKNOWN_SUBGRAPH" for d in erc_missing.diagnostics)

    # 2. 子图循环嵌套
    cyclic_a = SubgraphDef(
        id="a",
        nodes=[NodeIR(id="sub_b", type="subgraph", params={"subgraph_id": "b"})],
    )
    cyclic_b = SubgraphDef(
        id="b",
        nodes=[NodeIR(id="sub_a", type="subgraph", params={"subgraph_id": "a"})],
    )
    g_cycle = GraphIR(
        name="cycle_sg",
        nodes=[NodeIR(id="root_a", type="subgraph", params={"subgraph_id": "a"})],
        edges=[],
        subgraphs={"a": cyclic_a, "b": cyclic_b},
    )
    erc_cycle = run_erc(g_cycle, reg)
    assert not erc_cycle.passed
    assert any(d.code == "SUBGRAPH_CYCLE" for d in erc_cycle.diagnostics)
