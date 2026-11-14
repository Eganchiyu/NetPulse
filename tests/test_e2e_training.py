"""端到端训练闭环与权重导出重载测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from netpulse.core.compiler import GraphCompiler
from netpulse.core.ir import GraphIR
from netpulse.core.registry import get_default_registry
from netpulse.core.trainer import Trainer
from netpulse.core.types import Signal


def test_spiral_quickstart_e2e(tmp_path: Path) -> None:
    reg = get_default_registry(None)
    graph = GraphIR.from_file("examples/spiral_quickstart.json")
    compiled = GraphCompiler.compile(graph, reg)
    trainer = Trainer(compiled)
    res = trainer.fit(verbose=False)

    assert res.steps == 150
    assert res.final_val_acc is not None and res.final_val_acc >= 0.95

    ckpt = trainer.save_checkpoint(tmp_path / "spiral.pt")
    assert ckpt.exists()
    loaded = torch.load(ckpt, map_location="cpu")
    rebuilt = GraphCompiler.compile(GraphIR.model_validate(loaded["graph_ir"]), reg)
    rebuilt.load_state_dict(loaded["state_dict"])

    sample = {
        "src": {
            "x": Signal(data=torch.randn(4, 2), meta={"layout": "BD"}),
            "y": Signal(data=torch.zeros(4, dtype=torch.long), meta={"layout": "B"}),
        }
    }
    out = rebuilt(sample, mode="infer")
    assert out.outputs["mlp"]["out"].shape == (4, 3)


def test_mnist_tap_reg_e2e(tmp_path: Path) -> None:
    reg = get_default_registry(None)
    graph = GraphIR.from_file("examples/mnist_tap_reg.json")
    graph.trainer.epochs = 1
    graph.trainer.max_steps = 20
    compiled = GraphCompiler.compile(graph, reg)
    trainer = Trainer(compiled)
    res = trainer.fit(verbose=False)

    assert res.steps == 20
    assert res.final_val_acc is not None and res.final_val_acc >= 0.65
    assert "loss/ce" in res.history[-1]
    assert "loss/ortho_reg" in res.history[-1]
