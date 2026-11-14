"""NetPulse: 节点流计算图沙盒内核。"""

from netpulse.core.types import (
    ComponentSignature,
    Diagnostic,
    GraphOutput,
    PortSpec,
    ResolvedSpec,
    Signal,
    TapSpec,
)
from netpulse.core.ir import EdgeIR, GraphIR, LossConfig, NodeIR, TrainerConfig
from netpulse.core.component import BaseComponent
from netpulse.core.registry import ComponentRegistry, get_default_registry, register_component
from netpulse.core.erc import ERCResult, run_erc
from netpulse.core.compiler import CompiledGraph, GraphCompiler
from netpulse.core.trainer import TrainResult, Trainer

__all__ = [
    "Signal",
    "PortSpec",
    "TapSpec",
    "ComponentSignature",
    "ResolvedSpec",
    "GraphOutput",
    "Diagnostic",
    "NodeIR",
    "EdgeIR",
    "LossConfig",
    "TrainerConfig",
    "GraphIR",
    "BaseComponent",
    "ComponentRegistry",
    "get_default_registry",
    "register_component",
    "ERCResult",
    "run_erc",
    "CompiledGraph",
    "GraphCompiler",
    "TrainResult",
    "Trainer",
]
