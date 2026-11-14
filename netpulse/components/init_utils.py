"""权重与偏置统一初始化工具。"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def _resolve_spec(cfg: Any) -> tuple[str, dict[str, Any]]:
    if cfg is None:
        return "default", {}
    if isinstance(cfg, str):
        return cfg.lower(), {}
    if isinstance(cfg, dict):
        name = str(cfg.get("type", cfg.get("mode", "default"))).lower()
        params = {k: v for k, v in cfg.items() if k not in ("type", "mode")}
        return name, params
    raise ValueError(f"Unsupported init config: {cfg}")


def apply_init(
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    weight_init: Any = None,
    bias_init: Any = None,
) -> None:
    """对非 meta 设备上的权重与偏置执行初始化。"""
    if weight is not None and weight.device.type != "meta":
        w_name, w_params = _resolve_spec(weight_init)
        if w_name == "default":
            pass
        elif w_name == "kaiming_normal":
            nn.init.kaiming_normal_(
                weight,
                a=float(w_params.get("a", 0.0)),
                nonlinearity=str(w_params.get("nonlinearity", "relu")),
            )
        elif w_name == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                weight,
                a=float(w_params.get("a", 0.0)),
                nonlinearity=str(w_params.get("nonlinearity", "relu")),
            )
        elif w_name == "xavier_normal":
            nn.init.xavier_normal_(weight, gain=float(w_params.get("gain", 1.0)))
        elif w_name == "xavier_uniform":
            nn.init.xavier_uniform_(weight, gain=float(w_params.get("gain", 1.0)))
        elif w_name == "orthogonal":
            nn.init.orthogonal_(weight, gain=float(w_params.get("gain", 1.0)))
        elif w_name == "normal":
            nn.init.normal_(
                weight,
                mean=float(w_params.get("mean", 0.0)),
                std=float(w_params.get("std", 0.02)),
            )
        elif w_name == "uniform":
            nn.init.uniform_(
                weight,
                a=float(w_params.get("a", -0.05)),
                b=float(w_params.get("b", 0.05)),
            )
        elif w_name == "zeros":
            nn.init.zeros_(weight)
        elif w_name == "ones":
            nn.init.ones_(weight)
        elif w_name == "constant":
            nn.init.constant_(weight, float(w_params.get("val", 0.0)))
        else:
            raise ValueError(f"Unsupported weight init method: {w_name}")

    if bias is not None and bias.device.type != "meta":
        b_name, b_params = _resolve_spec(bias_init)
        if b_name == "default":
            pass
        elif b_name == "zeros":
            nn.init.zeros_(bias)
        elif b_name == "ones":
            nn.init.ones_(bias)
        elif b_name == "constant":
            nn.init.constant_(bias, float(b_params.get("val", 0.0)))
        elif b_name == "normal":
            nn.init.normal_(
                bias,
                mean=float(b_params.get("mean", 0.0)),
                std=float(b_params.get("std", 0.01)),
            )
        else:
            raise ValueError(f"Unsupported bias init method: {b_name}")
