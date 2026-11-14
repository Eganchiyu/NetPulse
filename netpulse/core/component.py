"""元件基类定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch.nn as nn

from netpulse.core.types import CategoryKind, ComponentSignature, ParamSpec, ResolvedSpec


class BaseComponent(ABC):
    """所有可注册元件必须继承的基类。"""

    type_id: ClassVar[str]
    category: ClassVar[CategoryKind]
    display_name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @classmethod
    def param_schema(cls) -> list[ParamSpec]:
        """声明元件可编辑参数的 Schema（供 WebUI /object_info 渲染表单）。"""
        return []

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """提取参数默认值字典（仅包含非 None 默认值）。"""
        defaults: dict[str, Any] = {}
        for p in cls.param_schema():
            if p.default is not None:
                defaults[p.name] = p.default
        return defaults

    @classmethod
    @abstractmethod
    def declare_ports(cls, params: dict[str, Any]) -> ComponentSignature:
        """仅依赖节点静态参数声明输入/输出/Tap端口。"""

    @abstractmethod
    def build(
        self,
        params: dict[str, Any],
        in_specs: dict[str, ResolvedSpec],
    ) -> nn.Module:
        """根据解析后的上游维度构建 PyTorch nn.Module。"""
