"""元件注册表与自定义插件扫描。"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Type

from netpulse.core.component import BaseComponent


class ComponentRegistry:
    """管理所有已注册的元件类。"""

    def __init__(self) -> None:
        self._components: dict[str, Type[BaseComponent]] = {}

    def register(self, type_id: str, cls: Type[BaseComponent]) -> None:
        if not issubclass(cls, BaseComponent):
            raise TypeError(f"{cls} must inherit from BaseComponent")
        cls.type_id = type_id
        self._components[type_id] = cls

    def get(self, type_id: str) -> Type[BaseComponent]:
        if type_id not in self._components:
            raise KeyError(f"Unknown component type: '{type_id}'")
        return self._components[type_id]

    def has(self, type_id: str) -> bool:
        return type_id in self._components

    def list_types(self) -> list[str]:
        return sorted(self._components.keys())

    def load_plugins(self, plugins_dir: str | Path) -> list[str]:
        """扫描 plugins 目录下的 .py 文件并自动注册其中的 @register_component 元件。"""
        root = Path(plugins_dir)
        if not root.exists():
            return []
        loaded: list[str] = []
        for path in sorted(root.glob("*.py")):
            if path.name.startswith("_"):
                continue
            mod_name = f"netpulse_plugin_{path.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                loaded.append(str(path))
        return loaded


_DEFAULT_REGISTRY = ComponentRegistry()
_BUILTINS_LOADED = False


def register_component(
    type_id: str,
    registry: ComponentRegistry | None = None,
):
    """装饰器：将元件类注册到指定或全局默认注册表。"""

    def decorator(cls: Type[BaseComponent]) -> Type[BaseComponent]:
        target = registry or _DEFAULT_REGISTRY
        target.register(type_id, cls)
        return cls

    return decorator


def get_default_registry(plugins_dir: str | Path | None = "plugins") -> ComponentRegistry:
    """获取包含内置元件及可选 plugins 自定义元件的默认注册表。"""
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        importlib.import_module("netpulse.components")
        _BUILTINS_LOADED = True
    if plugins_dir is not None:
        _DEFAULT_REGISTRY.load_plugins(plugins_dir)
    return _DEFAULT_REGISTRY
