"""元件注册表与自定义插件扫描。"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Type

from netpulse.core.component import BaseComponent

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """管理所有已注册的元件类。"""

    def __init__(self) -> None:
        self._components: dict[str, Type[BaseComponent]] = {}
        self._loaded_plugin_paths: set[str] = set()
        self.plugin_errors: dict[str, str] = {}

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

    def describe_component(self, type_id: str) -> dict[str, Any]:
        """导出单个元件的元信息（对标 ComfyUI /object_info）。"""
        cls = self.get(type_id)
        defaults = cls.default_params()
        try:
            sig = cls.declare_ports(defaults)
        except Exception:
            sig = cls.declare_ports({})
        return {
            "type_id": type_id,
            "display_name": cls.display_name or type_id,
            "category": cls.category,
            "description": cls.description or (cls.__doc__ or "").strip(),
            "params": [p.to_dict() for p in cls.param_schema()],
            "default_params": defaults,
            "inputs": [
                {"name": p.name, "kind": p.kind, "layout": p.layout, "optional": p.optional}
                for p in sig.inputs
            ],
            "outputs": [
                {"name": p.name, "kind": p.kind, "layout": p.layout}
                for p in sig.outputs
            ],
            "taps": [
                {"name": t.name, "source": t.source, "layout": t.spec.layout}
                for t in sig.taps
            ],
        }

    def describe_all(self) -> dict[str, dict[str, Any]]:
        """导出全部已注册元件的 Schema 与端口定义。"""
        return {type_id: self.describe_component(type_id) for type_id in self.list_types()}

    def load_plugins(self, plugins_dir: str | Path) -> list[str]:
        """扫描 plugins 目录下的 .py 文件与子包（__init__.py），单插件失败不中断整体加载。"""
        root = Path(plugins_dir)
        if not root.exists():
            return []

        candidates: list[Path] = []
        for path in sorted(root.glob("*.py")):
            if not path.name.startswith("_"):
                candidates.append(path)
        for sub in sorted(root.iterdir()):
            init_file = sub / "__init__.py"
            if sub.is_dir() and not sub.name.startswith("_") and init_file.exists():
                candidates.append(init_file)

        loaded: list[str] = []
        for path in candidates:
            resolved = str(path.resolve())
            if resolved in self._loaded_plugin_paths:
                loaded.append(str(path))
                continue
            stem = path.parent.name if path.name == "__init__.py" else path.stem
            mod_name = f"netpulse_plugin_{stem}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = module
                    spec.loader.exec_module(module)
                    self._loaded_plugin_paths.add(resolved)
                    self.plugin_errors.pop(str(path), None)
                    loaded.append(str(path))
            except Exception as exc:
                self.plugin_errors[str(path)] = f"{type(exc).__name__}: {exc}"
                logger.warning("Failed to load plugin %s: %s", path, exc)
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
