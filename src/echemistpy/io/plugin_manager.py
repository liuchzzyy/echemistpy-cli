"""echemistpy IO 系统的插件注册管理器。

本模块提供一个简单的插件注册系统，用于管理数据加载器和保存器插件，
无需外部依赖（如 pluggy）。

主要功能：
- 注册和管理文件格式读取器（loaders）
- 注册和管理数据保存器（savers）
- 支持同一扩展名的多个加载器（通过 instrument 区分）
- 单例模式确保全局唯一的插件管理器实例

使用示例：
    >>> from echemistpy.io.plugin_manager import get_plugin_manager
    >>> pm = get_plugin_manager()
    >>> # pm.register_loader(['.mpt'], BiologicMPTReader)
    >>> loader = pm.get_loader('.mpt', instrument='biologic')
"""

from __future__ import annotations

from typing import Any, Optional

from traitlets import Bool, Dict, HasTraits


class IOPluginManager(HasTraits):
    """IO 插件管理器，使用 traitlets 管理插件注册。

    采用单例模式，确保全局只有一个插件管理器实例。
    支持多种文件格式和仪器的插件注册。

    Attributes:
        loaders: 字典，映射文件扩展名到加载器类列表
        savers: 字典，映射格式名称到保存器类
        initialized: 布尔值，指示默认插件是否已初始化

    Example:
        >>> pm = IOPluginManager.get_instance()
        >>> # pm.register_loader(['.csv'], MyCSVReader)
        >>> reader_class = pm.get_loader('.csv')
    """

    _instance = None

    loaders = Dict(help="字典，映射文件扩展名到加载器类列表")
    savers = Dict(help="字典，映射格式名称到保存器类")
    initialized = Bool(False, help="默认插件是否已初始化")

    @classmethod
    def get_instance(cls) -> IOPluginManager:
        """获取全局插件管理器实例（单例模式）。

        Returns:
            全局唯一的 IOPluginManager 实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_loader(self, extensions: list[str], loader_class: Any) -> None:
        """为指定的文件扩展名注册加载器类。

        Args:
            extensions: 文件扩展名列表（如 ['.mpt', '.mpr']）
            loader_class: 处理这些文件的类或工厂函数
        """
        # 批量收集所有更改，一次性更新 traitlets
        updates = {}
        for ext in extensions:
            ext_clean = ext.lower()
            if not ext_clean.startswith("."):
                ext_clean = f".{ext_clean}"

            # 获取当前扩展的加载器列表
            current_list = list(self.loaders.get(ext_clean, []))

            # 避免重复注册
            if loader_class not in current_list:
                current_list.append(loader_class)
                updates[ext_clean] = current_list

        # 一次性更新所有更改
        if updates:
            new_loaders = dict(self.loaders)
            new_loaders.update(updates)
            self.loaders = new_loaders

    def register_saver(self, formats: list[str], saver_class: Any) -> None:
        """为指定的格式注册保存器类。

        Args:
            formats: 格式名称列表（如 ['csv', 'json']）
            saver_class: 处理数据保存的类或工厂函数
        """
        current_savers = dict(self.savers)
        for fmt in formats:
            current_savers[fmt.lower()] = saver_class
        self.savers = current_savers

    @staticmethod
    def _get_instrument_name(loader_cls: Any) -> str:
        """Helper to safely extract instrument name from a loader class."""
        loader_inst = getattr(loader_cls, "instrument", None)
        if loader_inst is None:
            return ""

        # Check if it's a traitlet (has default_value)
        if hasattr(loader_inst, "default_value"):
            return str(loader_inst.default_value)  # type: ignore

        return str(loader_inst)

    def get_loader(self, extension: str, instrument: Optional[str] = None) -> Optional[Any]:
        """获取指定扩展名的加载器，可选择按仪器过滤。

        如果扩展名有多个加载器且未指定仪器，返回第一个注册的加载器。

        Args:
            extension: 文件扩展名（如 '.mpt'）
            instrument: 可选的仪器名称过滤器

        Returns:
            加载器类，如果未找到则返回 None
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        loaders = self.loaders.get(ext, [])
        if not loaders:
            return None

        if instrument:
            inst_lower = instrument.lower()
            # 尝试找到匹配仪器名称的加载器
            for loader in loaders:
                inst_name = self._get_instrument_name(loader)

                if inst_name and inst_name.lower() == inst_lower:
                    return loader

                # 作为后备，检查类名是否包含仪器名称
                if inst_lower in loader.__name__.lower():
                    return loader

            # 如果指定了仪器但未找到匹配，返回 None
            return None

        # 如果未提供仪器，默认返回第一个
        return loaders[0]

    def get_saver(self, fmt: str) -> Optional[Any]:
        """获取指定格式的保存器。

        Args:
            fmt: 格式名称（如 'csv'）

        Returns:
            保存器类，如果未找到则返回 None
        """
        return self.savers.get(fmt.lower())

    def list_supported_extensions(self) -> list[str]:
        """列出所有支持的文件扩展名。

        Returns:
            支持的文件扩展名列表（包含点号）
        """
        return list(self.loaders.keys())

    def get_supported_loaders(self) -> dict[str, list[str]]:
        """获取支持的加载器字典。

        Returns:
            映射扩展名到加载器名称列表的字典
        """
        return {ext: [loader.__name__ if hasattr(loader, "__name__") else str(loader) for loader in loaders] for ext, loaders in self.loaders.items()}

    def get_loader_instruments(self, extension: str) -> list[str]:
        """获取指定扩展名的可用仪器名称列表。

        Args:
            extension: 文件扩展名（如 '.xlsx'）

        Returns:
            仪器名称或类名称的列表
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        loaders = self.loaders.get(ext, [])
        instruments = []
        for loader in loaders:
            inst_name = self._get_instrument_name(loader)
            if inst_name:
                instruments.append(inst_name)
            else:
                instruments.append(loader.__name__)
        return instruments


def get_plugin_manager() -> IOPluginManager:
    """获取全局插件管理器实例的便捷函数。

    Returns:
        全局 IOPluginManager 实例
    """
    return IOPluginManager.get_instance()


__all__ = [
    "IOPluginManager",
    "get_plugin_manager",
]
