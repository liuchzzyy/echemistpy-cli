"""科学数据的统一文件加载接口。

本模块提供简化的数据文件加载接口。自动检测文件格式并委托给
插件目录中相应的读取器进行处理。

主要功能：
- 自动格式检测：根据文件扩展名自动选择合适的读取器
- 多仪器支持：同一扩展名可支持多个仪器的读取器
- 元数据覆盖：支持手动覆盖样本名称、操作员等元数据
- 目录加载：支持加载整个目录的数据文件

使用示例：
    >>> from echemistpy.io import load
    >>> # 自动检测格式 (需要文件存在)
    >>> # raw_data, raw_info = load("data.mpt", sample_name="MySample")
    >>> # 指定仪器（对于有多个读取器的格式）
    >>> # raw_data, raw_info = load("data.xlsx", instrument="lanhe")
    >>> # 加载目录
    >>> # raw_data, raw_info = load("./data_dir", instrument="biologic")
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from echemistpy.io.plugin_manager import get_plugin_manager
from echemistpy.io.standardizer import (
    standardize_names,
)
from echemistpy.io.structures import (
    RawData,
    RawDataInfo,
)

if TYPE_CHECKING:
    pass


def load(  # noqa: PLR0912
    path: str | Path,
    fmt: Optional[str] = None,
    technique: Optional[str | list[str]] = None,
    instrument: Optional[str] = None,
    standardize: bool = True,
    **kwargs: Any,
) -> tuple[RawData, RawDataInfo]:
    """加载数据文件并返回标准化的 RawData 和 RawDataInfo。

    Args:
        path: 数据文件路径
        fmt: 可选的格式覆盖（如 '.mpt'）
        technique: 可选的技术类型提示（字符串或字符串列表）
        instrument: 可选的仪器名称覆盖
        standardize: 是否自动标准化数据（默认为 True）
        **kwargs: 传递给特定读取器的额外参数：
            - sample_name: 可选的样本名称覆盖
            - start_time: 可选的开始时间覆盖
            - operator: 可选的操作员名称覆盖
            - active_material_mass: 可选的活性物质质量覆盖
            - wave_number: 可选的波数覆盖

    Returns:
        (RawData, RawDataInfo) 元组

    Raises:
        ValueError: 如果目录加载未指定 instrument 或 fmt
        ValueError: 如果找不到指定扩展名的加载器
        ValueError: 如果多个加载器可用但未指定 instrument
        ValueError: 如果指定的 instrument 没有匹配的加载器
        FileNotFoundError: 如果文件不存在
        RuntimeError: 如果读取器类未实现 load() 方法
    """
    # Extract metadata overrides from kwargs
    overrides: dict[str, Any] = {
        "sample_name": kwargs.get("sample_name"),
        "start_time": kwargs.get("start_time"),
        "operator": kwargs.get("operator"),
        "active_material_mass": kwargs.get("active_material_mass"),
        "wave_number": kwargs.get("wave_number"),
        "instrument": instrument,
    }
    if technique:
        overrides["technique"] = [technique] if isinstance(technique, str) else technique

    path = Path(path) if isinstance(path, str) else path
    ext = fmt if fmt else path.suffix.lower()

    pm = get_plugin_manager()

    # For directory loading, instrument or fmt must be specified
    if path.is_dir() and not instrument and not fmt:
        raise ValueError(f"Loading a directory requires specifying 'instrument' or 'fmt'. Path: {path}")

    # If it's a directory and no extension is provided, we need to find a loader by instrument
    if path.is_dir() and not ext and instrument:
        # 精确匹配 instrument 名称，避免子串匹配错误
        instrument_normalized = instrument.lower().strip()
        for supported_ext in pm.list_supported_extensions():
            available_instruments = [inst.lower().strip() for inst in pm.get_loader_instruments(supported_ext)]
            if instrument_normalized in available_instruments:
                ext = supported_ext
                break

    # Check available instruments for this extension
    available_instruments = pm.get_loader_instruments(ext)

    if not available_instruments:
        if path.is_dir():
            raise ValueError(f"Could not determine loader for directory: {path}. Please specify 'instrument' or 'fmt'.")
        raise ValueError(f"No loader registered for extension: {ext}")

    # If multiple loaders exist but no instrument is specified, prompt the user
    if instrument is None and len(available_instruments) > 1:
        raise ValueError(f"Multiple loaders available for extension '{ext}'. Please specify 'instrument' to choose one.\nAvailable instruments: {available_instruments}")

    reader_class = pm.get_loader(ext, instrument=instrument)

    if reader_class is None:
        # This happens if instrument was provided but didn't match any registered loader
        raise ValueError(f"No loader found for extension '{ext}' with instrument '{instrument}'.\nAvailable instruments for this format: {available_instruments}")

    # Instantiate reader and load raw data
    # Pass standard metadata to reader if provided
    for k, v in overrides.items():
        if v is not None:
            kwargs[k] = v

    reader = reader_class(filepath=path, **kwargs)
    if not hasattr(reader, "load"):
        raise RuntimeError(f"Reader class {reader_class.__name__} does not implement 'load' method yet.")

    # Pass kwargs to load() to support reader-specific parameters like 'edges'
    raw_data, raw_info = reader.load(**kwargs)

    # Filter out None values and apply manual overrides
    raw_info.update({k: v for k, v in overrides.items() if v is not None})

    if not standardize:
        return raw_data, raw_info

    # Auto-standardize
    standardized_data, standardized_info = standardize_names(raw_data, raw_info, technique_hint=technique)

    return standardized_data, standardized_info


def _register_loader(extensions: list[str], loader_class: Any) -> None:
    """Register a new loader class for specific extensions.

    Args:
        extensions: List of file extensions (e.g., ['.mpt', '.xlsx'])
        loader_class: The class to handle these files
    """
    pm = get_plugin_manager()
    pm.register_loader(extensions, loader_class)


# ============================================================================
# Utility Functions
# ============================================================================


def list_supported_formats() -> Dict[str, str]:
    """Return a dictionary of supported file formats and their descriptions."""
    pm = get_plugin_manager()
    loaders = pm.get_supported_loaders()

    formats = {}
    for ext, plugin_names in loaders.items():
        # Clean extension string for display
        ext_display = ext.lower()

        # Build description
        plugins_str = ", ".join(plugin_names)
        formats[ext_display] = f"Loaded by: {plugins_str}"

    return formats


# ============================================================================
# Plugin System Initialization
# ============================================================================


def _initialize_default_plugins() -> None:  # noqa: PLR0912
    """Initialize and register default loader and saver plugins by scanning plugins directory."""
    pm = get_plugin_manager()
    if pm.initialized:
        return

    # Dynamically discover and import plugins
    import echemistpy.io.plugins as plugins_pkg  # noqa: PLC0415

    plugins_path = str(Path(plugins_pkg.__file__).parent) if plugins_pkg.__file__ else None
    if not plugins_path:
        return

    for _, name, _ in pkgutil.iter_modules([plugins_path]):  # noqa: PLR1702
        if name.startswith("_"):
            continue

        try:
            # Import the module
            full_name = f"echemistpy.io.plugins.{name}"
            module = importlib.import_module(full_name)

            # Look for reader classes
            # Reader classes usually register themselves upon import if they have the registration logic
            # If not, we can inspect and register them here (assuming they have specific attributes)

            # Current implementation assumes plugins are imported and registered manually in the old code.
            # However, looking at the plugin files, they seem to just define classes inheriting from BaseReader.
            # They don't seem to self-register.
            # So we need to inspect the module and find BaseReader subclasses.

            from echemistpy.io.base_reader import BaseReader  # noqa: PLC0415

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseReader) and attr is not BaseReader:
                    # Found a reader class!
                    # Now we need to know which extensions it supports.
                    # We can use the _get_file_extension method or try to inspect metadata
                    # Since _get_file_extension is an instance method in BaseReader, this is tricky without instantiation.
                    # But wait, BaseReader defines _get_file_extension.

                    # Strategy: Use a temporary instance or inspect class attributes if available.
                    # The old code registered specific classes to specific extensions.
                    # Let's try to infer from class variables or method if static.

                    # For now, let's replicate the mapping logic based on class names or known patterns
                    # OR update BaseReader/Plugins to have a 'SUPPORTED_EXTENSIONS' class var.
                    # The current BiologicMPTReader has _get_file_extension as a staticmethod returning '.mpt'.

                    extensions = []
                    if hasattr(attr, "_get_file_extension"):
                        # Check if it's a static method or class method we can call without instance
                        try:
                            ext = attr._get_file_extension()  # type: ignore
                            if isinstance(ext, str):
                                extensions.append(ext)
                        except (TypeError, AttributeError):
                            # Instance method, can't call
                            pass

                    # Fallback mapping based on class names (to match previous logic)
                    if not extensions:
                        name_lower = attr.__name__.lower()
                        if "biologic" in name_lower:
                            extensions = [".mpt"]
                        elif "lanhe" in name_lower:
                            extensions = [".xlsx"]
                        elif "mspd" in name_lower:
                            extensions = [".xye"]
                        elif "claess" in name_lower:
                            extensions = [".dat"]
                        elif "mistral" in name_lower:
                            extensions = [".hdf5"]

                    if extensions:
                        pm.register_loader(extensions, attr)

        except ImportError as e:
            warnings.warn(f"Failed to load plugin {name}: {e}", stacklevel=2)
        except Exception as e:
            warnings.warn(f"Error initializing plugin {name}: {e}", stacklevel=2)

    pm.initialized = True


# Initialize plugins on module import
_initialize_default_plugins()


__all__ = [
    "list_supported_formats",
    "load",
]
