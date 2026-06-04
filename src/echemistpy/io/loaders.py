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
    >>> # bundle = load("data.mpt", sample_name="MySample")
    >>> # 指定仪器（对于有多个读取器的格式）
    >>> # bundle = load("data.xlsx", instrument="lanhe")
    >>> # 加载目录
    >>> # bundle = load("./data_dir", instrument="biologic")
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from echemistpy.data.models import DataBundle
from echemistpy.data.standardize import standardize_bundle
from echemistpy.io.contracts import ReaderSpec
from echemistpy.io.plugin_manager import get_plugin_manager

if TYPE_CHECKING:
    pass


def _normalize_extension(extension: str) -> str:
    ext = extension.lower().strip()
    return ext if ext.startswith(".") else f".{ext}"


def _instrument_extensions(instrument: str) -> list[str]:
    pm = get_plugin_manager()
    target = instrument.lower().strip()
    extensions = []
    for ext in pm.list_supported_extensions():
        instruments = [name.lower().strip() for name in pm.get_loader_instruments(ext)]
        if target in instruments:
            extensions.append(ext)
    return sorted(extensions)


def _directory_extension(path: Path, instrument: str) -> str:
    matches = [ext for ext in _instrument_extensions(instrument) if any(path.rglob(f"*{ext}"))]
    if not matches:
        raise ValueError(f"无法确定目录的 reader: {path}。请指定 instrument 或 fmt。")
    if len(matches) > 1:
        raise ValueError(f"目录 '{path}' 中找到多个匹配格式，instrument='{instrument}': {matches}。请指定 fmt。")
    return matches[0]


def load(
    path: str | Path,
    fmt: Optional[str] = None,
    technique: Optional[str | list[str]] = None,
    instrument: Optional[str] = None,
    standardize: bool = True,
    **kwargs: Any,
) -> DataBundle:
    """加载数据文件并返回标准化的 DataBundle。

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
        DataBundle 数据包

    Raises:
        ValueError: 如果目录加载未指定 instrument 或 fmt。
        ValueError: 如果找不到指定扩展名的 reader。
        ValueError: 如果多个 reader 可用但未指定 instrument。
        ValueError: 如果指定的 instrument 没有匹配的 reader。
        FileNotFoundError: 如果文件不存在
        RuntimeError: 如果 reader 类未实现 load() 方法。
    """
    # 提取元数据覆盖项。
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
    ext = _normalize_extension(fmt) if fmt else path.suffix.lower()

    pm = get_plugin_manager()

    # 目录加载必须指定 instrument 或 fmt。
    if path.is_dir() and not instrument and not fmt:
        raise ValueError(f"加载目录必须指定 instrument 或 fmt。路径: {path}")

    # 目录路径没有扩展名时，通过 instrument 推断 reader 扩展名。
    if path.is_dir() and not ext and instrument:
        ext = _directory_extension(path, instrument)

    # 查询该扩展名支持的仪器。
    available_instruments = pm.get_loader_instruments(ext)

    if not available_instruments:
        if path.is_dir():
            raise ValueError(f"无法确定目录的 reader: {path}。请指定 instrument 或 fmt。")
        raise ValueError(f"未注册扩展名对应的 reader: {ext}")

    # 同一扩展名有多个 reader 时必须显式指定 instrument。
    if instrument is None and len(available_instruments) > 1:
        raise ValueError(f"扩展名 '{ext}' 有多个 reader。请指定 instrument 进行选择。\n可用仪器: {available_instruments}")

    reader_class = pm.get_loader(ext, instrument=instrument)

    if reader_class is None:
        # instrument 已提供但没有匹配 reader。
        raise ValueError(f"未找到扩展名 '{ext}' 且 instrument='{instrument}' 的 reader。\n该格式可用仪器: {available_instruments}")

    # 将标准元数据覆盖项传给 reader。
    for k, v in overrides.items():
        if v is not None:
            kwargs[k] = v

    reader = reader_class(filepath=path, **kwargs)
    if not hasattr(reader, "load"):
        raise RuntimeError(f"Reader 类 {reader_class.__name__} 尚未实现 load() 方法。")

    # 将 kwargs 传给 load()，支持 edges 等 reader 特定参数。
    bundle = reader.load(**kwargs)

    # 应用手动覆盖项。
    bundle.meta.update({k: v for k, v in overrides.items() if v is not None})
    bundle.provenance.setdefault("source_path", str(path))
    bundle.provenance.setdefault("source_format", ext)

    if not standardize:
        return bundle

    return standardize_bundle(bundle, technique_hint=technique)


def _register_loader(extensions: list[str], loader_class: Any) -> None:
    """为指定扩展名注册 reader 类。

    Args:
        extensions: 文件扩展名列表（例如 ['.mpt', '.xlsx']）
        loader_class: 处理这些文件的类
    """
    pm = get_plugin_manager()
    pm.register_loader(extensions, loader_class)


# ============================================================================
# Utility Functions
# ============================================================================


def list_supported_formats() -> dict[str, str]:
    """返回支持的文件格式及其说明。"""
    pm = get_plugin_manager()
    formats: dict[str, str] = {}
    for spec in pm.list_reader_specs():
        detail = f"{spec.name}; instruments={','.join(spec.instruments)}; techniques={','.join(spec.techniques)}"
        for ext in spec.extensions:
            formats.setdefault(ext, detail)
            if formats[ext] != detail:
                formats[ext] = f"{formats[ext]} | {detail}"
    return dict(sorted(formats.items()))


def list_reader_specs() -> tuple[ReaderSpec, ...]:
    """返回已声明的 reader 能力。"""
    pm = get_plugin_manager()
    return tuple(pm.list_reader_specs())


# ============================================================================
# Plugin System Initialization
# ============================================================================


def _reader_spec(reader_class: type[Any]) -> ReaderSpec | None:
    """返回 reader 类声明的 ReaderSpec。"""
    spec = getattr(reader_class, "spec", None)
    if isinstance(spec, ReaderSpec):
        return spec
    return None


def _register_reader_classes(module: Any) -> None:
    """注册一个插件模块中声明的 reader 类。"""
    from echemistpy.io.base_reader import BaseReader  # noqa: PLC0415

    pm = get_plugin_manager()
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not (isinstance(attr, type) and issubclass(attr, BaseReader) and attr is not BaseReader):
            continue
        spec = _reader_spec(attr)
        if spec is None:
            warnings.warn(f"Reader {attr.__module__}.{attr.__name__} 未声明 ReaderSpec，已跳过注册。", stacklevel=2)
            continue
        pm.register_loader(list(spec.extensions), attr)


def _initialize_default_plugins() -> None:
    """扫描 plugins 目录并注册默认 reader 插件。"""
    pm = get_plugin_manager()
    if pm.initialized:
        return

    # 动态发现并导入插件。
    import echemistpy.io.plugins as plugins_pkg  # noqa: PLC0415

    plugins_path = str(Path(plugins_pkg.__file__).parent) if plugins_pkg.__file__ else None
    if not plugins_path:
        return

    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        if name.startswith("_"):
            continue

        try:
            full_name = f"echemistpy.io.plugins.{name}"
            module = importlib.import_module(full_name)
            _register_reader_classes(module)

        except ImportError as e:
            warnings.warn(f"加载插件失败 {name}: {e}", stacklevel=2)
        except Exception as e:
            warnings.warn(f"初始化插件出错 {name}: {e}", stacklevel=2)

    pm.initialized = True


# 模块导入时初始化插件。
_initialize_default_plugins()


__all__ = [
    "list_reader_specs",
    "list_supported_formats",
    "load",
]
