"""所有数据读取器的基类。

本模块为文件读取器提供通用基类，以减少代码重复
并确保一致的行为。

主要功能：
- 文件/目录加载：统一的加载接口
- 元数据管理：集中处理通用元数据覆盖
- 错误处理：细化的异常处理和日志记录
- DataTree 支持：自动将目录文件组织为分层结构
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import xarray as xr

from echemistpy.data.models import DataBundle, Metadata
from echemistpy.data.utils import merge_metadata, sanitize_variable_names
from echemistpy.io.contracts import ReaderSpec

logger = logging.getLogger(__name__)


class BaseReader:
    """所有文件读取器的基类。

    为以下内容提供通用功能：
    - 文件/目录加载
    - 元数据管理
    - 通用 reader 配置

    类变量：
        supports_directories: 是否支持目录加载
        instrument: 仪器标识符

    属性：
        filepath: 文件或目录路径
        sample_name: 样本名称
        start_time: 开始时间
        operator: 操作员名称
        active_material_mass: 活性物质质量
        wave_number: 波数
        technique: 技术类型列表
    """

    # --- 子类应覆盖的类变量 ---
    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "unknown"

    def __init__(self, filepath: Optional[Union[str, Path]] = None, **kwargs: Any) -> None:
        """初始化读取器。

        Args:
            filepath: 要读取的文件或目录路径
            **kwargs: 额外的元数据覆盖
        """
        self.filepath = ""
        self.sample_name: str | None = None
        self.start_time: str | None = None
        self.operator: str | None = None
        self.active_material_mass: str | None = None
        self.wave_number: str | None = None
        self.__dict__["instrument"] = str(getattr(type(self), "instrument", "unknown"))
        self.technique = list(getattr(self, "technique", ["unknown"]))
        self.reader_options: dict[str, Any] = {}

        for key, value in kwargs.items():
            normalized_value = value
            if key == "technique" and isinstance(value, str):
                normalized_value = [value]
            if key == "instrument":
                self.__dict__["instrument"] = str(normalized_value)
            elif hasattr(self, key):
                setattr(self, key, normalized_value)
            else:
                # reader 专有参数仍会通过 load(**kwargs) 传入具体实现。
                self.reader_options[key] = normalized_value

        if filepath is not None:
            self.filepath = str(filepath)

    def load(self, **kwargs: Any) -> DataBundle:
        """从文件或目录加载数据。

        Args:
            **kwargs: 特定读取器的额外参数

        Returns:
            DataBundle 数据包

        Raises:
            ValueError: 如果未设置 filepath
            FileNotFoundError: 如果路径不存在
            ValueError: 如果路径既不是文件也不是目录
            RuntimeError: 如果目录加载失败
        """
        if not self.filepath:
            raise ValueError("调用 load() 前必须设置 filepath。")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")

        if path.is_file():
            return self._load_single_file(path, **kwargs)
        if path.is_dir():
            if not self.supports_directories:
                raise ValueError(f"{self.__class__.__name__} 不支持目录加载。")
            return self._load_directory(path, **kwargs)

        raise ValueError(f"路径既不是文件也不是目录: {path}")

    def _load_single_file(self, path: Path, **kwargs: Any) -> DataBundle:
        """加载单个文件。

        必须由子类实现。

        Args:
            path: 文件路径
            **kwargs: 额外的参数

        Returns:
            DataBundle 数据包

        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} 必须实现 _load_single_file()。")

    def _load_directory(self, path: Path, **kwargs: Any) -> DataBundle:
        """将目录中所有支持的文件加载到 DataTree 中。

        目录加载的默认实现。子类可以覆盖此方法以实现自定义行为。

        Args:
            path: 目录路径
            **kwargs: 额外的参数

        Returns:
            DataBundle，内部数据为 DataTree

        Raises:
            FileNotFoundError: 如果未找到支持的文件
            RuntimeError: 如果所有文件加载失败
        """
        extensions = self._extensions()
        files = sorted({file for ext in extensions for file in path.rglob(f"*{ext}")})

        if not files:
            patterns = ", ".join(f"*{ext}" for ext in extensions)
            raise FileNotFoundError(f"在 {path} 中未找到匹配 {patterns} 的文件。")

        tree_dict: dict[str, Any] = {}
        metadata_items: list[Metadata] = []

        for f in files:
            try:
                node_path, dataset, metadata = self._load_directory_file(f, path, **kwargs)
                tree_dict[node_path] = dataset
                metadata_items.append(metadata)
            except FileNotFoundError as e:
                logger.error("文件不存在: %s - %s", f, e)
            except PermissionError as e:
                logger.error("权限不足，无法读取文件: %s - %s", f, e)
            except (ValueError, KeyError) as e:
                # 数据格式错误或缺少必要字段
                logger.warning("文件格式错误，跳过: %s - %s", f, e)
            except (OSError, TypeError) as e:
                # 其他 I/O 错误或类型错误
                logger.warning("读取文件失败，跳过: %s - %s", f, e, exc_info=True)
            except Exception as e:
                # 未预期的错误, 记录完整堆栈
                logger.exception("未预期的错误加载 %s: %s", f, e)

        if not tree_dict:
            patterns = ", ".join(f"*{ext}" for ext in extensions)
            raise RuntimeError(f"未能从 {path} 加载任何匹配 {patterns} 的文件。")

        # 创建 DataTree
        tree = xr.DataTree.from_dict(tree_dict, name=path.name)

        # 合并元数据
        merged_metadata = merge_metadata(
            metadata_items,
            path,
            sample_name_override=self.sample_name,
            operator_override=self.operator,
            start_time_override=self.start_time,
            active_material_mass_override=self.active_material_mass,
            wave_number_override=self.wave_number,
            technique=list(self.technique) if self.technique != ["unknown"] else None,
            instrument=self.instrument,
        )

        return DataBundle(data=tree, meta=merged_metadata, provenance={"source_path": str(path), "reader": self.__class__.__name__})

    def _load_directory_file(self, file_path: Path, root: Path, **kwargs: Any) -> tuple[str, xr.Dataset, Metadata]:
        """加载单个文件并返回 DataTree 节点路径、Dataset 和 Metadata。"""
        bundle = self._load_single_file(file_path, **kwargs)
        if not isinstance(bundle.data, xr.Dataset):
            raise ValueError(f"{self.__class__.__name__} 的目录子项必须加载为 xarray.Dataset。")

        dataset = sanitize_variable_names(bundle.data)
        if not isinstance(dataset, xr.Dataset):
            raise ValueError(f"{self.__class__.__name__} 的目录子项清理后必须仍为 xarray.Dataset。")

        rel_path = file_path.relative_to(root).with_suffix("")
        node_path = "/" + "/".join(rel_path.parts)
        return node_path, dataset, bundle.meta

    @classmethod
    def _extensions(cls) -> tuple[str, ...]:
        """返回 reader 声明的文件扩展名。"""
        spec = getattr(cls, "spec", None)
        if isinstance(spec, ReaderSpec) and spec.extensions:
            return spec.extensions
        raise ValueError(f"{cls.__name__} 必须声明 ReaderSpec.extensions。")

    def _create_metadata(
        self,
        metadata: dict[str, Any],
        default_sample_name: str,
        technique_override: Optional[list[str]] = None,
    ) -> Metadata:
        """从原始元数据和 reader 配置创建 Metadata。

        Args:
            metadata: 文件中提取的原始元数据字典
            default_sample_name: 默认样本名
            technique_override: 技术类型覆盖列表

        Returns:
            Metadata 对象
        """
        tech_list = technique_override or list(self.technique)

        return Metadata(
            sample_name=self.sample_name or metadata.get("sample_name", default_sample_name),
            start_time=self.start_time or metadata.get("start_time"),
            operator=self.operator or metadata.get("operator"),
            technique=tech_list,
            instrument=self.instrument,
            active_material_mass=self.active_material_mass or metadata.get("active_material_mass"),
            wave_number=self.wave_number or metadata.get("wave_number"),
            raw_metadata=metadata,
        )


__all__ = ["BaseReader"]
