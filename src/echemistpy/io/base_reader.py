"""所有数据读取器的基类。

本模块为文件读取器提供通用基类，以减少代码重复
并确保一致的行为。

主要功能：
- 文件/目录加载：统一的加载接口
- 元数据管理：使用 traitlets 管理可配置元数据
- 错误处理：细化的异常处理和日志记录
- DataTree 支持：自动将目录文件组织为分层结构
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import xarray as xr
from traitlets import HasTraits, Unicode
from traitlets import List as TList

from echemistpy.io.contracts import ReaderSpec
from echemistpy.io.reader_utils import merge_infos, sanitize_variable_names
from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class BaseReader(HasTraits):
    """所有文件读取器的基类。

    为以下内容提供通用功能：
    - 文件/目录加载
    - 元数据管理
    - 基于 traitlets 的配置

    Class Variables:
        supports_directories: 是否支持目录加载
        instrument: 仪器标识符

    Attributes:
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

    # --- 所有读取器的通用 traitlets ---
    filepath = Unicode(help="文件或目录路径")
    sample_name = Unicode(None, allow_none=True, help="样本名称")
    start_time = Unicode(None, allow_none=True, help="开始时间")
    operator = Unicode(None, allow_none=True, help="操作员名称")
    active_material_mass = Unicode(None, allow_none=True, help="活性物质质量")
    wave_number = Unicode(None, allow_none=True, help="波数")
    technique = TList(Unicode(), default_value=["unknown"], help="技术类型列表")

    def __init__(self, filepath: Optional[Union[str, Path]] = None, **kwargs: Any) -> None:
        """初始化读取器。

        Args:
            filepath: 要读取的文件或目录路径
            **kwargs: 额外的元数据覆盖
        """
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self, **kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """从文件或目录加载数据。

        Args:
            **kwargs: 特定读取器的额外参数

        Returns:
            (RawData, RawDataInfo) 元组

        Raises:
            ValueError: 如果未设置 filepath
            FileNotFoundError: 如果路径不存在
            ValueError: 如果路径既不是文件也不是目录
            RuntimeError: 如果目录加载失败
        """
        if not self.filepath:
            raise ValueError("filepath must be set before calling load()")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return self._load_single_file(path, **kwargs)
        if path.is_dir():
            if not self.supports_directories:
                raise ValueError(f"{self.__class__.__name__} does not support directory loading")
            return self._load_directory(path, **kwargs)

        raise ValueError(f"Path is neither a file nor a directory: {path}")

    def _load_single_file(self, path: Path, **kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """加载单个文件。

        必须由子类实现。

        Args:
            path: 文件路径
            **kwargs: 额外的参数

        Returns:
            (RawData, RawDataInfo) 元组

        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _load_single_file()")

    def _load_directory(self, path: Path, **kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """将目录中所有支持的文件加载到 DataTree 中。

        目录加载的默认实现。子类可以覆盖此方法以实现自定义行为。

        Args:
            path: 目录路径
            **kwargs: 额外的参数

        Returns:
            (RawData with DataTree, merged RawDataInfo) 元组

        Raises:
            FileNotFoundError: 如果未找到支持的文件
            RuntimeError: 如果所有文件加载失败
        """
        extensions = self._extensions()
        files = sorted({file for ext in extensions for file in path.rglob(f"*{ext}")})

        if not files:
            patterns = ", ".join(f"*{ext}" for ext in extensions)
            raise FileNotFoundError(f"No files matching {patterns} found in {path}")

        tree_dict: dict[str, Any] = {}
        infos: list[RawDataInfo] = []

        for f in files:
            try:
                node_path, dataset, raw_info = self._load_directory_file(f, path, **kwargs)
                tree_dict[node_path] = dataset
                infos.append(raw_info)
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
            raise RuntimeError(f"Failed to load any files matching {patterns} from {path}")

        # 创建 DataTree
        tree = xr.DataTree.from_dict(tree_dict, name=path.name)

        # 合并元数据
        merged_info = merge_infos(
            infos,
            path,
            sample_name_override=self.sample_name,
            operator_override=self.operator,
            start_time_override=self.start_time,
            active_material_mass_override=self.active_material_mass,
            wave_number_override=self.wave_number,
            technique=list(self.technique) if self.technique != ["unknown"] else None,
            instrument=self.instrument,
        )

        return RawData(data=tree), merged_info

    def _load_directory_file(self, file_path: Path, root: Path, **kwargs: Any) -> tuple[str, xr.Dataset, RawDataInfo]:
        """Load one file and return its DataTree node path, dataset, and metadata."""
        raw_data, raw_info = self._load_single_file(file_path, **kwargs)
        if not isinstance(raw_data.data, xr.Dataset):
            raise ValueError(f"{self.__class__.__name__} directory items must load as xarray.Dataset")

        dataset = sanitize_variable_names(raw_data.data)
        if not isinstance(dataset, xr.Dataset):
            raise ValueError(f"{self.__class__.__name__} directory items must sanitize to xarray.Dataset")

        rel_path = file_path.relative_to(root).with_suffix("")
        node_path = "/" + "/".join(rel_path.parts)
        return node_path, dataset, raw_info

    @classmethod
    def _extensions(cls) -> tuple[str, ...]:
        """Return declared file extensions for this reader."""
        spec = getattr(cls, "spec", None)
        if isinstance(spec, ReaderSpec) and spec.extensions:
            return spec.extensions
        raise ValueError(f"{cls.__name__} must declare ReaderSpec.extensions")

    def _create_raw_info(
        self,
        metadata: dict[str, Any],
        default_sample_name: str,
        technique_override: Optional[list[str]] = None,
    ) -> RawDataInfo:
        """Create a RawDataInfo object from metadata and traitlets.

        Args:
            metadata: Metadata dictionary from file
            default_sample_name: Default sample name to use
            technique_override: Override technique list

        Returns:
            RawDataInfo object
        """
        tech_list = technique_override or list(self.technique)

        return RawDataInfo(
            sample_name=self.sample_name or metadata.get("sample_name", default_sample_name),
            start_time=self.start_time or metadata.get("start_time"),
            operator=self.operator or metadata.get("operator"),
            technique=tech_list,
            instrument=self.instrument,
            active_material_mass=self.active_material_mass or metadata.get("active_material_mass"),
            wave_number=self.wave_number or metadata.get("wave_number"),
            others=metadata,
        )


__all__ = ["BaseReader"]
