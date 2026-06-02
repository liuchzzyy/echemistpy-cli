"""Shared data structures for both :mod:`xarray` and NeXus containers."""

from __future__ import annotations

from typing import Any, Optional, TypeVar

import pandas as pd
import xarray as xr
from traitlets import Dict, HasTraits, Instance, List, Unicode, Union

T = TypeVar("T", bound="BaseInfo")


class MetadataInfoMixin:
    """Mixin providing common metadata operations for Info classes.

    This mixin provides shared functionality for RawDataInfo and ResultsDataInfo
    classes using traitlets.
    """

    def to_dict(self) -> dict[str, Any]:
        """将元数据转换为字典表示。

        Returns:
            包含所有元数据字段的字典（排除 None 值）
        """
        # trait_values() 返回实例的所有 trait 值
        # 无需 cast，因为 self 继承自 HasTraits
        trait_values = self.trait_values()  # type: ignore
        return {k: v for k, v in trait_values.items() if v is not None}

    def get(self, key: str, default: Any = None) -> Any:
        """通过键获取元数据值。

        首先检查标准字段，然后检查动态存储（parameters、others）。

        Args:
            key: 要检索的元数据键
            default: 如果键未找到时的默认值

        Returns:
            元数据值或默认值
        """
        # 首先检查标准字段
        if self.has_trait(key):  # type: ignore
            return getattr(self, key)

        # 检查动态存储位置（parameters 或 others，如果存在）
        for container_name in ["parameters", "others"]:
            if hasattr(self, container_name):
                container = getattr(self, container_name, None)
                if isinstance(container, dict) and key in container:
                    return container[key]

        return default

    def update(self, other: dict[str, Any]) -> None:
        """用新的键值对更新元数据。

        如果存在标准字段则更新，否则更新相应的动态存储（parameters 或 others，如果存在）。

        Args:
            other: 要添加/更新的元数据字典
        """
        # 确定动态容器（优先使用 parameters）
        container = None
        if hasattr(self, "parameters"):
            container = self.parameters
        elif hasattr(self, "others"):
            container = self.others

        for key, value in other.items():
            if self.has_trait(key):  # type: ignore
                setattr(self, key, value)
            elif container is not None and isinstance(container, dict):
                container[key] = value  # type: ignore


class XarrayDataMixin:
    """提供通用 xarray.Dataset 和 xarray.DataTree 操作的 Mixin。

        此 Mixin 通过为 RawData 和 ResultsData 类提供共享功能来消除代码重复，
    支持 Dataset 和 DataTree 两种后端。
    """

    data: xr.Dataset | xr.DataTree  # IDE 类型提示支持

    def copy(self, deep: bool = True) -> Any:
        """创建数据对象的副本。

        Args:
            deep: 是否对底层数据执行深拷贝

        Returns:
            具有复制数据的同类新实例
        """
        # 使用 type(self) 避免硬编码类名
        return type(self)(data=self.data.copy(deep=deep))

    def to_pandas(self) -> pd.DataFrame | pd.Series:
        """将数据转换为 pandas DataFrame 或 Series。

        对于 Dataset，这包装了 xarray 的 to_pandas()。
        对于 DataTree，返回根数据集的 pandas 表示。

        Returns:
            pandas.DataFrame 或 pandas.Series

        Raises:
            ValueError: 如果数据超过 1 维或是没有根数据的 DataTree
        """
        ds = self.data
        if isinstance(ds, xr.DataTree):
            if ds.dataset is None:
                raise ValueError("DataTree has no root dataset to convert to pandas.")
            ds = ds.dataset

        # 检查维度数
        n_dims = len(ds.dims)

        if n_dims > 1:
            raise ValueError(
                f"to_pandas() only works for data with 1 or fewer dimensions. This data has {n_dims} dimensions: {list(ds.dims.keys())}. Use self.data.to_dataframe() for multi-dimensional data."
            )

        return ds.to_pandas()

    @property
    def variables(self) -> list[str]:
        """获取根数据集中所有变量名的列表。

        Returns:
            变量名列表
        """
        ds = self.data
        if isinstance(ds, xr.DataTree):
            if ds.dataset is None:
                return []
            ds = ds.dataset
        return [str(k) for k in ds.data_vars]

    @property
    def coords(self) -> list[str]:
        """获取根数据集中所有坐标名的列表。

        Returns:
            坐标名列表
        """
        ds = self.data
        if isinstance(ds, xr.DataTree):
            if ds.dataset is None:
                return []
            ds = ds.dataset
        return [str(k) for k in ds.coords]

    def get_variables(self) -> list[str]:
        """获取数据集中所有变量名的列表。"""
        return self.variables

    def get_coords(self) -> list[str]:
        """获取数据集中所有坐标名的列表。"""
        return self.coords

    def select(self, variables: Optional[list[str]] = None) -> xr.Dataset | xr.DataTree:
        """从数据集中选择特定变量。

        Args:
            variables: 要选择的变量名列表，或 None 表示全部

        Returns:
            包含所选变量的 xarray.Dataset 或 xarray.DataTree
        """
        if variables is None:
            return self.data

        # 分别处理 Dataset 和 DataTree
        if isinstance(self.data, xr.Dataset):
            result = self.data[variables]
            if isinstance(result, xr.DataArray):
                return result.to_dataset()
            return result
        else:
            # 对于 DataTree，使用 list[str] 索引选择节点
            # 这里需要使用 Any 作为中间类型，因为 mypy 无法推断 DataTree 的索引返回类型
            return self.data[variables]  # type: ignore[return-value]

    def __getitem__(self, key: str) -> xr.DataArray | xr.DataTree:
        """按名称访问变量或节点。

        Args:
            key: 变量或节点名称

        Returns:
            xarray.DataArray 或 xarray.DataTree
        """
        return self.data[key]


class BaseInfo(HasTraits, MetadataInfoMixin):
    """Base class for all metadata info containers."""

    technique = List(Unicode(), default_value=["Unknown"])
    sample_name = Unicode("Unknown")
    start_time = Unicode(None, allow_none=True)
    operator = Unicode(None, allow_none=True)
    instrument = Unicode(None, allow_none=True)
    active_material_mass = Unicode(None, allow_none=True)
    wave_number = Unicode(None, allow_none=True)

    def copy(self: T) -> T:
        """Create a copy of the info object.

        Returns:
            A new instance of the same class with copied metadata
        """
        return self.__class__(**self.to_dict())

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with standard metadata defaults.

        Args:
            **kwargs: Trait values or metadata to be stored in dynamic containers.
        """
        # Separate traits from other metadata
        trait_names = self.trait_names()
        traits = {k: v for k, v in kwargs.items() if k in trait_names}
        others_dict = {k: v for k, v in kwargs.items() if k not in trait_names}

        super().__init__(**traits)

        # Initialize dynamic storage if present
        for container_name in ["parameters", "others"]:
            if hasattr(self, container_name):
                container = getattr(self, container_name)
                if container is None:
                    setattr(self, container_name, {})
                    container = getattr(self, container_name)

                if isinstance(container, dict):
                    # Add extra kwargs that weren't traits
                    container.update(others_dict)


class BaseData(HasTraits, XarrayDataMixin):
    """Base class for all xarray-based data containers."""

    data = Union([Instance(xr.Dataset), Instance(xr.DataTree)], help="xarray.Dataset or xarray.DataTree containing the data")

    @property
    def is_tree(self) -> bool:
        """Check if the data is an xarray.DataTree.

        Returns:
            True if data is a DataTree, False if it is a Dataset
        """
        return isinstance(self.data, xr.DataTree)


class RawDataInfo(BaseInfo):
    """Container for all metadata extracted from the file.

    This stores metadata with standardized keys (technique, sample_name, etc.)
    while keeping original instrument-specific metadata in the 'others' dictionary.
    """

    others = Dict(help="Dictionary containing all metadata key-value pairs")


class RawData(BaseData):
    """Container for measurement data using xarray.Dataset or xarray.DataTree.

    This represents the data loaded from a file, standardized with consistent
    column names and units. For hierarchical data (e.g., multiple scans or
    subfolders), an xarray.DataTree is used.
    """

    pass


class ResultsDataInfo(BaseInfo):
    """Metadata for analysis results.

    This class stores metadata about data analysis and processing, including
    parameters used, remarks about the analysis, and any additional metadata.

    Attributes:
        parameters: Dictionary of analysis parameters and settings
        others: Additional metadata not covered by standard fields
    """

    parameters = Dict(help="Dictionary of analysis parameters and settings")
    others = Dict(help="Additional metadata not covered by standard fields")


class ResultsData(BaseData):
    """Container for processed results data using xarray.Dataset or xarray.DataTree.

    This represents data after analysis and processing, such as fitted parameters,
    derived metrics, or processed signals. The xarray backend provides powerful
    data manipulation and visualization capabilities.

    Metadata about the analysis is stored separately in ResultsDataInfo for clean
    separation of data and metadata.
    """

    pass


class AnalysisDataInfo(BaseInfo):
    """分析结果的元数据容器.

    存储分析过程中的所有参数和配置，包括：
    - 数据列选择（时间、电位、电流、容量）
    - 归一化参数
    - 库伦效率计算配置
    - 其他分析参数

    Attributes:
        parameters: 分析参数字典，记录所有分析过程使用的配置

    Note:
        与 RawDataInfo 不同，AnalysisDataInfo 不包含 'others' 字段。
        所有分析相关的信息都应存储在 parameters 中。
    """

    parameters = Dict(help="Dictionary of analysis parameters and configuration")


class AnalysisData(BaseData):
    """分析后的数据容器.

    包含分析后的结果数据，如：
    - 归一化后的时间/容量序列
    - 库伦效率数据
    - 其他计算结果

    使用 xarray.Dataset 或 xarray.DataTree 存储数据，提供强大的
    数据操作和可视化能力。分析元数据单独存储在 AnalysisDataInfo 中。
    """

    pass


__all__ = [
    "AnalysisData",
    "AnalysisDataInfo",
    "RawData",
    "RawDataInfo",
    "ResultsData",
    "ResultsDataInfo",
]
