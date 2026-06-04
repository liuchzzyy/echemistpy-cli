"""统一数据容器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
import xarray as xr

from echemistpy.data.schema import ANALYSIS_SCHEMA, RAW_SCHEMA


@dataclass
class Metadata:
    """数据集元数据。

    标准字段放在一层属性中；仪器原始元数据放入 ``raw_metadata``。
    """

    technique: list[str] = field(default_factory=lambda: ["unknown"])
    sample_name: str = "Unknown"
    start_time: str | None = None
    operator: str | None = None
    instrument: str | None = None
    active_material_mass: str | None = None
    wave_number: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转成去除空值的字典。"""
        return {key: value for key, value in asdict(self).items() if value is not None and value != {}}

    def copy(self) -> "Metadata":
        """复制元数据。"""
        return Metadata(**self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        """按字段名或原始元数据键读取值。"""
        if hasattr(self, key):
            return getattr(self, key)
        return self.raw_metadata.get(key, default)

    def update(self, values: dict[str, Any]) -> None:
        """更新标准字段；未知键写入原始元数据。"""
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.raw_metadata[key] = value


@dataclass
class DataBundle:
    """标准原始数据包。"""

    data: xr.Dataset | xr.DataTree
    meta: Metadata = field(default_factory=Metadata)
    schema: str = RAW_SCHEMA
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_tree(self) -> bool:
        """是否为 DataTree。"""
        return isinstance(self.data, xr.DataTree)

    @property
    def variables(self) -> list[str]:
        """根数据集变量名。"""
        return [str(name) for name in _root_dataset(self.data).data_vars]

    @property
    def coords(self) -> list[str]:
        """根数据集坐标名。"""
        return [str(name) for name in _root_dataset(self.data).coords]

    def copy(self, deep: bool = True) -> "DataBundle":
        """复制数据包。"""
        return DataBundle(
            data=self.data.copy(deep=deep),
            meta=self.meta.copy(),
            schema=self.schema,
            provenance=dict(self.provenance),
            warnings=list(self.warnings),
        )

    def select(self, variables: list[str] | None = None) -> xr.Dataset | xr.DataTree:
        """选择变量；``None`` 表示返回完整数据。"""
        if variables is None:
            return self.data
        if isinstance(self.data, xr.Dataset):
            result = self.data[variables]
            return result.to_dataset() if isinstance(result, xr.DataArray) else result

        def _select_node(ds: xr.Dataset) -> xr.Dataset:
            existing = [name for name in variables if name in ds.data_vars or name in ds.coords]
            result = ds[existing] if existing else xr.Dataset()
            return result.to_dataset() if isinstance(result, xr.DataArray) else result

        return self.data.map_over_datasets(_select_node)

    def to_pandas(self) -> pd.DataFrame | pd.Series:
        """将根数据集转换为 pandas 对象。"""
        ds = _root_dataset(self.data)
        if len(ds.dims) > 1:
            raise ValueError(f"to_pandas() 只支持 0 或 1 维数据，当前维度为: {list(ds.dims.keys())}")
        return ds.to_pandas()

    def __getitem__(self, key: str) -> xr.DataArray | xr.DataTree:
        """按名称访问变量或节点。"""
        return self.data[key]


@dataclass
class AnalysisBundle:
    """标准分析结果包。"""

    data: xr.Dataset | xr.DataTree
    meta: Metadata = field(default_factory=Metadata)
    schema: str = ANALYSIS_SCHEMA
    provenance: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _root_dataset(data: xr.Dataset | xr.DataTree) -> xr.Dataset:
    """返回 Dataset 或 DataTree 的第一个有数据节点。"""
    if isinstance(data, xr.Dataset):
        return data
    if data.dataset is not None and (data.dataset.data_vars or data.dataset.sizes):
        return data.dataset
    for child in data.children.values():
        return _root_dataset(child)
    return xr.Dataset()


__all__ = [
    "AnalysisBundle",
    "DataBundle",
    "Metadata",
]
