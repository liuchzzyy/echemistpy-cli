"""已加载数据的检查摘要工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from echemistpy.data.models import DataBundle
from echemistpy.io.loaders import load


@dataclass(frozen=True)
class DataSummary:
    """单个已加载数据集的 JSON 安全摘要。"""

    path: str
    schema: str
    sample_name: str
    instrument: str | None
    technique: tuple[str, ...]
    is_tree: bool
    dims: dict[str, int]
    variables: tuple[str, ...]
    coords: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 安全字典。"""
        return asdict(self)


def summarize_data(
    bundle: DataBundle,
    path: str | Path,
) -> DataSummary:
    """根据 DataBundle 生成检查摘要。"""
    dataset = _root_dataset(bundle.data)
    return DataSummary(
        path=str(path),
        schema=bundle.schema,
        sample_name=bundle.meta.sample_name,
        instrument=bundle.meta.instrument,
        technique=tuple(bundle.meta.technique),
        is_tree=isinstance(bundle.data, xr.DataTree),
        dims={str(name): int(size) for name, size in dataset.sizes.items()},
        variables=tuple(str(name) for name in dataset.data_vars),
        coords=tuple(str(name) for name in dataset.coords),
    )


def inspect_data(
    path: str | Path,
    *,
    fmt: str | None = None,
    instrument: str | None = None,
    standardize: bool = True,
) -> DataSummary:
    """加载路径并返回简洁的数据摘要。"""
    bundle = load(
        path,
        fmt=fmt,
        instrument=instrument,
        standardize=standardize,
    )
    return summarize_data(bundle, path)


def _root_dataset(data: xr.Dataset | xr.DataTree) -> xr.Dataset:
    if isinstance(data, xr.Dataset):
        return data
    if data.dataset is not None and (data.dataset.data_vars or data.dataset.sizes):
        return data.dataset
    for child in data.children.values():
        return _root_dataset(child)
    return xr.Dataset()


__all__ = [
    "DataSummary",
    "inspect_data",
    "summarize_data",
]
