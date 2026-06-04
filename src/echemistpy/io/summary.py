"""Data inspection helpers for loaded datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from echemistpy.data.schema import RAW_SCHEMA
from echemistpy.io.loaders import load
from echemistpy.io.structures import RawData, RawDataInfo


@dataclass(frozen=True)
class DataSummary:
    """Small JSON-safe summary for one loaded dataset."""

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
        """Return a JSON-safe dictionary."""
        return asdict(self)


def summarize_data(
    raw_data: RawData,
    raw_info: RawDataInfo,
    path: str | Path,
    schema: str = RAW_SCHEMA,
) -> DataSummary:
    """Build an inspection summary from loaded data."""
    dataset = _root_dataset(raw_data.data)
    return DataSummary(
        path=str(path),
        schema=schema,
        sample_name=raw_info.sample_name,
        instrument=raw_info.instrument,
        technique=tuple(raw_info.technique),
        is_tree=raw_data.is_tree,
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
    """Load a path and return a concise data summary."""
    raw_data, raw_info = load(
        path,
        fmt=fmt,
        instrument=instrument,
        standardize=standardize,
    )
    return summarize_data(raw_data, raw_info, path)


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
