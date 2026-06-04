"""电化学循环数据处理工具。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import xarray as xr

from echemistpy.analysis.echem.columns import primary_dimension


def has_cycle_column(ds: xr.Dataset) -> bool:
    """判断数据集中是否有循环编号列。"""
    return "cycle_number" in ds.data_vars or "cycle_number" in ds.coords


def cycle_numbers(ds: xr.Dataset) -> np.ndarray:
    """返回循环编号数组。"""
    if not has_cycle_column(ds):
        raise ValueError("数据集中未找到 cycle_number 列。")
    return np.asarray(ds["cycle_number"].values)


def split_by_cycle(ds: xr.Dataset) -> dict[int, xr.Dataset]:
    """按 cycle_number 将 Dataset 切分为多个循环。"""
    values = cycle_numbers(ds)
    dim = primary_dimension(ds, "cycle_number")
    cycles: dict[int, xr.Dataset] = {}
    for cycle in np.unique(values):
        if np.isnan(cycle):
            continue
        cycle_id = int(cycle)
        cycles[cycle_id] = ds.isel({dim: values == cycle})
    return cycles


def pad_cycle_arrays(
    cycles: Mapping[int, xr.Dataset],
    getter: Callable[[xr.Dataset, int], Sequence[float] | np.ndarray],
) -> np.ndarray:
    """将不等长循环序列补 NaN 后组成二维数组。"""
    cycle_ids = sorted(cycles)
    if not cycle_ids:
        return np.empty((0, 0), dtype=float)

    arrays = [np.asarray(getter(cycles[cycle_id], cycle_id), dtype=float) for cycle_id in cycle_ids]
    max_len = max((len(array) for array in arrays), default=0)
    result = np.full((len(cycle_ids), max_len), np.nan, dtype=float)
    for index, array in enumerate(arrays):
        result[index, : len(array)] = array
    return result


__all__ = [
    "cycle_numbers",
    "has_cycle_column",
    "pad_cycle_arrays",
    "split_by_cycle",
]
