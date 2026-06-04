"""电化学分析使用的标准列选择工具。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.data.models import AnalysisBundle, DataBundle

TIME_COLUMNS: tuple[str, ...] = ("time_s", "test_time_s", "step_time_s", "systime")
POTENTIAL_COLUMNS: tuple[str, ...] = ("ewe_v", "voltage_v", "ece_v")
CURRENT_COLUMNS: tuple[str, ...] = ("current_ma", "current_ua", "abs_current_ma")
CAPACITY_COLUMNS: tuple[str, ...] = (
    "capacity_mah",
    "capacity_uah",
    "specific_capacity_mah_g",
    "charge_discharge_capacity_mah",
    "charge_capacity_mah",
    "discharge_capacity_mah",
)
EIS_COLUMNS: tuple[str, ...] = ("frequency_hz", "re_z_ohm", "neg_im_z_ohm", "z_mag_ohm", "phase_deg")


def root_dataset(bundle_or_data: DataBundle | AnalysisBundle | xr.Dataset | xr.DataTree) -> xr.Dataset:
    """从数据包或 xarray 对象中取出第一个可用 Dataset。"""
    data = bundle_or_data.data if isinstance(bundle_or_data, DataBundle | AnalysisBundle) else bundle_or_data
    if isinstance(data, xr.Dataset):
        return data
    if data.dataset is not None and (data.dataset.data_vars or data.dataset.sizes):
        return data.dataset
    for child in data.children.values():
        return root_dataset(child)
    raise ValueError("DataTree 中没有可用于电化学分析的 Dataset。")


def available_names(ds: xr.Dataset) -> set[str]:
    """返回 Dataset 中所有变量名和坐标名。"""
    return {str(name) for name in list(ds.data_vars) + list(ds.coords)}


def pick_column(ds: xr.Dataset, candidates: Iterable[str]) -> str | None:
    """从候选列名中返回第一个存在的列。"""
    names = available_names(ds)
    for column in candidates:
        if column in names:
            return column
    return None


def require_column(ds: xr.Dataset, candidates: Iterable[str], label: str) -> str:
    """选择必需列；不存在时抛出带中文说明的错误。"""
    candidate_list = tuple(candidates)
    column = pick_column(ds, candidate_list)
    if column is None:
        raise ValueError(f"未找到{label}列，已搜索: {candidate_list}")
    return column


def primary_dimension(ds: xr.Dataset, column: str | None = None) -> str:
    """返回一维电化学数据的主维度名。"""
    if column and column in ds and ds[column].dims:
        return str(ds[column].dims[0])
    if ds.sizes:
        return str(next(iter(ds.sizes)))
    raise ValueError("Dataset 没有可用维度。")


def numeric_time(values: np.ndarray) -> np.ndarray:
    """将 datetime 或数值时间转换为以秒为单位的浮点数组。"""
    array = np.asarray(values)
    if array.size == 0:
        return array.astype(float)
    if np.issubdtype(array.dtype, np.datetime64):
        time_index = pd.to_datetime(array)
        return (time_index - time_index[0]).total_seconds().to_numpy(dtype=float)
    return array.astype(float)


def numeric_array(ds: xr.Dataset, column: str) -> np.ndarray:
    """读取变量或坐标并转为浮点数组。"""
    if column not in ds:
        raise KeyError(f"Dataset 中不存在列: {column}")
    if column in TIME_COLUMNS:
        return numeric_time(np.asarray(ds[column].values))
    return np.asarray(ds[column].values, dtype=float)


def current_to_ma(values: np.ndarray, column: str) -> np.ndarray:
    """将电流数组统一转换为 mA。"""
    current = np.asarray(values, dtype=float)
    if column.endswith("_ua"):
        return current / 1000.0
    return current


def capacity_to_mah(values: np.ndarray, column: str) -> np.ndarray:
    """将容量数组统一转换为 mAh。"""
    capacity = np.asarray(values, dtype=float)
    if column.endswith("_uah"):
        return capacity / 1000.0
    return capacity


__all__ = [
    "CAPACITY_COLUMNS",
    "CURRENT_COLUMNS",
    "EIS_COLUMNS",
    "POTENTIAL_COLUMNS",
    "TIME_COLUMNS",
    "available_names",
    "capacity_to_mah",
    "current_to_ma",
    "numeric_array",
    "numeric_time",
    "pick_column",
    "primary_dimension",
    "require_column",
    "root_dataset",
]
