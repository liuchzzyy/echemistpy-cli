"""电化学 GCD 分析器。"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from echemistpy.analysis.echem.capacity import CoulombicEfficiencyColumns, calculate_coulombic_efficiency
from echemistpy.analysis.echem.columns import (
    CAPACITY_COLUMNS,
    CURRENT_COLUMNS,
    POTENTIAL_COLUMNS,
    TIME_COLUMNS,
    capacity_to_mah,
    numeric_array,
    pick_column,
    primary_dimension,
    require_column,
    root_dataset,
)
from echemistpy.analysis.echem.cycles import has_cycle_column, pad_cycle_arrays, split_by_cycle
from echemistpy.analysis.registry import TechniqueAnalyzer
from echemistpy.data.models import AnalysisBundle, DataBundle


class GCDAnalyzer(TechniqueAnalyzer):
    """分析恒流充放电数据并生成绘图友好的结果。"""

    technique = "gcd"
    supported_techniques = ("gcd", "gpcl", "galvanostatic", "echem")
    name = "GCDAnalyzer"

    time_columns = TIME_COLUMNS
    potential_columns = POTENTIAL_COLUMNS
    current_columns = CURRENT_COLUMNS
    capacity_columns = CAPACITY_COLUMNS
    normalization_range = (0.0, 1.0)
    ce_order = "discharge"
    ce_min_denominator = 1e-6

    @property
    def required_columns(self) -> tuple[str, ...]:
        """返回动态校验使用的首选列。"""
        return ("time_s", "current_ma")

    def validate(self, bundle: DataBundle) -> None:
        """验证数据中存在时间列和电流列。"""
        ds = root_dataset(bundle)
        require_column(ds, self.time_columns, "时间")
        require_column(ds, self.current_columns, "电流")
        require_column(ds, self.capacity_columns, "容量")

    def preprocess(self, bundle: DataBundle) -> DataBundle:
        """按时间排序，并确保存在数值型 time_s 坐标。"""
        ds = root_dataset(bundle)
        time_key = require_column(ds, self.time_columns, "时间")
        current_key = require_column(ds, self.current_columns, "电流")
        dim = primary_dimension(ds, current_key)
        time_values = numeric_array(ds, time_key)

        if time_values.size > 1 and not np.all(np.diff(time_values) >= 0):
            ds = ds.sortby(time_key)
            time_values = numeric_array(ds, time_key)

        if "time_s" not in ds.coords:
            ds = ds.assign_coords(time_s=(dim, time_values))

        bundle.data = ds
        return bundle

    def _compute(self, bundle: DataBundle) -> AnalysisBundle:
        """执行 GCD 分析并返回标准 AnalysisBundle。"""
        ds = root_dataset(bundle)
        time_key = require_column(ds, self.time_columns, "时间")
        current_key = require_column(ds, self.current_columns, "电流")
        potential_key = pick_column(ds, self.potential_columns)
        capacity_key = require_column(ds, self.capacity_columns, "容量")

        if has_cycle_column(ds):
            result_ds = self._compute_cycle_dataset(ds, time_key, current_key, potential_key, capacity_key)
        else:
            result_ds = self._compute_1d_dataset(ds, time_key, current_key, potential_key, capacity_key)

        parameters = {
            "analyzer": self.name,
            "normalization_range": list(self.normalization_range),
            "ce_order": self.ce_order,
            "used_columns": {
                "time_column": time_key,
                "current_column": current_key,
                "potential_column": potential_key,
                "capacity_column": capacity_key,
            },
        }
        return AnalysisBundle(data=result_ds, meta=bundle.meta.copy(), parameters=parameters)

    def _compute_cycle_dataset(
        self,
        ds: xr.Dataset,
        time_key: str,
        current_key: str,
        potential_key: str | None,
        capacity_key: str,
    ) -> xr.Dataset:
        """按循环构建二维分析结果。"""
        cycles = split_by_cycle(ds)
        cycle_ids = sorted(cycles)
        time_matrix = pad_cycle_arrays(cycles, lambda cycle_ds, _cycle_id: numeric_array(cycle_ds, time_key))
        current_matrix = pad_cycle_arrays(cycles, lambda cycle_ds, _cycle_id: numeric_array(cycle_ds, current_key))
        capacity_matrix = self._capacity_matrix(cycles, capacity_key)

        result_vars: dict[str, Any] = {
            "time_s": (["cycle_number", "record"], time_matrix),
            current_key: (["cycle_number", "record"], current_matrix),
            "capacity_mah": (["cycle_number", "record"], capacity_matrix),
            "normalized_time": (["cycle_number", "record"], _normalize(time_matrix, *self.normalization_range)),
            "normalized_capacity": (["cycle_number", "record"], _normalize(capacity_matrix, *self.normalization_range)),
        }
        if potential_key:
            result_vars["voltage_v"] = (["cycle_number", "record"], pad_cycle_arrays(cycles, lambda cycle_ds, _cycle_id: numeric_array(cycle_ds, potential_key)))

        ce = calculate_coulombic_efficiency(
            ds,
            columns=CoulombicEfficiencyColumns(current=current_key, capacity=capacity_key),
            order=self.ce_order,
            min_denominator=self.ce_min_denominator,
        )
        if not ce.empty:
            ce = ce.set_index("cycle_number").reindex(cycle_ids)
            result_vars["charge_capacity_mah"] = (["cycle_number"], ce["charge_capacity_mah"].to_numpy(dtype=float))
            result_vars["discharge_capacity_mah"] = (["cycle_number"], ce["discharge_capacity_mah"].to_numpy(dtype=float))
            result_vars["coulombic_efficiency_percent"] = (["cycle_number"], ce["coulombic_efficiency_percent"].to_numpy(dtype=float))

        return xr.Dataset(result_vars, coords={"cycle_number": cycle_ids, "record": np.arange(time_matrix.shape[1])})

    def _compute_1d_dataset(
        self,
        ds: xr.Dataset,
        time_key: str,
        current_key: str,
        potential_key: str | None,
        capacity_key: str,
    ) -> xr.Dataset:
        """构建无循环编号的一维分析结果。"""
        dim = primary_dimension(ds, current_key)
        time_values = numeric_array(ds, time_key)
        current_values = numeric_array(ds, current_key)
        capacity_values = capacity_to_mah(numeric_array(ds, capacity_key), capacity_key)

        result_vars: dict[str, Any] = {
            "time_s": ([dim], time_values),
            current_key: ([dim], current_values),
            "capacity_mah": ([dim], capacity_values),
            "normalized_time": ([dim], _normalize(time_values, *self.normalization_range)),
            "normalized_capacity": ([dim], _normalize(capacity_values, *self.normalization_range)),
        }
        if potential_key:
            result_vars["voltage_v"] = ([dim], numeric_array(ds, potential_key))

        return xr.Dataset(result_vars, coords={dim: ds[dim].values if dim in ds.coords else np.arange(time_values.size)})

    @staticmethod
    def _capacity_matrix(
        cycles: dict[int, xr.Dataset],
        capacity_key: str,
    ) -> np.ndarray:
        """返回按循环补齐后的容量矩阵，单位为 mAh。"""
        return pad_cycle_arrays(cycles, lambda cycle_ds, _cycle_id: capacity_to_mah(numeric_array(cycle_ds, capacity_key), capacity_key))


def _normalize(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """将数组归一化到指定范围，忽略 NaN。"""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return array
    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))
    if maximum <= minimum:
        return np.full_like(array, lower, dtype=float)
    return lower + (array - minimum) / (maximum - minimum) * (upper - lower)


__all__ = ["GCDAnalyzer"]
