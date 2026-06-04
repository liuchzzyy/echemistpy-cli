"""电化学容量和库伦效率计算。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.analysis.echem.columns import capacity_to_mah, current_to_ma, numeric_array, require_column
from echemistpy.analysis.echem.cycles import split_by_cycle


@dataclass(frozen=True)
class EfficiencyColumns:
    """库伦效率计算使用的列名。"""

    time: str
    current: str
    capacity: str | None = None


def integrate_capacity_mah(time_s: np.ndarray, current: np.ndarray, current_column: str) -> np.ndarray:
    """通过电流积分得到累计容量，单位为 mAh。"""
    time_values = np.asarray(time_s, dtype=float)
    current_ma = current_to_ma(np.asarray(current, dtype=float), current_column)
    if time_values.size == 0:
        return np.asarray([], dtype=float)
    if time_values.size == 1:
        return np.asarray([0.0], dtype=float)
    dt = np.gradient(time_values)
    return np.cumsum(current_ma * dt / 3600.0)


def signed_capacity_totals(time_s: np.ndarray, current: np.ndarray, current_column: str) -> tuple[float, float]:
    """按电流正负积分得到充电和放电容量。"""
    time_values = np.asarray(time_s, dtype=float)
    current_ma = current_to_ma(np.asarray(current, dtype=float), current_column)
    if time_values.size < 2:
        return 0.0, 0.0
    dt = np.gradient(time_values)
    increments = current_ma * dt / 3600.0
    charge = float(np.nansum(increments[increments > 0]))
    discharge = float(abs(np.nansum(increments[increments < 0])))
    return charge, discharge


def capacity_totals_from_column(capacity: np.ndarray, current: np.ndarray, capacity_column: str) -> tuple[float, float]:
    """根据容量列和电流方向估算充放电容量。"""
    capacity_values = capacity_to_mah(np.asarray(capacity, dtype=float), capacity_column)
    current_values = np.asarray(current, dtype=float)
    charge_mask = current_values > 0
    discharge_mask = current_values < 0
    charge = _span_capacity(capacity_values, charge_mask)
    discharge = abs(_span_capacity(capacity_values, discharge_mask))
    return charge, discharge


def calculate_coulombic_efficiency(
    ds: xr.Dataset,
    *,
    columns: EfficiencyColumns,
    order: str = "discharge",
    min_denominator: float = 1e-6,
) -> pd.DataFrame:
    """计算每个循环的充电容量、放电容量和库伦效率。"""
    if order not in {"discharge", "charge"}:
        raise ValueError(f"库伦效率 order 必须是 'discharge' 或 'charge'，当前为: {order}")

    current_key = require_column(ds, (columns.current,), "电流")
    cycles = split_by_cycle(ds)
    records = []
    for cycle_id, cycle_ds in cycles.items():
        current = numeric_array(cycle_ds, current_key)
        if columns.capacity and columns.capacity in cycle_ds:
            charge, discharge = capacity_totals_from_column(numeric_array(cycle_ds, columns.capacity), current, columns.capacity)
        else:
            charge, discharge = signed_capacity_totals(numeric_array(cycle_ds, columns.time), current, current_key)

        efficiency = _efficiency(charge, discharge, order, min_denominator, cycle_id)
        records.append(
            {
                "cycle_number": cycle_id,
                "charge_capacity_mah": charge,
                "discharge_capacity_mah": discharge,
                "coulombic_efficiency_percent": efficiency,
            }
        )

    return pd.DataFrame(records)


def _span_capacity(capacity: np.ndarray, mask: np.ndarray) -> float:
    """返回某一阶段容量首尾差值。"""
    if not np.any(mask):
        return 0.0
    indices = np.flatnonzero(mask)
    return float(capacity[indices[-1]] - capacity[indices[0]])


def _efficiency(charge: float, discharge: float, order: str, threshold: float, cycle_id: int) -> float:
    """根据容量和计算方向返回库伦效率。"""
    if order == "discharge":
        numerator = discharge
        denominator = charge
        label = "充电容量"
    else:
        numerator = charge
        denominator = discharge
        label = "放电容量"

    if denominator > threshold:
        return float(numerator / denominator * 100.0)

    warnings.warn(
        f"循环 {cycle_id} 的{label} ({denominator:.6g} mAh) 低于阈值 ({threshold:g} mAh)，库伦效率设为 NaN。",
        stacklevel=3,
    )
    return float("nan")


__all__ = [
    "EfficiencyColumns",
    "calculate_coulombic_efficiency",
    "capacity_totals_from_column",
    "integrate_capacity_mah",
    "signed_capacity_totals",
]
