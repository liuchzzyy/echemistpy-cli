"""电化学容量和库伦效率计算。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.analysis.echem.columns import (
    capacity_to_mah,
    numeric_array,
    require_column,
)
from echemistpy.analysis.echem.cycles import split_by_cycle


@dataclass(frozen=True)
class CoulombicEfficiencyColumns:
    """库伦效率计算使用的列名。"""

    current: str
    capacity: str


def charge_discharge_totals_from_capacity_column(
    capacity: np.ndarray, current: np.ndarray, capacity_column: str
) -> tuple[float, float]:
    """根据已读取的容量列和电流方向计算充放电容量。"""
    capacity_values = capacity_to_mah(
        np.asarray(capacity, dtype=float), capacity_column
    )
    current_values = np.asarray(current, dtype=float)
    charge = _sum_segment_capacity(capacity_values, current_values > 0)
    discharge = _sum_segment_capacity(capacity_values, current_values < 0)
    return charge, discharge


def calculate_coulombic_efficiency(
    ds: xr.Dataset,
    *,
    columns: CoulombicEfficiencyColumns,
    order: str = "discharge",
    min_denominator: float = 1e-6,
) -> pd.DataFrame:
    """计算每个循环的充电容量、放电容量和库伦效率。"""
    if order not in {"discharge", "charge"}:
        raise ValueError(
            f"库伦效率 order 必须是 'discharge' 或 'charge'，当前为: {order}"
        )

    current_key = require_column(ds, (columns.current,), "电流")
    capacity_key = require_column(ds, (columns.capacity,), "容量")
    cycles = split_by_cycle(ds)
    records = []
    for cycle_id, cycle_ds in cycles.items():
        current = numeric_array(cycle_ds, current_key)
        charge, discharge = charge_discharge_totals_from_capacity_column(
            numeric_array(cycle_ds, capacity_key), current, capacity_key
        )

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


def _sum_segment_capacity(capacity: np.ndarray, mask: np.ndarray) -> float:
    """按连续电流段汇总容量，兼容 step 内容量重置。"""
    total = 0.0
    for start, end in _active_runs(mask):
        segment = capacity[start : end + 1]
        finite_offsets = np.flatnonzero(np.isfinite(segment))
        if finite_offsets.size == 0:
            continue
        start_index = start + int(finite_offsets[0])
        end_index = start + int(finite_offsets[-1])
        baseline = _segment_baseline(capacity, start_index)
        total += abs(float(capacity[end_index] - baseline))
    return float(total)


def _active_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """返回布尔掩码中连续 True 片段的首尾索引。"""
    if not np.any(mask):
        return []
    indices = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]]))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _segment_baseline(capacity: np.ndarray, start_index: int) -> float:
    """返回连续容量段的基线，识别蓝和 CCS 这类 step 重置。"""
    first = float(capacity[start_index])
    if start_index == 0:
        return 0.0

    previous_values = capacity[:start_index]
    finite_previous = previous_values[np.isfinite(previous_values)]
    if finite_previous.size == 0:
        return 0.0

    previous = float(finite_previous[-1])
    if _looks_like_step_reset(first, previous):
        return 0.0
    return previous


def _looks_like_step_reset(first: float, previous: float) -> bool:
    """判断新电流段容量是否从 step 内累计值重新开始。"""
    return abs(previous) > 0.0 and abs(first) <= abs(previous) * 0.5


def _efficiency(
    charge: float, discharge: float, order: str, threshold: float, cycle_id: int
) -> float:
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
    "CoulombicEfficiencyColumns",
    "calculate_coulombic_efficiency",
    "charge_discharge_totals_from_capacity_column",
]
