"""电化学绘图公共工具。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from echemistpy.analysis.echem.columns import numeric_array, pick_column, root_dataset
from echemistpy.analysis.echem.cycles import has_cycle_column, split_by_cycle
from echemistpy.data.models import AnalysisBundle, DataBundle


@dataclass(frozen=True)
class CyclePlotOptions:
    """循环曲线绘图选项。"""

    cycles: Iterable[int] | None = None
    max_cycles: int | None = 8
    label_prefix: str = "cycle"


def dataset(bundle: DataBundle | AnalysisBundle) -> xr.Dataset:
    """返回绘图使用的根 Dataset。"""
    return root_dataset(bundle)


def require_plot_column(ds: xr.Dataset, candidates: Iterable[str], label: str) -> str:
    """选择绘图所需列。"""
    candidate_tuple = tuple(candidates)
    column = pick_column(ds, candidate_tuple)
    if column is None:
        raise ValueError(f"绘图缺少{label}列，已搜索: {candidate_tuple}")
    return column


def plot_xy_by_cycle(
    ax: Axes,
    ds: xr.Dataset,
    keys: tuple[str, str],
    options: CyclePlotOptions | None = None,
) -> list[int]:
    """按循环编号绘制二维或一维数据。"""
    x_key, y_key = keys
    cycle_options = options or CyclePlotOptions()
    if _is_cycle_matrix(ds, x_key, y_key):
        plotted = _plot_cycle_matrix(ax, ds, keys, cycle_options)
    elif has_cycle_column(ds):
        plotted = _plot_split_cycles(ax, ds, keys, cycle_options)
    else:
        ax.plot(numeric_array(ds, x_key), numeric_array(ds, y_key))
        plotted = []
    if plotted and len(plotted) <= 8:
        ax.legend()
    return plotted


def line_count(ax: Axes) -> int:
    """返回坐标轴中的曲线数量。"""
    return len(ax.lines)


def _is_cycle_matrix(ds: xr.Dataset, x_key: str, y_key: str) -> bool:
    """判断变量是否为 cycle_number-record 二维矩阵。"""
    return x_key in ds and y_key in ds and "cycle_number" in ds[x_key].dims and "record" in ds[x_key].dims and "cycle_number" in ds[y_key].dims and "record" in ds[y_key].dims


def _selected_cycles(all_cycles: Iterable[int], cycles: Iterable[int] | None, max_cycles: int | None) -> list[int]:
    """返回需要绘制的循环编号。"""
    selected = sorted(int(cycle) for cycle in (cycles if cycles is not None else all_cycles))
    if max_cycles is not None:
        selected = selected[:max_cycles]
    return selected


def _plot_cycle_matrix(
    ax: Axes,
    ds: xr.Dataset,
    keys: tuple[str, str],
    options: CyclePlotOptions,
) -> list[int]:
    """绘制二维循环矩阵。"""
    x_key, y_key = keys
    available_cycles = [int(cycle) for cycle in np.asarray(ds["cycle_number"].values)]
    selected = _selected_cycles(available_cycles, options.cycles, options.max_cycles)
    for cycle in selected:
        if cycle not in available_cycles:
            continue
        index = available_cycles.index(cycle)
        x = np.asarray(ds[x_key].isel(cycle_number=index).values, dtype=float)
        y = np.asarray(ds[y_key].isel(cycle_number=index).values, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], label=f"{options.label_prefix} {cycle}")
    return selected


def _plot_split_cycles(
    ax: Axes,
    ds: xr.Dataset,
    keys: tuple[str, str],
    options: CyclePlotOptions,
) -> list[int]:
    """绘制一维数据中按 cycle_number 切分的曲线。"""
    x_key, y_key = keys
    cycle_map = split_by_cycle(ds)
    selected = _selected_cycles(cycle_map, options.cycles, options.max_cycles)
    for cycle in selected:
        if cycle not in cycle_map:
            continue
        cycle_ds = cycle_map[cycle]
        ax.plot(numeric_array(cycle_ds, x_key), numeric_array(cycle_ds, y_key), label=f"{options.label_prefix} {cycle}")
    return selected


__all__ = [
    "CyclePlotOptions",
    "dataset",
    "line_count",
    "plot_xy_by_cycle",
    "require_plot_column",
]
