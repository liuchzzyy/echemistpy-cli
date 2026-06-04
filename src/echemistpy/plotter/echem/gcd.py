"""GCD 和循环性能单图绘图器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from echemistpy.analysis.echem.columns import (
    capacity_to_mah,
    current_to_ma,
    numeric_array,
    pick_column,
)
from echemistpy.analysis.echem.cycles import has_cycle_column, split_by_cycle
from echemistpy.data.models import AnalysisBundle, DataBundle
from echemistpy.plotter.contracts import PlotSpec
from echemistpy.plotter.echem.common import dataset, line_count, require_plot_column
from echemistpy.plotter.registry import BasePlotter

MASS_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(mg|g|ug|µg)", re.IGNORECASE)


class GCDPlotter(BasePlotter):
    """绘制恒流充放电容量-电压/电流曲线。"""

    spec = PlotSpec(
        kind="echem-gcd",
        domain="echem",
        input_schema=("echemistpy-raw-v1", "echemistpy-analysis-v1"),
        description="容量-电压恒流充放电曲线",
    )

    def _render(
        self, ax: Axes, bundle: DataBundle | AnalysisBundle, **options: Any
    ) -> dict[str, Any]:
        """绘制 GCD 曲线，默认只画首圈。"""
        ds = dataset(bundle)
        voltage_key = require_plot_column(ds, ("voltage_v", "ewe_v", "ece_v"), "电压")
        current_key = require_plot_column(
            ds, ("current_ma", "current_ua", "abs_current_ma"), "电流"
        )
        x_source = _capacity_axis_source(ds, bundle)
        available_cycles = _valid_gcd_cycle_numbers(ds, x_source, voltage_key)
        if not available_cycles:
            raise ValueError(
                "没有有效的 GCD 容量-电压曲线；请检查容量列或选择其他文件。"
            )
        selected_cycles = _selected_cycle_numbers(
            available_cycles, options.get("cycles"), options.get("max_cycles", 1)
        )
        display_cycles = _display_cycle_map(available_cycles)
        right_ax = ax.twinx()
        plotted_cycles: list[int] = []
        total_selected = len(selected_cycles)
        context = GCDCyclePlotContext(
            ax=ax,
            right_ax=right_ax,
            x_source=x_source,
            voltage_key=voltage_key,
            current_key=current_key,
            total_selected=total_selected,
        )

        for cycle in selected_cycles:
            cycle_ds = _cycle_dataset(ds, cycle)
            if cycle_ds is None:
                continue

            display_cycle = display_cycles.get(cycle, cycle)
            plotted = _plot_gcd_cycle(context, cycle_ds, display_cycle)
            if plotted:
                plotted_cycles.append(display_cycle)

        ax.set_xlabel(x_source.label)
        ax.set_ylabel("Voltage / V")
        right_ax.set_ylabel("Current / mA")
        _annotate_active_mass(ax, x_source)
        _merge_legends(ax, right_ax)
        return {
            "kind": self.spec.kind,
            "x": x_source.column,
            "y": voltage_key,
            "right_y": current_key,
            "capacity_unit": x_source.unit,
            "active_material_mass_g": x_source.mass_g,
            "cycles": plotted_cycles,
            "lines": line_count(ax) + line_count(right_ax),
            "_extra_axes": (right_ax,),
        }


class CyclingCapacityPlotter(BasePlotter):
    """绘制循环充放电容量。"""

    spec = PlotSpec(
        kind="echem-cycling",
        domain="echem",
        input_schema=("echemistpy-analysis-v1",),
        required_variables=(
            "cycle_number",
            "charge_capacity_mah",
            "discharge_capacity_mah",
        ),
        description="循环容量图",
    )

    def _render(
        self, ax: Axes, bundle: DataBundle | AnalysisBundle, **_options: Any
    ) -> dict[str, Any]:
        """绘制循环容量曲线。"""
        ds = dataset(bundle)
        x = numeric_array(ds, "cycle_number")
        charge = numeric_array(ds, "charge_capacity_mah")
        discharge = numeric_array(ds, "discharge_capacity_mah")
        ax.plot(x, charge, marker="o", markersize=2.5, label="Charge")
        ax.plot(x, discharge, marker="s", markersize=2.5, label="Discharge")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Capacity / mAh")
        _pad_single_cycle_axis(ax, x)
        ax.legend()
        return {
            "kind": self.spec.kind,
            "x": "cycle_number",
            "y": ("charge_capacity_mah", "discharge_capacity_mah"),
            "lines": line_count(ax),
        }


class EfficiencyPlotter(BasePlotter):
    """绘制库伦效率和充放电容量。"""

    spec = PlotSpec(
        kind="echem-efficiency",
        domain="echem",
        input_schema=("echemistpy-analysis-v1",),
        required_variables=("cycle_number", "coulombic_efficiency_percent"),
        description="库伦效率图",
    )

    def _render(
        self, ax: Axes, bundle: DataBundle | AnalysisBundle, **_options: Any
    ) -> dict[str, Any]:
        """绘制库伦效率和充放电容量组合图。"""
        ds = dataset(bundle)
        mass_g = _active_material_mass_g(bundle)
        x = _display_cycle_values(numeric_array(ds, "cycle_number"))
        ce = numeric_array(ds, "coulombic_efficiency_percent")
        right_ax = ax.twinx()

        ce_mask = np.isfinite(x) & np.isfinite(ce)
        if np.any(ce_mask):
            ax.plot(
                x[ce_mask],
                ce[ce_mask],
                marker="o",
                markersize=2.5,
                label="Coulombic efficiency",
            )
        else:
            ax.set_ylim(0.0, 105.0)

        charge = numeric_array(ds, "charge_capacity_mah")
        discharge = numeric_array(ds, "discharge_capacity_mah")
        charge_y, capacity_label = _cycle_capacity_values(charge, mass_g)
        discharge_y, _capacity_label_text = _cycle_capacity_values(discharge, mass_g)
        charge_mask = np.isfinite(x) & np.isfinite(charge_y)
        discharge_mask = np.isfinite(x) & np.isfinite(discharge_y)
        if _has_meaningful_series(charge_y[charge_mask]):
            right_ax.plot(
                x[charge_mask],
                charge_y[charge_mask],
                marker="o",
                markersize=2.5,
                label="Charge capacity",
            )
        if _has_meaningful_series(discharge_y[discharge_mask]):
            right_ax.plot(
                x[discharge_mask],
                discharge_y[discharge_mask],
                marker="s",
                markersize=2.5,
                label="Discharge capacity",
            )
        if line_count(ax) + line_count(right_ax) == 0:
            raise ValueError("没有有效的库伦效率或容量数据可用于绘图。")

        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Coulombic efficiency / %")
        right_ax.set_ylabel(capacity_label)
        _pad_single_cycle_axis(ax, x)
        _merge_legends(ax, right_ax)
        return {
            "kind": self.spec.kind,
            "x": "cycle_number",
            "y": "coulombic_efficiency_percent",
            "right_y": ("charge_capacity_mah", "discharge_capacity_mah"),
            "capacity_unit": "mAh/g" if mass_g else "mAh",
            "active_material_mass_g": mass_g,
            "lines": line_count(ax) + line_count(right_ax),
            "_extra_axes": (right_ax,),
        }


@dataclass(frozen=True)
class CapacityAxisSource:
    """GCD 容量轴配置。"""

    column: str
    label: str
    unit: str
    mass_g: float | None = None


@dataclass(frozen=True)
class GCDCyclePlotContext:
    """单个 GCD 循环绘图上下文。"""

    ax: Axes
    right_ax: Axes
    x_source: CapacityAxisSource
    voltage_key: str
    current_key: str
    total_selected: int


def _capacity_axis_source(
    ds: xr.Dataset, bundle: DataBundle | AnalysisBundle
) -> CapacityAxisSource:
    """选择 GCD x 轴容量列，优先比容量，其次 mAh。"""
    specific_column = pick_column(
        ds, ("specific_capacity_mah_g", "specific_capacity_cal_mah_g")
    )
    mass_g = _active_material_mass_g(bundle)
    if specific_column:
        return CapacityAxisSource(
            column=specific_column,
            label="Specific capacity / mAh g$^{-1}$",
            unit="mAh/g",
            mass_g=mass_g,
        )

    capacity_column = require_plot_column(
        ds, ("capacity_mah", "capacity_uah", "charge_discharge_capacity_mah"), "容量"
    )
    if mass_g:
        return CapacityAxisSource(
            column=capacity_column,
            label="Specific capacity / mAh g$^{-1}$",
            unit="mAh/g",
            mass_g=mass_g,
        )
    return CapacityAxisSource(
        column=capacity_column, label="Capacity / mAh", unit="mAh"
    )


def _capacity_values(ds: xr.Dataset, source: CapacityAxisSource) -> np.ndarray:
    """按容量轴配置返回绘图用 x 值。"""
    values = numeric_array(ds, source.column)
    if source.unit == "mAh/g":
        if source.column in {"specific_capacity_mah_g", "specific_capacity_cal_mah_g"}:
            return values
        if source.mass_g:
            return capacity_to_mah(values, source.column) / source.mass_g
    return capacity_to_mah(values, source.column)


def _cycle_capacity_values(
    values_mah: np.ndarray, mass_g: float | None
) -> tuple[np.ndarray, str]:
    """返回循环容量曲线的 y 值和轴标签。"""
    values = np.asarray(values_mah, dtype=float)
    if mass_g:
        return values / mass_g, "Specific capacity / mAh g$^{-1}$"
    return values, "Capacity / mAh"


def _plot_gcd_cycle(
    context: GCDCyclePlotContext,
    cycle_ds: xr.Dataset,
    display_cycle: int,
) -> bool:
    """绘制单个 GCD 循环，返回是否成功绘制。"""
    x_values = _capacity_values(cycle_ds, context.x_source)
    voltage = numeric_array(cycle_ds, context.voltage_key)
    current = current_to_ma(
        numeric_array(cycle_ds, context.current_key), context.current_key
    )
    voltage_mask = np.isfinite(x_values) & np.isfinite(voltage)
    if not np.any(voltage_mask):
        return False

    (voltage_line,) = context.ax.plot(
        x_values[voltage_mask],
        voltage[voltage_mask],
        label=_curve_label("Voltage", display_cycle, context.total_selected),
    )
    current_mask = np.isfinite(x_values) & np.isfinite(current)
    if np.any(current_mask):
        context.right_ax.plot(
            x_values[current_mask],
            current[current_mask],
            linestyle="--",
            linewidth=0.9,
            color=voltage_line.get_color(),
            alpha=0.75,
            label=_curve_label("Current", display_cycle, context.total_selected),
        )
    return True


def _selected_cycle_numbers(
    available: list[int], cycles: Any, max_cycles: int | None
) -> list[int]:
    """返回要绘制的内部循环编号；传入 cycles 按显示圈数解释。"""
    if not available:
        return []
    if cycles is None:
        limit = 1 if max_cycles is None else max_cycles
        return available[:limit]
    display_to_raw = {
        display: raw for raw, display in _display_cycle_map(available).items()
    }
    requested = (
        [cycles] if isinstance(cycles, int) else [int(cycle) for cycle in cycles]
    )
    selected = []
    for cycle in requested:
        raw_cycle = display_to_raw.get(cycle, cycle)
        if raw_cycle in available:
            selected.append(raw_cycle)
    return selected


def _available_cycle_numbers(ds: xr.Dataset) -> list[int]:
    """返回 Dataset 中存在的循环编号。"""
    if _is_cycle_matrix(ds):
        return [int(cycle) for cycle in np.asarray(ds["cycle_number"].values)]
    if has_cycle_column(ds):
        return sorted(int(cycle) for cycle in split_by_cycle(ds))
    return [1]


def _valid_gcd_cycle_numbers(
    ds: xr.Dataset, x_source: CapacityAxisSource, voltage_key: str
) -> list[int]:
    """返回容量和电压数据有效的 GCD 循环编号。"""
    valid_cycles = []
    for cycle in _available_cycle_numbers(ds):
        cycle_ds = _cycle_dataset(ds, cycle)
        if cycle_ds is not None and _has_gcd_curve(cycle_ds, x_source, voltage_key):
            valid_cycles.append(cycle)
    return valid_cycles


def _has_gcd_curve(
    ds: xr.Dataset, x_source: CapacityAxisSource, voltage_key: str
) -> bool:
    """判断单个循环是否有可绘制的容量-电压曲线。"""
    x_values = _capacity_values(ds, x_source)
    voltage = numeric_array(ds, voltage_key)
    mask = np.isfinite(x_values) & np.isfinite(voltage)
    if np.count_nonzero(mask) < 2:
        return False
    return float(np.nanmax(x_values[mask]) - np.nanmin(x_values[mask])) > 1e-12


def _has_meaningful_series(values: np.ndarray) -> bool:
    """判断序列是否包含可绘制的有效数值。"""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size and np.nanmax(np.abs(finite)) > 1e-12)


def _display_cycle_map(available_cycles: list[int]) -> dict[int, int]:
    """将内部循环编号映射为面向用户的一起始圈数。"""
    if available_cycles and min(available_cycles) == 0:
        return {cycle: cycle + 1 for cycle in available_cycles}
    return {cycle: cycle for cycle in available_cycles}


def _display_cycle_values(values: np.ndarray) -> np.ndarray:
    """将循环坐标转换为显示用的一起始编号。"""
    cycles = np.asarray(values, dtype=float)
    finite = cycles[np.isfinite(cycles)]
    if finite.size and np.nanmin(finite) == 0:
        return cycles + 1
    return cycles


def _curve_label(signal: str, display_cycle: int, total_selected: int) -> str:
    """返回图例标签。"""
    if total_selected == 1:
        return signal
    return f"{signal} cycle {display_cycle}"


def _cycle_dataset(ds: xr.Dataset, cycle: int) -> xr.Dataset | None:
    """返回指定循环的数据；无循环列时将整体视为第 1 圈。"""
    if _is_cycle_matrix(ds):
        available = [int(value) for value in np.asarray(ds["cycle_number"].values)]
        if cycle not in available:
            return None
        return ds.isel(cycle_number=available.index(cycle))
    if has_cycle_column(ds):
        return split_by_cycle(ds).get(cycle)
    return ds if cycle == 1 else None


def _is_cycle_matrix(ds: xr.Dataset) -> bool:
    """判断是否为分析结果中的 cycle_number-record 矩阵。"""
    return "cycle_number" in ds.dims and "record" in ds.dims


def _active_material_mass_g(bundle: DataBundle | AnalysisBundle) -> float | None:
    """从元数据中解析活性物质质量，单位为 g。"""
    raw_metadata = bundle.meta.raw_metadata
    for key in ("active_material_mass_g", "mass_g"):
        value = raw_metadata.get(key)
        if value is not None:
            try:
                mass = float(value)
            except (TypeError, ValueError):
                continue
            if mass > 0:
                return mass

    for value in (
        bundle.meta.active_material_mass,
        raw_metadata.get("active_material_mass"),
    ):
        mass = _parse_mass_g(value)
        if mass:
            return mass
    return None


def _parse_mass_g(value: Any) -> float | None:
    """从带单位文本中解析质量，单位为 g。"""
    if value is None:
        return None
    match = MASS_PATTERN.search(str(value))
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "g":
        return amount
    if unit == "mg":
        return amount / 1000.0
    return amount / 1_000_000.0


def _annotate_active_mass(ax: Axes, source: CapacityAxisSource) -> None:
    """在比容量图中标注活性物质质量。"""
    if source.unit != "mAh/g" or source.mass_g is None:
        return
    ax.text(
        0.03,
        0.04,
        f"Active material: {source.mass_g * 1000:g} mg",
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="bottom",
    )


def _merge_legends(ax: Axes, right_ax: Axes) -> None:
    """合并左右 y 轴图例。"""
    handles, labels = ax.get_legend_handles_labels()
    right_handles, right_labels = right_ax.get_legend_handles_labels()
    all_handles = handles + right_handles
    all_labels = labels + right_labels
    if all_handles:
        ax.legend(all_handles, all_labels)


def _pad_single_cycle_axis(ax: Axes, x: np.ndarray) -> None:
    """单个循环点时扩大 x 轴范围。"""
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 1:
        center = float(finite[0])
        ax.set_xlim(center - 0.5, center + 0.5)


__all__ = ["CyclingCapacityPlotter", "EfficiencyPlotter", "GCDPlotter"]
