"""GCD 和循环性能单图绘图器。"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes

from echemistpy.analysis.echem.columns import numeric_array
from echemistpy.data.models import AnalysisBundle, DataBundle
from echemistpy.plotter.contracts import PlotSpec
from echemistpy.plotter.echem.common import CyclePlotOptions, dataset, line_count, plot_xy_by_cycle, require_plot_column
from echemistpy.plotter.registry import BasePlotter


class GCDPlotter(BasePlotter):
    """绘制恒流充放电容量-电压曲线。"""

    spec = PlotSpec(
        kind="echem-gcd",
        domain="echem",
        input_schema=("echemistpy-raw-v1", "echemistpy-analysis-v1"),
        description="容量-电压恒流充放电曲线",
    )

    def _render(self, ax: Axes, bundle: DataBundle | AnalysisBundle, **options: Any) -> dict[str, Any]:
        """绘制 GCD 曲线。"""
        ds = dataset(bundle)
        x_key = require_plot_column(ds, ("capacity_mah", "capacity_uah", "specific_capacity_mah_g", "charge_discharge_capacity_mah"), "容量")
        y_key = require_plot_column(ds, ("voltage_v", "ewe_v", "ece_v"), "电压")
        cycles = options.get("cycles")
        max_cycles = options.get("max_cycles", 8)
        plotted_cycles = plot_xy_by_cycle(ax, ds, (x_key, y_key), CyclePlotOptions(cycles=cycles, max_cycles=max_cycles))
        ax.set_xlabel(_capacity_label(x_key))
        ax.set_ylabel("Voltage / V")
        return {"kind": self.spec.kind, "x": x_key, "y": y_key, "cycles": plotted_cycles, "lines": line_count(ax)}


class CyclingCapacityPlotter(BasePlotter):
    """绘制循环充放电容量。"""

    spec = PlotSpec(
        kind="echem-cycling",
        domain="echem",
        input_schema=("echemistpy-analysis-v1",),
        required_variables=("cycle_number", "charge_capacity_mah", "discharge_capacity_mah"),
        description="循环容量图",
    )

    def _render(self, ax: Axes, bundle: DataBundle | AnalysisBundle, **_options: Any) -> dict[str, Any]:
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
        return {"kind": self.spec.kind, "x": "cycle_number", "y": ("charge_capacity_mah", "discharge_capacity_mah"), "lines": line_count(ax)}


class EfficiencyPlotter(BasePlotter):
    """绘制库伦效率。"""

    spec = PlotSpec(
        kind="echem-efficiency",
        domain="echem",
        input_schema=("echemistpy-analysis-v1",),
        required_variables=("cycle_number", "coulombic_efficiency_percent"),
        description="库伦效率图",
    )

    def _render(self, ax: Axes, bundle: DataBundle | AnalysisBundle, **_options: Any) -> dict[str, Any]:
        """绘制库伦效率曲线。"""
        ds = dataset(bundle)
        ax.plot(numeric_array(ds, "cycle_number"), numeric_array(ds, "coulombic_efficiency_percent"), marker="o", markersize=2.5)
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Coulombic efficiency / %")
        _pad_single_cycle_axis(ax, numeric_array(ds, "cycle_number"))
        return {"kind": self.spec.kind, "x": "cycle_number", "y": "coulombic_efficiency_percent", "lines": line_count(ax)}


def _capacity_label(column: str) -> str:
    """根据容量列名返回坐标轴标签。"""
    if column == "capacity_uah":
        return "Capacity / uAh"
    if column == "specific_capacity_mah_g":
        return "Specific capacity / mAh g$^{-1}$"
    return "Capacity / mAh"


def _pad_single_cycle_axis(ax: Axes, x: np.ndarray) -> None:
    """单个循环点时扩大 x 轴范围。"""
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 1:
        center = float(finite[0])
        ax.set_xlim(center - 0.5, center + 0.5)


__all__ = ["CyclingCapacityPlotter", "EfficiencyPlotter", "GCDPlotter"]
