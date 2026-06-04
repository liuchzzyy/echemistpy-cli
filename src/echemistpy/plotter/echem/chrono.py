"""CA/CP 时间序列单图绘图器。"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes

from echemistpy.data.models import AnalysisBundle, DataBundle
from echemistpy.plotter.contracts import PlotSpec
from echemistpy.plotter.echem.common import CyclePlotOptions, dataset, line_count, plot_xy_by_cycle, require_plot_column
from echemistpy.plotter.registry import BasePlotter


class ChronoPlotter(BasePlotter):
    """绘制时间-电流或时间-电压曲线。"""

    spec = PlotSpec(
        kind="echem-chrono",
        domain="echem",
        input_schema=("echemistpy-raw-v1", "echemistpy-analysis-v1"),
        description="计时电流或计时电位图",
    )

    def _render(self, ax: Axes, bundle: DataBundle | AnalysisBundle, **options: Any) -> dict[str, Any]:
        """绘制时间序列曲线。"""
        ds = dataset(bundle)
        x_key = require_plot_column(ds, ("time_s", "test_time_s", "step_time_s"), "时间")
        y_candidates = ("current_ma", "current_ua") if options.get("variable", "current") == "current" else ("voltage_v", "ewe_v", "ece_v")
        y_key = require_plot_column(ds, y_candidates, "信号")
        cycles = options.get("cycles")
        max_cycles = options.get("max_cycles", 8)
        plotted_cycles = plot_xy_by_cycle(ax, ds, (x_key, y_key), CyclePlotOptions(cycles=cycles, max_cycles=max_cycles))
        ax.set_xlabel("Time / s")
        ax.set_ylabel(_signal_label(y_key))
        return {"kind": self.spec.kind, "x": x_key, "y": y_key, "cycles": plotted_cycles, "lines": line_count(ax)}


def _signal_label(column: str) -> str:
    """根据信号列返回坐标轴标签。"""
    if column == "current_ma":
        return "Current / mA"
    if column == "current_ua":
        return "Current / uA"
    return "Voltage / V"


__all__ = ["ChronoPlotter"]
