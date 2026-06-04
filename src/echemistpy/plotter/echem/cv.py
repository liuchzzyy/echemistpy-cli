"""CV 单图绘图器。"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes

from echemistpy.data.models import AnalysisBundle, DataBundle
from echemistpy.plotter.contracts import PlotSpec
from echemistpy.plotter.echem.common import CyclePlotOptions, dataset, line_count, plot_xy_by_cycle, require_plot_column
from echemistpy.plotter.registry import BasePlotter


class CVPlotter(BasePlotter):
    """绘制循环伏安曲线。"""

    spec = PlotSpec(
        kind="echem-cv",
        domain="echem",
        input_schema=("echemistpy-raw-v1", "echemistpy-analysis-v1"),
        description="电流-电位循环伏安曲线",
    )

    def _render(self, ax: Axes, bundle: DataBundle | AnalysisBundle, **options: Any) -> dict[str, Any]:
        """绘制 CV 曲线。"""
        ds = dataset(bundle)
        x_key = require_plot_column(ds, ("ewe_v", "voltage_v"), "电位")
        y_key = require_plot_column(ds, ("current_ma", "current_ua"), "电流")
        cycles = options.get("cycles")
        max_cycles = options.get("max_cycles", 8)
        plotted_cycles = plot_xy_by_cycle(ax, ds, (x_key, y_key), CyclePlotOptions(cycles=cycles, max_cycles=max_cycles))
        ax.set_xlabel("Potential / V")
        ax.set_ylabel("Current / mA" if y_key == "current_ma" else "Current / uA")
        return {"kind": self.spec.kind, "x": x_key, "y": y_key, "cycles": plotted_cycles, "lines": line_count(ax)}


__all__ = ["CVPlotter"]
