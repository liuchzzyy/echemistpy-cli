"""EIS 单图绘图器。"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes

from echemistpy.analysis.echem.columns import numeric_array
from echemistpy.data.models import AnalysisBundle, DataBundle
from echemistpy.plotter.contracts import PlotSpec
from echemistpy.plotter.echem.common import dataset, line_count, require_plot_column
from echemistpy.plotter.registry import BasePlotter


class NyquistPlotter(BasePlotter):
    """绘制 Nyquist 图。"""

    spec = PlotSpec(
        kind="echem-nyquist",
        domain="echem",
        input_schema=("echemistpy-raw-v1", "echemistpy-analysis-v1"),
        required_variables=("re_z_ohm", "neg_im_z_ohm"),
        description="Nyquist 阻抗图",
    )

    def _render(self, ax: Axes, bundle: DataBundle | AnalysisBundle, **_options: Any) -> dict[str, Any]:
        """绘制 Nyquist 曲线。"""
        ds = dataset(bundle)
        x_key = require_plot_column(ds, ("re_z_ohm",), "阻抗实部")
        y_key = require_plot_column(ds, ("neg_im_z_ohm",), "阻抗虚部")
        ax.plot(numeric_array(ds, x_key), numeric_array(ds, y_key), marker="o", markersize=2.5)
        ax.set_xlabel("Re(Z) / Ohm")
        ax.set_ylabel("-Im(Z) / Ohm")
        ax.set_aspect("equal", adjustable="datalim")
        return {"kind": self.spec.kind, "x": x_key, "y": y_key, "lines": line_count(ax)}


class BodeMagnitudePlotter(BasePlotter):
    """绘制 Bode 模量图。"""

    spec = PlotSpec(
        kind="echem-bode-magnitude",
        domain="echem",
        input_schema=("echemistpy-raw-v1", "echemistpy-analysis-v1"),
        required_variables=("frequency_hz", "z_mag_ohm"),
        description="Bode 阻抗模量图",
    )

    def _render(self, ax: Axes, bundle: DataBundle | AnalysisBundle, **_options: Any) -> dict[str, Any]:
        """绘制频率-阻抗模量曲线。"""
        ds = dataset(bundle)
        ax.loglog(numeric_array(ds, "frequency_hz"), numeric_array(ds, "z_mag_ohm"), marker="o", markersize=2.5)
        ax.set_xlabel("Frequency / Hz")
        ax.set_ylabel("|Z| / Ohm")
        return {"kind": self.spec.kind, "x": "frequency_hz", "y": "z_mag_ohm", "lines": line_count(ax)}


class BodePhasePlotter(BasePlotter):
    """绘制 Bode 相位图。"""

    spec = PlotSpec(
        kind="echem-bode-phase",
        domain="echem",
        input_schema=("echemistpy-raw-v1", "echemistpy-analysis-v1"),
        required_variables=("frequency_hz", "phase_deg"),
        description="Bode 相位图",
    )

    def _render(self, ax: Axes, bundle: DataBundle | AnalysisBundle, **_options: Any) -> dict[str, Any]:
        """绘制频率-相位曲线。"""
        ds = dataset(bundle)
        ax.semilogx(numeric_array(ds, "frequency_hz"), numeric_array(ds, "phase_deg"), marker="o", markersize=2.5)
        ax.set_xlabel("Frequency / Hz")
        ax.set_ylabel("Phase / deg")
        return {"kind": self.spec.kind, "x": "frequency_hz", "y": "phase_deg", "lines": line_count(ax)}


__all__ = ["BodeMagnitudePlotter", "BodePhasePlotter", "NyquistPlotter"]
