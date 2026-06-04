"""echemistpy 绘图 API。"""

from echemistpy.plotter.colors import tol_cmap, tol_cset
from echemistpy.plotter.contracts import PlotResult, PlotSpec
from echemistpy.plotter.output import save_plot_result, timestamped_log_dir
from echemistpy.plotter.registry import BasePlotter, PlotterRegistry, create_default_plotter_registry, plot_bundle
from echemistpy.plotter.style import DEFAULT_FIGURE_SIZE
from echemistpy.plotter.xas import plot_echem_xas

__all__ = [
    "DEFAULT_FIGURE_SIZE",
    "BasePlotter",
    "PlotResult",
    "PlotSpec",
    "PlotterRegistry",
    "create_default_plotter_registry",
    "plot_bundle",
    "plot_echem_xas",
    "save_plot_result",
    "timestamped_log_dir",
    "tol_cmap",
    "tol_cset",
]
