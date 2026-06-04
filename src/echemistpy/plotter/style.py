"""plotter 默认 Matplotlib 样式。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib.pyplot as plt
from cycler import cycler

from echemistpy.plotter.colors import tol_cset

DEFAULT_FIGURE_SIZE: tuple[float, float] = (3.2, 2.5)
DEFAULT_STYLE = {
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "font.family": "sans-serif",
    "font.size": 8.0,
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.tt": "Arial",
    "mathtext.cal": "Arial",
    "mathtext.default": "regular",
    "axes.linewidth": 0.8,
    "axes.titlesize": 8.0,
    "axes.labelsize": 9.0,
    "xtick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.6,
    "xtick.labelsize": 8.0,
    "xtick.direction": "out",
    "xtick.top": False,
    "ytick.major.size": 3.0,
    "ytick.major.width": 0.8,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.6,
    "ytick.labelsize": 8.0,
    "ytick.direction": "out",
    "ytick.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.25,
    "legend.borderpad": 0.2,
    "legend.columnspacing": 0.8,
    "figure.subplot.left": 0.20,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.20,
    "figure.subplot.top": 0.92,
    "figure.subplot.wspace": 0.20,
    "figure.subplot.hspace": 0.20,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.transparent": False,
    "savefig.pad_inches": 0.05,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


@contextmanager
def plot_style() -> Iterator[None]:
    """应用项目默认绘图样式和 Paul Tol 线条配色。"""
    colors = list(tol_cset("bright"))
    with plt.rc_context({**DEFAULT_STYLE, "axes.prop_cycle": cycler(color=colors)}):
        yield


def create_figure(figsize: tuple[float, float] = DEFAULT_FIGURE_SIZE):
    """创建单图 Figure 和 Axes。"""
    with plot_style():
        figure, ax = plt.subplots(figsize=figsize)
    return figure, ax


__all__ = ["DEFAULT_FIGURE_SIZE", "DEFAULT_STYLE", "create_figure", "plot_style"]
