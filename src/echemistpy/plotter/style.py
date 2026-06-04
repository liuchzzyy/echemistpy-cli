"""plotter 默认 Matplotlib 样式。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
from cycler import cycler

from echemistpy.plotter.colors import tol_cset

DEFAULT_FIGURE_SIZE: tuple[float, float] = (3.2, 2.5)
STYLE_PATH = Path(__file__).with_name("liuchzzyy.mplstyle")


@contextmanager
def plot_style() -> Iterator[None]:
    """应用项目默认绘图样式和 Paul Tol 线条配色。"""
    colors = list(tol_cset("bright"))
    with plt.style.context(STYLE_PATH), plt.rc_context({"axes.prop_cycle": cycler(color=colors)}):
        yield


def create_figure(figsize: tuple[float, float] = DEFAULT_FIGURE_SIZE):
    """创建单图 Figure 和 Axes。"""
    with plot_style():
        figure, ax = plt.subplots(figsize=figsize)
    return figure, ax


__all__ = ["DEFAULT_FIGURE_SIZE", "STYLE_PATH", "create_figure", "plot_style"]
