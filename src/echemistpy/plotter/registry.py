"""绘图器注册表和公共入口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import xarray as xr

from echemistpy.data.models import AnalysisBundle, DataBundle
from echemistpy.plotter.contracts import PlotResult, PlotSpec
from echemistpy.plotter.style import DEFAULT_FIGURE_SIZE, create_figure, plot_style


class BasePlotter(ABC):
    """单图绘图器基类。"""

    spec = PlotSpec(kind="", domain="", input_schema=())

    def render(
        self,
        bundle: DataBundle | AnalysisBundle,
        *,
        figsize: tuple[float, float] = DEFAULT_FIGURE_SIZE,
        **options: Any,
    ) -> PlotResult:
        """校验输入并绘制单张图。"""
        self.validate(bundle)
        with plot_style():
            figure, ax = create_figure(figsize=figsize)
            metadata = self._render(ax, bundle, **options)
            extra_axes = tuple(metadata.pop("_extra_axes", ()))
            figure.tight_layout()
        return PlotResult(figure=figure, axes=(ax, *extra_axes), kind=self.spec.kind, metadata=metadata)

    def validate(self, bundle: DataBundle | AnalysisBundle) -> None:
        """检查绘图所需变量是否存在。"""
        available = _available_names(bundle)
        missing = [name for name in self.spec.required_variables if name not in available]
        if missing:
            raise ValueError(f"绘图类型 {self.spec.kind!r} 缺少变量: {missing}")

    @abstractmethod
    def _render(self, ax: Any, bundle: DataBundle | AnalysisBundle, **options: Any) -> dict[str, Any]:
        """在单个坐标轴上绘图。"""


class PlotterRegistry:
    """按 kind 管理绘图器。"""

    def __init__(self, plotters: list[BasePlotter] | None = None) -> None:
        """初始化绘图器注册表。"""
        self._plotters = {plotter.spec.kind: plotter for plotter in plotters or []}

    def register(self, plotter: BasePlotter) -> None:
        """注册绘图器。"""
        if not plotter.spec.kind:
            raise ValueError("绘图器 spec.kind 不能为空。")
        self._plotters[plotter.spec.kind] = plotter

    def get(self, kind: str) -> BasePlotter:
        """按 kind 返回绘图器。"""
        try:
            return self._plotters[kind]
        except KeyError as exc:
            available = ", ".join(self.available())
            raise KeyError(f"未注册绘图类型 {kind!r}。可用类型: {available}") from exc

    def available(self) -> list[str]:
        """返回可用绘图类型。"""
        return sorted(self._plotters)


def create_default_plotter_registry() -> PlotterRegistry:
    """创建包含内置绘图器的注册表。"""
    from echemistpy.plotter.echem import (  # noqa: PLC0415
        BodeMagnitudePlotter,
        BodePhasePlotter,
        ChronoPlotter,
        CVPlotter,
        CyclingCapacityPlotter,
        EfficiencyPlotter,
        GCDPlotter,
        NyquistPlotter,
    )

    registry = PlotterRegistry()
    for plotter in (
        CVPlotter(),
        GCDPlotter(),
        CyclingCapacityPlotter(),
        EfficiencyPlotter(),
        NyquistPlotter(),
        BodeMagnitudePlotter(),
        BodePhasePlotter(),
        ChronoPlotter(),
    ):
        registry.register(plotter)
    return registry


def plot_bundle(
    bundle: DataBundle | AnalysisBundle,
    *,
    kind: str,
    registry: PlotterRegistry | None = None,
    **options: Any,
) -> PlotResult:
    """使用默认或指定注册表绘制数据包。"""
    active_registry = registry or create_default_plotter_registry()
    return active_registry.get(kind).render(bundle, **options)


def _available_names(bundle: DataBundle | AnalysisBundle) -> set[str]:
    """返回绘图校验可用的变量和坐标名。"""
    data = bundle.data
    if isinstance(data, xr.Dataset):
        return {str(name) for name in list(data.data_vars) + list(data.coords)}
    if data.dataset is not None:
        return {str(name) for name in list(data.dataset.data_vars) + list(data.dataset.coords)}
    return set()


__all__ = [
    "BasePlotter",
    "PlotterRegistry",
    "create_default_plotter_registry",
    "plot_bundle",
]
