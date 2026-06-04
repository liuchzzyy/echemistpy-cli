"""绘图层公共契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PlotSpec:
    """绘图器能力声明。"""

    kind: str
    domain: str
    input_schema: tuple[str, ...]
    required_variables: tuple[str, ...] = ()
    description: str = ""


@dataclass
class PlotResult:
    """单图绘图结果。"""

    figure: Figure
    axes: tuple[Axes, ...]
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ax(self) -> Axes:
        """返回主坐标轴。"""
        return self.axes[0]


__all__ = ["PlotResult", "PlotSpec"]
