"""绘图使用的 Paul Tol 色盲友好配色。"""

from __future__ import annotations

import warnings
from collections import namedtuple
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array

# 配色来源：Paul Tol, SRON, https://personal.sron.nl/~pault/ 。
# 原始配色许可为标准 BSD 3-Clause License。

QUALITATIVE_COLORSETS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "bright": (
        "BrightColorSet",
        ("blue", "red", "green", "yellow", "cyan", "purple", "grey", "black"),
        ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"),
    ),
    "high-contrast": (
        "HighContrastColorSet",
        ("blue", "yellow", "red", "black"),
        ("#004488", "#DDAA33", "#BB5566", "#000000"),
    ),
    "vibrant": (
        "VibrantColorSet",
        ("orange", "blue", "cyan", "magenta", "red", "teal", "grey", "black"),
        ("#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988", "#BBBBBB", "#000000"),
    ),
    "muted": (
        "MutedColorSet",
        ("rose", "indigo", "sand", "green", "cyan", "wine", "teal", "olive", "purple", "pale_grey", "black"),
        ("#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE", "#882255", "#44AA99", "#999933", "#AA4499", "#DDDDDD", "#000000"),
    ),
    "medium-contrast": (
        "MediumContrastColorSet",
        ("light_blue", "dark_blue", "light_yellow", "dark_red", "dark_yellow", "light_red", "black"),
        ("#6699CC", "#004488", "#EECC66", "#994455", "#997700", "#EE99AA", "#000000"),
    ),
    "light": (
        "LightColorSet",
        ("light_blue", "orange", "light_yellow", "pink", "light_cyan", "mint", "pear", "olive", "pale_grey", "black"),
        ("#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00", "#DDDDDD", "#000000"),
    ),
}

CONTINUOUS_COLORMAPS: dict[str, tuple[str, ...]] = {
    "sunset": ("#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1", "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B", "#DD3D2D", "#A50026"),
    "nightfall": ("#125A56", "#00767B", "#238F9D", "#42A7C6", "#60BCE9", "#9DCCEF", "#C6DBED", "#DEE6E7", "#ECEADA", "#F0E6B2", "#F9D576", "#FFB954", "#FD9A44", "#F57634", "#E94C1F", "#D11807", "#A01813"),
    "BuRd": ("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"),
    "PRGn": ("#762A83", "#9970AB", "#C2A5CF", "#E7D4E8", "#F7F7F7", "#D9F0D3", "#ACD39E", "#5AAE61", "#1B7837"),
    "YlOrBr": ("#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"),
    "WhOrBr": ("#FFFFFF", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"),
    "iridescent": ("#FEFBE9", "#FCF7D5", "#F5F3C1", "#EAF0B5", "#DDECBF", "#D0E7CA", "#C2E3D2", "#B5DDD8", "#A8D8DC", "#9BD2E1", "#8DCBE4", "#81C4E7", "#7BBCE7", "#7EB2E4", "#88A5DD", "#9398D2", "#9B8AC4", "#9D7DB2", "#9A709E", "#906388", "#805770", "#684957", "#46353A"),
    "rainbow_PuRd": ("#6F4C9B", "#6059A9", "#5568B8", "#4E79C5", "#4D8AC6", "#4E96BC", "#549EB3", "#59A5A9", "#60AB9E", "#69B190", "#77B77D", "#8CBC68", "#A6BE54", "#BEBC48", "#D1B541", "#DDAA3C", "#E49C39", "#E78C35", "#E67932", "#E4632D", "#DF4828", "#DA2222"),
    "rainbow_PuBr": ("#6F4C9B", "#6059A9", "#5568B8", "#4E79C5", "#4D8AC6", "#4E96BC", "#549EB3", "#59A5A9", "#60AB9E", "#69B190", "#77B77D", "#8CBC68", "#A6BE54", "#BEBC48", "#D1B541", "#DDAA3C", "#E49C39", "#E78C35", "#E67932", "#E4632D", "#DF4828", "#DA2222", "#B8221E", "#95211B", "#721E17", "#521A13"),
    "rainbow_WhRd": ("#E8ECFB", "#DDD8EF", "#D1C1E1", "#C3A8D1", "#B58FC2", "#A778B4", "#9B62A7", "#8C4E99", "#6F4C9B", "#6059A9", "#5568B8", "#4E79C5", "#4D8AC6", "#4E96BC", "#549EB3", "#59A5A9", "#60AB9E", "#69B190", "#77B77D", "#8CBC68", "#A6BE54", "#BEBC48", "#D1B541", "#DDAA3C", "#E49C39", "#E78C35", "#E67932", "#E4632D", "#DF4828", "#DA2222"),
    "rainbow_WhBr": ("#E8ECFB", "#DDD8EF", "#D1C1E1", "#C3A8D1", "#B58FC2", "#A778B4", "#9B62A7", "#8C4E99", "#6F4C9B", "#6059A9", "#5568B8", "#4E79C5", "#4D8AC6", "#4E96BC", "#549EB3", "#59A5A9", "#60AB9E", "#69B190", "#77B77D", "#8CBC68", "#A6BE54", "#BEBC48", "#D1B541", "#DDAA3C", "#E49C39", "#E78C35", "#E67932", "#E4632D", "#DF4828", "#DA2222", "#B8221E", "#95211B", "#721E17", "#521A13"),
}
DISCRETE_COLORMAPS = {f"{name}_discrete": colors for name, colors in CONTINUOUS_COLORMAPS.items() if name in {"sunset", "nightfall", "BuRd", "PRGn", "YlOrBr"}}
COLORMAPS = {**CONTINUOUS_COLORMAPS, **DISCRETE_COLORMAPS}


def discretemap(colormap: str, hexclrs: tuple[str, ...] | list[str]) -> LinearSegmentedColormap:
    """根据离散颜色列表创建无插值色图。"""
    colors = to_rgba_array(hexclrs)
    colors = np.vstack([colors[0], colors, colors[-1]])
    color_dict: dict[Literal["red", "green", "blue", "alpha"], Sequence[tuple[float, ...]]] = {}
    keys: tuple[Literal["red"], Literal["green"], Literal["blue"]] = ("red", "green", "blue")
    for index, key in enumerate(keys):
        color_dict[key] = [(i / (len(colors) - 2.0), float(colors[i, index]), float(colors[i + 1, index])) for i in range(len(colors) - 1)]
    return LinearSegmentedColormap(colormap, color_dict)


def tol_cmap(colormap: str | None = None, lut: int | None = None) -> LinearSegmentedColormap | tuple[str, ...]:
    """返回 Paul Tol 连续或离散色图。"""
    if colormap is None:
        return (*tuple(COLORMAPS), "rainbow_discrete")
    if colormap == "rainbow_discrete":
        return _rainbow_discrete(lut)
    if colormap not in COLORMAPS:
        warnings.warn(f"未知色图 {colormap!r}，已改用 'rainbow_PuRd'。", stacklevel=2)
        colormap = "rainbow_PuRd"
    if colormap.endswith("_discrete"):
        return discretemap(colormap, COLORMAPS[colormap])
    cmap = LinearSegmentedColormap.from_list(colormap, COLORMAPS[colormap])
    cmap.set_bad("#FFFFFF")
    return cmap


def tol_cset(colorset: str | None = None) -> Any:
    """返回 Paul Tol 定性配色集合。"""
    if colorset is None:
        return tuple(QUALITATIVE_COLORSETS)
    if colorset not in QUALITATIVE_COLORSETS:
        warnings.warn(f"未知配色集合 {colorset!r}，已改用 'bright'。", stacklevel=2)
        colorset = "bright"
    type_name, fields, colors = QUALITATIVE_COLORSETS[colorset]
    color_type = namedtuple(type_name, fields)
    return color_type(*colors)


def _rainbow_discrete(lut: int | None = None) -> LinearSegmentedColormap:
    """返回 Paul Tol rainbow 离散色图。"""
    colors = (
        "#E8ECFB",
        "#D9CCE3",
        "#D1BBD7",
        "#CAACCB",
        "#BA8DB4",
        "#AE76A3",
        "#AA6F9E",
        "#994F88",
        "#882E72",
        "#1965B0",
        "#437DBF",
        "#5289C7",
        "#6195CF",
        "#7BAFDE",
        "#4EB265",
        "#90C987",
        "#CAE0AB",
        "#F7F056",
        "#F7CB45",
        "#F6C141",
        "#F4A736",
        "#F1932D",
        "#EE8026",
        "#E8601C",
        "#E65518",
        "#DC050C",
        "#A5170E",
        "#72190E",
        "#42150A",
    )
    indexes = (
        (9,),
        (9, 25),
        (9, 17, 25),
        (9, 14, 17, 25),
        (9, 13, 14, 17, 25),
        (9, 13, 14, 16, 17, 25),
        (8, 9, 13, 14, 16, 17, 25),
        (8, 9, 13, 14, 16, 17, 22, 25),
        (8, 9, 13, 14, 16, 17, 22, 25, 27),
        (8, 9, 13, 14, 16, 17, 20, 23, 25, 27),
        (8, 9, 11, 13, 14, 16, 17, 20, 23, 25, 27),
        (2, 5, 8, 9, 11, 13, 14, 16, 17, 20, 23, 25),
        (2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 20, 23, 25),
        (2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25),
        (2, 5, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27),
        (2, 4, 6, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27),
        (2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 27),
        (2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 26, 27),
        (1, 3, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 21, 23, 25, 26, 27),
        (1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 21, 23, 25, 26, 27),
        (1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 25, 26, 27),
        (1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 25, 26, 27, 28),
        (0, 1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 25, 26, 27, 28),
    )
    level = 22 if lut is None or lut < 1 or lut > len(indexes) else lut
    cmap = discretemap("rainbow_discrete", [colors[index] for index in indexes[level - 1]])
    cmap.set_bad("#777777" if level == len(indexes) else "#FFFFFF")
    return cmap


__all__ = [
    "COLORMAPS",
    "QUALITATIVE_COLORSETS",
    "discretemap",
    "tol_cmap",
    "tol_cset",
]
