"""XAS 元素数据库。

存储不同元素的 XAS 分析标准参数，包括 pre-edge、归一化范围和 EXAFS 默认参数。
"""

from __future__ import annotations

from typing import Any, TypedDict


class XASParameters(TypedDict):
    e0: float
    pre_edge_range: tuple[float, float]
    norm_range: tuple[float, float]
    kmin: float
    kmax: float
    rbkg: float


ELEMENT_DB: dict[str, XASParameters] = {
    "Mn": {
        "e0": 6539.0,
        "pre_edge_range": (-200, -30),
        "norm_range": (150, 800),
        "kmin": 3.0,
        "kmax": 12.0,
        "rbkg": 1.0,
    },
    "Fe": {
        "e0": 7112.0,
        "pre_edge_range": (-200, -30),
        "norm_range": (150, 800),
        "kmin": 3.0,
        "kmax": 12.0,
        "rbkg": 1.0,
    },
    "Zn": {
        "e0": 9659.0,
        "pre_edge_range": (-200, -30),
        "norm_range": (150, 800),
        "kmin": 3.0,
        "kmax": 12.0,
        "rbkg": 1.0,
    },
}


def get_element_config(element: str) -> dict[str, Any]:
    """返回指定元素的 XAS 分析配置。"""
    params = ELEMENT_DB.get(element)
    if not params:
        return {}

    return {
        "theoretical_e0": params["e0"],
        "normalize": {
            "pre1": params["pre_edge_range"][0],
            "pre2": params["pre_edge_range"][1],
            "norm1": params["norm_range"][0],
            "norm2": params["norm_range"][1],
        },
        "autobk": {
            "rbkg": params["rbkg"],
            "kweight": 2,
        },
        "fft": {
            "kmin": params["kmin"],
            "kmax": params["kmax"],
            "window": "hanning",
        },
    }
