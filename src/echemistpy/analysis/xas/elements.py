"""XAS Element Database.

Stores standard parameters for XAS analysis (pre-edge, normalization ranges)
for different elements.
"""

from __future__ import annotations
from typing import Any, TypedDict


class XASParameters(TypedDict):
    e0: float  # Approximate edge energy
    pre_edge_range: tuple[float, float]  # Relative to E0
    norm_range: tuple[float, float]  # Relative to E0
    kmin: float
    kmax: float
    rbkg: float


# Dictionary of element parameters
# Ranges are typically relative to E0: [min, max]
# pre-edge: usually [-150, -30] or similar
# norm: usually [50, end] or similar
ELEMENT_DB: dict[str, XASParameters] = {
    "Mn": {
        "e0": 6539.0,
        "pre_edge_range": (-200, -30),
        "norm_range": (150, 800),  # Depends on scan length
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
    """Get XAS analysis configuration for a specific element."""
    params = ELEMENT_DB.get(element)
    if not params:
        return {}

    return {
        "theoretical_e0": params["e0"],  # Add explicit theoretical E0
        "normalize": {
            # We don't hardcode e0 here, letting larch auto-find it,
            # but we can provide ranges if larch supports them.
            # Larch pre_edge takes: pre1, pre2, norm1, norm2 (relative or absolute?)
            # Usually absolute.
            # So we might need to know E0 to set them, or just rely on defaults.
            # However, providing hints is good.
            # Let's return standard config structure.
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
