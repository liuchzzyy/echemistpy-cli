from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from echemistpy.analysis.registry import create_default_registry
from echemistpy.analysis.txm import STXMAnalyzer, TXMAnalyzer
from echemistpy.analysis.xas import XASAnalyzer
from echemistpy.data.models import DataBundle, Metadata
from echemistpy.plotter import plot_echem_xas


def test_txm_analyzer_accepts_raw_energy_coordinate() -> None:
    energy = np.array([700.0, 701.0, 702.0, 703.0])
    data = np.stack(
        [
            np.full((3, 2), 0.10),
            np.full((3, 2), 0.15),
            np.full((3, 2), 0.21),
            np.full((3, 2), 0.30),
        ]
    )
    ds = xr.Dataset(
        {"optical_density": (("energy", "y", "x"), data)},
        coords={"energy": energy, "y": [0.0, 1.0, 2.0], "x": [0.0, 1.0]},
    )
    bundle = DataBundle(data=ds, meta=Metadata(technique=["txm"], instrument="mistral"))

    result = TXMAnalyzer(align_images=False, energy_step=1.0, pca_components=2, n_clusters=2).analyze(bundle)

    assert "energy_ev" in result.data.coords
    assert "denoised" in result.data
    assert "background_removed" in result.data
    assert result.provenance["analyzer"] == "TXMAnalyzer"


def test_txm_registry_uses_new_name_and_keeps_alias() -> None:
    registry = create_default_registry()
    analyzer = registry.get_analyzer("txm")

    assert isinstance(analyzer, TXMAnalyzer)
    assert STXMAnalyzer is TXMAnalyzer


def test_xas_analyzer_accepts_energyc_alias_and_e0_param() -> None:
    energy = np.linspace(7100.0, 7120.0, 25)
    mu = np.vstack([np.linspace(0.1, 1.0, 25), np.linspace(0.2, 1.2, 25)])
    ds = xr.Dataset(
        {"absorption": (("record", "energyc"), mu)},
        coords={"record": [0, 1], "energyc": energy},
    )
    bundle = DataBundle(data=ds, meta=Metadata(technique=["xas"]))

    result = XASAnalyzer(normalize_params={"e0": 7110.0}).analyze(bundle)

    assert "energy_ev" in result.data.coords
    assert "absorption" in result.data
    assert "e0_ev" in result.data


def test_plot_echem_xas_from_plotter() -> None:
    times = pd.date_range("2025-01-01 00:00:00", periods=4, freq="10s")
    echem = xr.Dataset(
        {
            "ewe_v": ("record", [3.0, 3.1, 3.2, 3.3]),
            "current_ma": ("record", [0.1, 0.1, -0.1, -0.1]),
        },
        coords={"record": np.arange(4), "systime": ("record", times)},
    )
    xas = xr.Dataset(
        {"absorption": (("record", "energy_ev"), np.ones((2, 3)))},
        coords={
            "record": [0, 1],
            "energy_ev": [1.0, 2.0, 3.0],
            "systime": ("record", times[[1, 3]]),
            "file_name": ("record", ["scan_a", "scan_b"]),
        },
    )

    fig = plot_echem_xas(echem, xas)

    assert isinstance(fig, Figure)
    plt.close(fig)
