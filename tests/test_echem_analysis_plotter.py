from pathlib import Path

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from echemistpy.analysis.echem import GCDAnalyzer
from echemistpy.data import DataBundle, Metadata
from echemistpy.io import load
from echemistpy.plotter import DEFAULT_FIGURE_SIZE, plot_bundle, save_plot_result, timestamped_log_dir

CV_SAMPLE = Path("Samples/Echem/Biologic/Trial01_CV/LC-Zn-1M Zn-alphaMnO2-CV_01mVs_02_CV_C02.mpr")
GCD_SAMPLE = Path("Samples/Echem/Biologic/Trial02_GCD/ACETATE BUFFER CYCING_C01.mpr")
EIS_SAMPLE = Path("Samples/Echem/Biologic/Trial03_EIS/EMD-2V-2mAh-1M+02M-40mL_02_PEIS_C01.mpr")


def test_gcd_analyzer_returns_cycle_analysis_bundle() -> None:
    bundle = load(GCD_SAMPLE, instrument="biologic")

    result = GCDAnalyzer().analyze(bundle)

    assert result.data.sizes["cycle_number"] >= 1
    assert {"capacity_mah", "voltage_v", "charge_capacity_mah", "discharge_capacity_mah", "coulombic_efficiency_percent"}.issubset(result.data.data_vars)
    assert result.parameters["used_columns"]["current_column"] == "current_ma"


def test_gcd_analyzer_converts_capacity_uah_to_mah() -> None:
    ds = xr.Dataset(
        {
            "cycle_number": (("record",), [1, 1, 1]),
            "current_ua": (("record",), [10.0, 10.0, -10.0]),
            "capacity_uah": (("record",), [0.0, 1000.0, 500.0]),
            "voltage_v": (("record",), [1.0, 1.1, 1.0]),
        },
        coords={"record": [1, 2, 3], "time_s": (("record",), [0.0, 1.0, 2.0])},
    )
    bundle = DataBundle(data=ds, meta=Metadata(technique=["gcd"]))

    result = GCDAnalyzer().analyze(bundle)

    assert float(result.data["capacity_mah"].max()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("sample", "kind", "expected_xlabel", "expected_ylabel"),
    [
        (CV_SAMPLE, "echem-cv", "Potential / V", "Current / mA"),
        (EIS_SAMPLE, "echem-nyquist", "Re(Z) / Ohm", "-Im(Z) / Ohm"),
        (EIS_SAMPLE, "echem-bode-magnitude", "Frequency / Hz", "|Z| / Ohm"),
        (EIS_SAMPLE, "echem-bode-phase", "Frequency / Hz", "Phase / deg"),
    ],
)
def test_echem_raw_plotters_draw_single_default_sized_figure(sample: Path, kind: str, expected_xlabel: str, expected_ylabel: str) -> None:
    bundle = load(sample, instrument="biologic")

    result = plot_bundle(bundle, kind=kind, max_cycles=2)

    try:
        assert result.figure.get_size_inches().tolist() == list(DEFAULT_FIGURE_SIZE)
        assert len(result.axes) == 1
        assert result.ax.get_xlabel() == expected_xlabel
        assert result.ax.get_ylabel() == expected_ylabel
        assert result.metadata["lines"] >= 1
    finally:
        plt.close(result.figure)


def test_gcd_plotter_defaults_to_first_cycle_with_current_axis() -> None:
    bundle = _two_cycle_gcd_bundle()

    result = plot_bundle(bundle, kind="echem-gcd")

    try:
        _assert_default_sized_figure(result)
        _assert_gcd_dual_axis_labels(result, "Capacity / mAh")
        assert result.metadata["lines"] == 2
        assert result.metadata["cycles"] == [1]
    finally:
        plt.close(result.figure)


def test_gcd_plotter_can_select_cycle_and_use_specific_capacity_from_mass() -> None:
    bundle = _two_cycle_gcd_bundle(active_material_mass="2 mg")

    result = plot_bundle(bundle, kind="echem-gcd", cycles=[2])

    try:
        _assert_gcd_dual_axis_labels(result, "Specific capacity / mAh g$^{-1}$")
        assert result.metadata["capacity_unit"] == "mAh/g"
        assert result.metadata["cycles"] == [2]
        assert result.metadata["active_material_mass_g"] == pytest.approx(0.002)
        assert np.asarray(result.ax.lines[0].get_xdata()).max() == pytest.approx(250.0)
    finally:
        plt.close(result.figure)


def test_gcd_plotter_skips_zero_capacity_initial_cycle_by_default() -> None:
    ds = xr.Dataset(
        {
            "cycle_number": (("record",), [0, 0, 1, 1]),
            "capacity_mah": (("record",), [0.0, 0.0, 0.0, 1.0]),
            "voltage_v": (("record",), [1.4, 1.39, 1.3, 0.9]),
            "current_ma": (("record",), [0.0, 0.0, -1.0, -1.0]),
        },
        coords={"record": [1, 2, 3, 4]},
    )
    bundle = DataBundle(data=ds, meta=Metadata(technique=["gcd"]))

    result = plot_bundle(bundle, kind="echem-gcd")

    try:
        assert result.metadata["cycles"] == [1]
        assert np.asarray(result.ax.lines[0].get_xdata()).max() == pytest.approx(1.0)
    finally:
        plt.close(result.figure)


def test_gcd_plotter_rejects_data_without_capacity_window() -> None:
    ds = xr.Dataset(
        {
            "cycle_number": (("record",), [0, 0]),
            "capacity_mah": (("record",), [0.0, 0.0]),
            "voltage_v": (("record",), [1.4, 1.39]),
            "current_ma": (("record",), [0.0, 0.0]),
        },
        coords={"record": [1, 2]},
    )
    bundle = DataBundle(data=ds, meta=Metadata(technique=["gcd"]))

    with pytest.raises(ValueError, match="没有有效的 GCD"):
        plot_bundle(bundle, kind="echem-gcd")


def test_gcd_plotter_uses_existing_specific_capacity_column() -> None:
    ds = xr.Dataset(
        {
            "cycle_number": (("record",), [1, 1]),
            "specific_capacity_mah_g": (("record",), [0.0, 120.0]),
            "voltage_v": (("record",), [1.3, 0.9]),
            "current_ma": (("record",), [-1.0, -1.0]),
        },
        coords={"record": [1, 2]},
    )
    bundle = DataBundle(data=ds, meta=Metadata(technique=["gcd"], active_material_mass="1 mg"))

    result = plot_bundle(bundle, kind="echem-gcd")

    try:
        assert result.ax.get_xlabel() == "Specific capacity / mAh g$^{-1}$"
        assert np.asarray(result.ax.lines[0].get_xdata()).max() == pytest.approx(120.0)
    finally:
        plt.close(result.figure)


def test_echem_cycling_plotter_draws_single_default_sized_figure() -> None:
    analysis = GCDAnalyzer().analyze(load(GCD_SAMPLE, instrument="biologic"))

    result = plot_bundle(analysis, kind="echem-cycling")

    try:
        assert result.figure.get_size_inches().tolist() == list(DEFAULT_FIGURE_SIZE)
        assert len(result.axes) == 1
        assert result.ax.get_xlabel() == "Cycle number"
        assert result.ax.get_ylabel() == "Capacity / mAh"
        assert result.metadata["lines"] >= 1
    finally:
        plt.close(result.figure)


def test_echem_efficiency_plotter_draws_ce_and_capacity_axes() -> None:
    ds = xr.Dataset(
        {
            "coulombic_efficiency_percent": (("cycle_number",), [98.5, 99.0]),
            "charge_capacity_mah": (("cycle_number",), [1.0, 0.95]),
            "discharge_capacity_mah": (("cycle_number",), [0.985, 0.94]),
        },
        coords={"cycle_number": [1, 2]},
    )
    analysis = DataBundle(data=ds, meta=Metadata(technique=["gcd"], active_material_mass="2 mg"))

    result = plot_bundle(analysis, kind="echem-efficiency")

    try:
        _assert_default_sized_figure(result)
        assert result.ax.get_xlabel() == "Cycle number"
        assert result.ax.get_ylabel() == "Coulombic efficiency / %"
        assert result.axes[1].get_ylabel() == "Specific capacity / mAh g$^{-1}$"
        assert result.metadata["lines"] == 3
    finally:
        plt.close(result.figure)


def test_unknown_plotter_kind_raises_clear_error() -> None:
    bundle = load(CV_SAMPLE, instrument="biologic")

    with pytest.raises(KeyError, match="未注册绘图类型"):
        plot_bundle(bundle, kind="echem-unknown")


def test_timestamped_log_dir_and_save_plot_result(tmp_path: Path) -> None:
    bundle = load(CV_SAMPLE, instrument="biologic")
    result = plot_bundle(bundle, kind="echem-cv", cycles=[1], max_cycles=1)
    output_dir = timestamped_log_dir(domain="echem", root=tmp_path, timestamp="20260604_120000")

    try:
        output = save_plot_result(result, "cv.png", output_dir=output_dir)

        assert output == tmp_path / "echem_20260604_120000" / "cv.png"
        assert output.exists()
    finally:
        plt.close(result.figure)


def _two_cycle_gcd_bundle(active_material_mass: str | None = None) -> DataBundle:
    ds = xr.Dataset(
        {
            "cycle_number": (("record",), [1, 1, 2, 2]),
            "capacity_mah": (("record",), [0.0, 1.0, 0.0, 0.5]),
            "voltage_v": (("record",), [1.3, 0.9, 1.25, 1.0]),
            "current_ma": (("record",), [-1.0, -1.0, -2.0, -2.0]),
        },
        coords={"record": [1, 2, 3, 4]},
    )
    return DataBundle(data=ds, meta=Metadata(technique=["gcd"], active_material_mass=active_material_mass))


def _assert_default_sized_figure(result) -> None:
    assert result.figure.get_size_inches().tolist() == list(DEFAULT_FIGURE_SIZE)


def _assert_gcd_dual_axis_labels(result, xlabel: str) -> None:
    assert len(result.axes) == 2
    assert result.ax.get_xlabel() == xlabel
    assert result.ax.get_ylabel() == "Voltage / V"
    assert result.axes[1].get_ylabel() == "Current / mA"
