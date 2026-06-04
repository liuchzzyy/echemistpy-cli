from pathlib import Path
from typing import Any

import numpy as np
import pytest

from echemistpy.io import load

SAMPLES = Path("Samples/Echem")
LANHE_CCS = SAMPLES / "Lanhe" / "AA.ccs"
LANHE_XLSX = SAMPLES / "Lanhe" / "AA.xlsx"
BIOLOGIC_MPR = SAMPLES / "Biologic" / "Trial02_GCD" / "ACETATE BUFFER CYCING_C01.mpr"
BIOLOGIC_FILES = {
    SAMPLES / "Biologic" / "Trial01_CV" / "LC-Zn-1M Zn-alphaMnO2-CV_01mVs_01_OCV_C02.mpr": {
        "rows": 61,
        "technique": {"echem", "ocv"},
        "variables": {"ewe_v", "flags"},
    },
    SAMPLES / "Biologic" / "Trial01_CV" / "LC-Zn-1M Zn-alphaMnO2-CV_01mVs_02_CV_C02.mpr": {
        "rows": 11764,
        "technique": {"echem", "cv"},
        "variables": {"cycle_number", "step_time_s", "ewe_v", "current_ma", "control_voltage_v", "charge_c", "current_range"},
    },
    BIOLOGIC_MPR: {
        "rows": 75,
        "technique": {"echem", "gcd", "gpcl"},
        "variables": {"cycle_number", "ewe_v", "ece_v", "current_ma", "capacity_mah", "dq_mah", "charge_discharge_capacity_mah", "half_cycle"},
    },
    SAMPLES / "Biologic" / "Trial03_EIS" / "EMD-2V-2mAh-1M+02M-40mL_02_PEIS_C01.mpr": {
        "rows": 37,
        "technique": {"echem", "eis", "peis"},
        "variables": {"frequency_hz", "re_z_ohm", "neg_im_z_ohm", "z_mag_ohm", "phase_deg", "capacitance_series_uf", "abs_current_ma"},
    },
}


@pytest.mark.parametrize("path", [LANHE_CCS, LANHE_XLSX])
def test_lanhe_samples_load_to_echem_schema(path: Path) -> None:
    bundle = load(path, instrument="lanhe")
    dataset = bundle.data

    assert bundle.meta.instrument == "lanhe"
    assert bundle.meta.technique == ["echem", "gcd"]
    assert dataset.sizes["record"] == 85756
    assert {
        "cycle_number",
        "step_number",
        "work_mode",
        "step_in_process",
        "voltage_v",
        "current_ua",
        "capacity_uah",
        "energy_uwh",
        "specific_energy_mwh_g",
        "humidity_percent",
        "channel_number",
    }.issubset(dataset.data_vars)
    assert {"record", "time_s", "systime"}.issubset(dataset.coords)
    assert np.issubdtype(dataset["time_s"].dtype, np.number)
    assert "WorkMode" not in dataset.data_vars
    assert "Energy_uWh" not in dataset.data_vars


@pytest.mark.parametrize(("path", "expected"), BIOLOGIC_FILES.items())
def test_biologic_mpr_samples_load_to_echem_schema(path: Path, expected: dict[str, Any]) -> None:
    bundle = load(path, instrument="biologic")
    dataset = bundle.data

    assert bundle.meta.instrument == "biologic"
    assert set(bundle.meta.technique) == expected["technique"]
    assert dataset.sizes["record"] == expected["rows"]
    assert expected["variables"].issubset(dataset.data_vars)
    assert {"record", "time_s", "systime"}.issubset(dataset.coords)
    assert np.issubdtype(dataset["time_s"].dtype, np.number)
    assert all("/" not in str(name) and "|" not in str(name) and "µ" not in str(name) for name in dataset.data_vars)


def test_biologic_directory_loads_mpr_files() -> None:
    bundle = load(SAMPLES / "Biologic", instrument="biologic")

    assert bundle.is_tree
    assert bundle.meta.instrument == "biologic"
    assert bundle.meta.raw_metadata["n_files"] == 4
