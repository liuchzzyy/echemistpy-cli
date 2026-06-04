import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.data import (
    DataBundle,
    DataStandardizer,
    Metadata,
    save_bundle,
    save_combined,
    standardize_bundle,
)
from echemistpy.io import load
from echemistpy.io.summary import summarize_data
from echemistpy.io.writer import write_bundle


def test_standardize_bundle_uses_echem_aliases_for_subtechniques() -> None:
    ds = xr.Dataset(
        {
            "Voltage/V": (("record",), [1.0, 2.0]),
            "Current/uA": (("record",), [100.0, 200.0]),
            "Capacity/uAh": (("record",), [5.0, 10.0]),
        },
        coords={"record": [1, 2]},
    )
    metadata = Metadata(technique=["gcd"], instrument="lanhe")

    bundle = standardize_bundle(DataBundle(data=ds, meta=metadata))

    assert {"voltage_v", "current_ua", "capacity_uah"}.issubset(bundle.data.data_vars)
    assert bundle.meta.technique == ["gcd"]
    assert bundle.provenance["standardized"] is True


def test_standardizer_preserves_non_equivalent_rename_conflicts() -> None:
    ds = xr.Dataset(
        {
            "Time": (("record",), [1.0, 2.0]),
            "time_s": (("record",), [10.0, 20.0]),
        },
        coords={"record": [1, 2]},
    )

    out = DataStandardizer(ds, techniques="echem").standardize_column_names().get_dataset()

    assert "time_s" in out
    assert "time_s_1" in out
    np.testing.assert_allclose(out["time_s"].values, [10.0, 20.0])
    np.testing.assert_allclose(out["time_s_1"].values, [1.0, 2.0])


def test_save_bundle_writes_data_and_metadata(tmp_path: Path) -> None:
    ds = xr.Dataset(
        {"Voltage/V": (("record",), [1.1, 1.2])},
        coords={"record": [1, 2], "time_s": (("record",), [0.0, 1.0])},
    )
    bundle = standardize_bundle(
        DataBundle(
            data=ds,
            meta=Metadata(technique=["gcd"], instrument="lanhe", sample_name="cell-a"),
            provenance={"source_path": "in-memory"},
        )
    )

    csv_path = tmp_path / "bundle.csv"
    save_bundle(bundle, csv_path)
    csv = pd.read_csv(csv_path)
    assert "voltage_v" in csv.columns

    info_path = tmp_path / "bundle.dat"
    save_bundle(bundle, info_path)
    saved_info = json.loads(info_path.read_text(encoding="utf-8"))
    assert saved_info["schema"] == bundle.schema
    assert saved_info["sample_name"] == "cell-a"

    nc_path = tmp_path / "bundle.nc"
    save_combined(bundle, nc_path)
    with xr.open_dataset(nc_path, engine="h5netcdf") as loaded:
        assert loaded.attrs["schema"] == bundle.schema
        assert loaded.attrs["sample_name"] == "cell-a"
        assert "voltage_v" in loaded


def test_io_writer_returns_output_path(tmp_path: Path) -> None:
    ds = xr.Dataset({"current/mA": (("record",), [1.0, 2.0])}, coords={"record": [1, 2]})
    bundle = standardize_bundle(DataBundle(data=ds, meta=Metadata(technique=["cv"], instrument="biologic")))

    output = tmp_path / "written.csv"
    written = write_bundle(bundle, output)

    assert written == output
    assert output.exists()
    assert "current_ma" in pd.read_csv(output).columns


def test_summary_supports_bundle() -> None:
    ds = xr.Dataset({"current/mA": (("record",), [1.0, 2.0])}, coords={"record": [1, 2]})
    metadata = Metadata(technique=["cv"], instrument="biologic", sample_name="cv-cell")
    bundle = standardize_bundle(DataBundle(data=ds, meta=metadata))

    bundle_summary = summarize_data(bundle, "memory")

    assert bundle_summary.sample_name == "cv-cell"
    assert "current_ma" in bundle_summary.variables


def test_load_returns_echem_bundle_shape() -> None:
    path = Path("Samples/Echem/Biologic/Trial02_GCD/ACETATE BUFFER CYCING_C01.mpr")

    bundle = load(path, instrument="biologic")

    assert isinstance(bundle, DataBundle)
    assert bundle.meta.instrument == "biologic"
    assert {"echem", "gcd", "gpcl"}.issubset(set(bundle.meta.technique))
    assert bundle.data.sizes["record"] == 75
