from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from echemistpy.io import load
from echemistpy.io.plugins.xrd_mspd import MSPDReader


def _write_xye(path: Path, date: str, intensity_offset: float = 0.0) -> None:
    lines = [
        f"# {path.name} : mythproc.py merge source.dat",
        "# IsMon = [100, 101, 102]  IsPos = [-1.5000, -2.5000, -3.5000]",
        "# Wave = 0.4962  CalPath /tmp/calibration.cal",
        f"# Date = {date} Dt = 4.8000  Bin = 0.0060  <imon> = 101 +/- 1 (Min/Max) 100 / 102",
        "# icurr 250.1  mocoIn 1.0E-05  mocoOut 0.8  i15 7000  imon 101  i7 4",
        "#      3",
        f"1.0000 {10.0 + intensity_offset} 1.0",
        f"1.0060 {20.0 + intensity_offset} 1.5",
        f"1.0120 {30.0 + intensity_offset} 2.0",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_mspd_reader_loads_single_xye(tmp_path: Path) -> None:
    xye_path = tmp_path / "sample_f00001.xye"
    _write_xye(xye_path, "2025-09-23_17:48:07")

    bundle = MSPDReader(xye_path).load()

    assert bundle.meta.technique == ["xrd", "in_situ"]
    assert bundle.meta.start_time == "2025-09-23 17:48:07"
    assert bundle.meta.wave_number == "0.4962"
    np.testing.assert_allclose([float(bundle.meta.raw_metadata["exposure_time_s"])], [4.8])
    np.testing.assert_allclose([float(bundle.meta.raw_metadata["monitor_mean"])], [101.0])
    assert isinstance(bundle.data, xr.Dataset)
    assert bundle.data.sizes["2theta"] == 3
    assert "intensity_error" in bundle.data
    assert "d_spacing" in bundle.data.coords
    np.testing.assert_allclose(np.asarray(bundle.data["intensity"].values, dtype=float), np.array([10.0, 20.0, 30.0]))


def test_mspd_reader_merges_xye_directory_as_operando(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    _write_xye(run_dir / "sample_f00001.xye", "2025-09-23_17:48:07", intensity_offset=0.0)
    _write_xye(run_dir / "sample_f00002.xye", "2025-09-23_17:48:17", intensity_offset=5.0)

    bundle = load(tmp_path, instrument="alba_mspd", standardize=False)

    assert isinstance(bundle.data, xr.DataTree)
    node_ds = bundle.data["run_a"].dataset
    assert bundle.meta.technique == ["xrd", "operando"]
    assert bundle.meta.raw_metadata["n_files"] == 2
    assert node_ds.sizes["record"] == 2
    assert node_ds.sizes["2theta"] == 3
    assert list(node_ds["filename"].values) == ["sample_f00001.xye", "sample_f00002.xye"]
    np.testing.assert_allclose(node_ds["time_s"].values, [0.0, 10.0])
    np.testing.assert_allclose(node_ds["exposure_time_s"].values, [4.8, 4.8])
    np.testing.assert_allclose(node_ds["intensity"].isel(record=1).values, [15.0, 25.0, 35.0])
