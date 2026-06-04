"""BioLogic MPR binary reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from galvani import BioLogic

from echemistpy.data.models import RawData, RawDataInfo
from echemistpy.io.base_reader import BaseReader
from echemistpy.io.contracts import ReaderSpec

BIOLOGIC_STEP_TIME_COLUMN_ID = 182


class BiologicMprReader(BaseReader):
    """Reader for BioLogic EC-Lab binary MPR files."""

    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "biologic"
    spec: ClassVar[ReaderSpec] = ReaderSpec(
        name="biologic_mpr",
        extensions=(".mpr",),
        instruments=("biologic",),
        techniques=("echem",),
        supports_directory=True,
        description="BioLogic EC-Lab binary MPR files",
    )

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """Initialize the BioLogic MPR reader."""
        if "technique" not in kwargs:
            kwargs["technique"] = ["echem"]
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Load one BioLogic MPR file."""
        _patch_galvani_columns()
        mpr = BioLogic.MPRfile(str(path))
        dataset = self._create_dataset(mpr)
        metadata = self._metadata(path, mpr)
        raw_info = RawDataInfo(
            sample_name=self.sample_name or path.stem,
            start_time=self.start_time or metadata.get("start_time"),
            operator=self.operator,
            technique=self._techniques(mpr.data.dtype.names or ()),
            instrument=self.instrument,
            active_material_mass=self.active_material_mass,
            wave_number=self.wave_number,
            others=metadata,
        )
        return RawData(data=dataset), raw_info

    @staticmethod
    def _create_dataset(mpr: BioLogic.MPRfile) -> xr.Dataset:
        """Convert a Galvani MPR object to an xarray Dataset."""
        record_count = len(mpr.data)
        data_vars = {
            str(name): (("record",), _plain_array(mpr.data[name]))
            for name in mpr.data.dtype.names or ()
            if name != "time/s"
        }
        dataset = xr.Dataset(data_vars=data_vars, coords={"record": np.arange(1, record_count + 1)})

        if "time/s" in (mpr.data.dtype.names or ()):
            time_values = pd.to_numeric(mpr.data["time/s"]).astype(float)
            dataset = dataset.assign_coords(time_s=(("record",), time_values))
            dataset.time_s.attrs.update({"units": "s", "long_name": "Relative Time"})

            timestamp = getattr(mpr, "timestamp", None)
            if timestamp is not None:
                systime = pd.to_datetime(timestamp) + pd.to_timedelta(time_values, unit="s")
                dataset = dataset.assign_coords(systime=(("record",), systime))
                dataset.systime.attrs.update({"long_name": "System Time"})

        dataset.attrs["source_format"] = "mpr"
        dataset.attrs["reader"] = BiologicMprReader.__name__
        return dataset

    @staticmethod
    def _metadata(path: Path, mpr: BioLogic.MPRfile) -> dict[str, Any]:
        timestamp = getattr(mpr, "timestamp", None)
        return {
            "file_path": str(path),
            "source_format": "mpr",
            "reader": BiologicMprReader.__name__,
            "start_time": None if timestamp is None else timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "column_ids": [int(column_id) for column_id in getattr(mpr, "cols", [])],
            "version": int(getattr(mpr, "version", -1)),
            "n_records": _plain_int(getattr(mpr, "npts", len(mpr.data))),
        }

    def _techniques(self, columns: tuple[str, ...]) -> list[str]:
        names = set(columns)
        techniques = list(self.technique)
        if "freq/Hz" in names:
            techniques.extend(["eis", "peis"])
        elif {"Q charge/discharge/mA.h", "half cycle"} & names:
            techniques.extend(["gcd", "gpcl"])
        elif {"control/V", "<I>/mA"} <= names:
            techniques.append("cv")
        elif "Ewe/V" in names:
            techniques.append("ocv")
        return list(dict.fromkeys(techniques))


def _patch_galvani_columns() -> None:
    """Register BioLogic column ids used by current EC-Lab exports."""
    BioLogic.VMPdata_colID_dtype_map.setdefault(
        BIOLOGIC_STEP_TIME_COLUMN_ID,
        ("step time/s", "<f8"),
    )


def _plain_array(values: np.ndarray) -> np.ndarray:
    """Return an array with metadata-safe scalar values."""
    if values.dtype.kind == "S":
        return values.astype(str)
    return values


def _plain_int(value: Any) -> int:
    """Convert Python or numpy scalar-like values to int."""
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array.item())
    return int(array.ravel()[0])


__all__ = ["BiologicMprReader"]
