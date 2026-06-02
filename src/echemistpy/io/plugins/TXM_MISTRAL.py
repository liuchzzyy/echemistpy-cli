# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""TXM Data Reader for MISTRAL beamline HDF5 files."""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import xarray as xr

from echemistpy.io.base_reader import BaseReader
from echemistpy.io.reader_utils import apply_standard_attrs_txm
from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class MISTRALReader(BaseReader):
    """Reader for MISTRAL TXM .hdf5 files."""

    # --- Constants ---
    INSTRUMENT_NAME: ClassVar[str] = "ALBA_MISTRAL"
    DEFAULT_TECHNIQUE: ClassVar[list[str]] = ["txm", "ex situ"]
    DATE_REGEX: ClassVar[re.Pattern[str]] = re.compile(r"(\d{8})")

    # --- Loader Metadata ---
    supports_directories: ClassVar[bool] = False
    instrument: ClassVar[str] = "mistral"

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """Initialize the MISTRAL reader.

        Args:
            filepath: Path to HDF5 file or directory
            **kwargs: Additional metadata overrides
        """
        # Set default technique
        if "technique" not in kwargs:
            kwargs["technique"] = self.DEFAULT_TECHNIQUE
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Load a single MISTRAL HDF5 file.

        Args:
            path: Path to the HDF5 file
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        with h5py.File(path, "r") as f:
            if "SpecNormalized" not in f:
                raise ValueError(f"File {path} does not contain 'SpecNormalized' group.")

            group = f["SpecNormalized"]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"'SpecNormalized' in {path} is not a valid HDF5 Group.")

            ds = self._extract_dataset(group)

            # Extract date from filename
            start_time = self._extract_date(path.name)

            # Create RawDataInfo
            metadata = {"file_path": str(path), "start_time": start_time}
            raw_info = self._create_raw_info(metadata, default_sample_name=path.stem)

            return RawData(data=ds), raw_info

    def _extract_dataset(self, group: h5py.Group) -> xr.Dataset:
        """Extract data from HDF5 group and create xarray.Dataset.

        Args:
            group: HDF5 group containing the data

        Returns:
            xarray.Dataset with extracted data
        """
        data_cube = group["spectroscopy_normalized_aligned"][:]
        energy = group["energy"][:]
        rotation_angle = group["rotation_angle"][:] if "rotation_angle" in group else None

        x_pixel_size = group["x_pixel_size"][0] if "x_pixel_size" in group else 1.0
        y_pixel_size = group["y_pixel_size"][0] if "y_pixel_size" in group else 1.0

        # Create coordinates
        x_coords = np.arange(data_cube.shape[2]) * x_pixel_size
        y_coords = np.arange(data_cube.shape[1]) * y_pixel_size

        # Create Dataset
        ds = xr.Dataset(
            data_vars={
                "transmission": (["energy", "y", "x"], data_cube),
                "optical_density": (["energy", "y", "x"], -np.log(data_cube.astype(np.float64))),
            },
            coords={
                "energy": energy,
                "y": y_coords,
                "x": x_coords,
            },
            attrs={
                "x_pixel_size": x_pixel_size,
                "y_pixel_size": y_pixel_size,
                "instrument": self.instrument,
            },
        )

        if rotation_angle is not None:
            ds["rotation_angle"] = (["energy"], rotation_angle)

        apply_standard_attrs_txm(ds)
        return ds

    @staticmethod
    def _extract_date(filename: str) -> str | None:
        """Extract date from filename (e.g., 20230701 -> 2023-07-01).

        Args:
            filename: Filename to extract date from

        Returns:
            Date string in YYYY-MM-DD format, or None if not found
        """
        match = MISTRALReader.DATE_REGEX.search(filename)
        if match:
            with contextlib.suppress(Exception):
                d = match.group(1)
                return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        return None

    @staticmethod
    def _get_file_extension() -> str:
        """Get the file extension for this reader.

        Returns:
            File extension including the dot
        """
        return ".hdf5"
