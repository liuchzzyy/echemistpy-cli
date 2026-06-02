# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""XRD Data Reader for MSPD .xye files with metadata extraction using traitlets."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.io.base_reader import BaseReader
from echemistpy.io.reader_utils import apply_standard_attrs_xrd, merge_infos
from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class MSPDReader(BaseReader):
    """Reader for MSPD XRD .xye files.

    Supports reading a single .xye file or a directory containing multiple .xye files.
    When reading a directory, files are organized into an xarray.DataTree.
    """

    # --- Constants ---
    DATE_FORMAT: ClassVar[str] = "%Y-%m-%d_%H:%M:%S"
    DEFAULT_TECHNIQUE_SINGLE: ClassVar[list[str]] = ["xrd", "in_situ"]
    DEFAULT_TECHNIQUE_DIR: ClassVar[list[str]] = ["xrd", "operando"]
    INSTRUMENT_NAME: ClassVar[str] = "ALBA_MSPD"

    # --- Loader Metadata ---
    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "alba_mspd"

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """Initialize the MSPD reader.

        Args:
            filepath: Path to .xye file or directory
            **kwargs: Additional metadata overrides
        """
        # Technique will be determined later based on file vs directory
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Load a single MSPD .xye file.

        Args:
            path: Path to the .xye file
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        ds, metadata = self._read_single_xye(path)

        # Clean and enhance metadata
        metadata["file_path"] = str(path)
        cleaned_metadata = self._clean_metadata(metadata, path)

        # Extract wave number
        wave_val = self._extract_wave_number(cleaned_metadata)

        # Use in_situ technique for single file
        technique = list(self.technique) if self.technique != ["unknown"] else self.DEFAULT_TECHNIQUE_SINGLE

        raw_info = RawDataInfo(
            sample_name=self.sample_name or cleaned_metadata.get("sample_name", path.stem),
            start_time=self.start_time or cleaned_metadata.get("start_time"),
            technique=technique,
            instrument=self.instrument,
            operator=self.operator or cleaned_metadata.get("operator"),
            active_material_mass=self.active_material_mass or cleaned_metadata.get("active_material_mass"),
            wave_number=wave_val,
            others=cleaned_metadata,
        )

        return RawData(data=ds), raw_info

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any], filepath: Path) -> dict[str, Any]:
        """Clean and structure metadata from raw header data.

        Args:
            metadata: Raw metadata dictionary from file header
            filepath: Path to the file being processed

        Returns:
            Cleaned metadata dictionary with standardized keys
        """
        cleaned: dict[str, Any] = {}

        # Extract start time
        if "Date" in metadata:
            try:
                dt = MSPDReader.parse_date(metadata["Date"])
                cleaned["start_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                cleaned["start_time"] = metadata["Date"]

        # Extract wavelength
        if "Wave" in metadata:
            cleaned["wavelength"] = metadata["Wave"]

        # Store sample name from filename
        cleaned["sample_name"] = filepath.stem
        cleaned["file_path"] = str(filepath)

        return cleaned

    def _extract_wave_number(self, metadata: dict[str, Any]) -> str | None:
        """Extract wave number from metadata or traits.

        Args:
            metadata: Metadata dictionary

        Returns:
            Wavelength as a string, or None
        """
        if self.wave_number:
            return self.wave_number
        wavelength = metadata.get("wavelength")
        return str(wavelength) if wavelength is not None else None

    def _load_directory(self, path: Path, **_kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Load all MSPD .xye files in a directory into a DataTree.

        Args:
            path: Path to the directory
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (RawData with DataTree, merged RawDataInfo)
        """
        xye_files = sorted(path.rglob("*.xye"))
        if not xye_files:
            raise FileNotFoundError(f"No .xye files found in {path}")

        # Group files by parent directory
        groups: dict[Path, list[Path]] = {}
        for f in xye_files:
            groups.setdefault(f.parent, []).append(f)

        tree_dict: dict[str, xr.Dataset] = {}
        all_infos: list[RawDataInfo] = []

        for parent, files in groups.items():
            try:
                merged_ds, node_infos = self._process_directory_group(files)
                if merged_ds is None:
                    continue

                # Determine relative path for the tree
                if parent == path:
                    node_path = "/"
                else:
                    rel_path = parent.relative_to(path)
                    node_path = "/" + "/".join(rel_path.parts)

                tree_dict[node_path] = merged_ds

                # Merge and store metadata for this node
                node_info = self._merge_node_infos(node_infos, parent)
                merged_ds.attrs.update(node_info.to_dict())
                all_infos.append(node_info)

            except Exception as e:
                logger.error("Failed to load/merge files in %s: %s", parent, e)

        if not tree_dict:
            raise RuntimeError(f"Failed to load any valid .xye files from {path}")

        # Create DataTree from dictionary
        tree = xr.DataTree.from_dict(tree_dict, name=path.name)

        # Merge all infos for the root
        root_info = self._merge_infos(all_infos, path)
        tree.attrs.update(root_info.to_dict())

        return RawData(data=tree), root_info

    def _process_directory_group(self, files: list[Path]) -> tuple[xr.Dataset | None, list[RawDataInfo]]:
        """Process a group of files in the same directory.

        Args:
            files: List of .xye file paths in the same directory

        Returns:
            Tuple of (merged_dataset, list_of_infos) or (None, []) if all files failed
        """
        datasets = []
        infos = []
        for f in files:
            try:
                raw_data, raw_info = self._load_single_file(f)
                datasets.append(raw_data.data)
                infos.append(raw_info)
            except Exception as e:
                logger.warning("Skipping file %s due to error: %s", f, e)

        if not datasets:
            return None, []

        # Merge datasets along 'record' dimension
        merged_ds = xr.concat(datasets, dim="record")

        # Extract and convert times
        systimes = pd.to_datetime([info.start_time for info in infos])
        rel_times = (systimes - systimes[0]).total_seconds() if not systimes.isnull().all() else [np.nan] * len(infos)

        # Add coordinates
        merged_ds = merged_ds.assign_coords(
            record=np.arange(len(datasets)),
            filename=("record", [f.name for f in files]),
            systime=("record", systimes),
            time_s=("record", rel_times),
        )

        # Add metadata to coordinates
        merged_ds.time_s.attrs.update({"units": "s", "long_name": "Relative Time"})
        merged_ds.systime.attrs.update({"long_name": "System Time"})

        return merged_ds, infos

    def _merge_node_infos(self, infos: list[RawDataInfo], parent_path: Path) -> RawDataInfo:
        """Merge RawDataInfo objects for a single directory node.

        Args:
            infos: List of RawDataInfo objects from files in the same directory
            parent_path: Parent directory path

        Returns:
            Merged RawDataInfo
        """
        if not infos:
            return RawDataInfo()

        # Collect sample names (one per file)
        sample_names = [info.sample_name for info in infos if info.sample_name]

        # Collect unique operators and start times
        operators = sorted({info.operator for info in infos if info.operator})
        start_times = sorted({info.start_time for info in infos if info.start_time})
        masses = sorted({info.active_material_mass for info in infos if info.active_material_mass})
        wave_numbers = sorted({info.wave_number for info in infos if info.wave_number})

        # Use directory-specific technique
        technique = list(self.technique) if self.technique != ["unknown"] else self.DEFAULT_TECHNIQUE_DIR

        # Build others dict
        others = {
            "n_files": len(infos),
            "sample_names": sample_names,
            "filenames": [Path(info.get("file_path", "")).name for info in infos if info.get("file_path")],
        }

        # Add lists if multiple unique values
        if len(operators) > 1:
            others["all_operators"] = operators
        if len(masses) > 1:
            others["all_active_material_masses"] = masses
        if len(wave_numbers) > 1:
            others["all_wave_numbers"] = wave_numbers

        return RawDataInfo(
            sample_name=self.sample_name or parent_path.name,
            technique=technique,
            instrument=self.instrument,
            operator=self.operator or (operators[0] if len(operators) == 1 else None),
            start_time=self.start_time or (start_times[0] if len(start_times) == 1 else None),
            active_material_mass=self.active_material_mass or (masses[0] if len(masses) == 1 else None),
            wave_number=self.wave_number or (wave_numbers[0] if len(wave_numbers) == 1 else None),
            others=others,
        )

    def _merge_infos(self, infos: list[RawDataInfo], root_path: Path) -> RawDataInfo:
        """Merge multiple RawDataInfo objects from different directories.

        Args:
            infos: List of RawDataInfo objects from subdirectories
            root_path: Root path for determining folder name

        Returns:
            Merged RawDataInfo
        """
        if not infos:
            return RawDataInfo()

        # Collect all sample names from all subdirectories
        all_sample_names = []
        for info in infos:
            # Each info might have multiple sample_names in its 'others'
            if "sample_names" in info.others:
                all_sample_names.extend(info.others["sample_names"])
            elif info.sample_name:
                all_sample_names.append(info.sample_name)

        # Use directory-specific technique
        technique = list(self.technique) if self.technique != ["unknown"] else self.DEFAULT_TECHNIQUE_DIR

        # Calculate total files
        total_files = sum(info.others.get("n_files", 1) for info in infos)

        # Use merge_infos from utils for common logic
        merged_info = merge_infos(
            infos,
            root_path,
            sample_name_override=self.sample_name,
            operator_override=self.operator,
            start_time_override=self.start_time,
            active_material_mass_override=self.active_material_mass,
            wave_number_override=self.wave_number,
            technique=technique,
            instrument=self.instrument,
        )

        # Update with MSPD-specific fields
        merged_info.others["sample_names"] = all_sample_names
        merged_info.others["n_files"] = total_files

        return merged_info

    def _read_single_xye(self, filepath: Path) -> tuple[xr.Dataset, dict[str, Any]]:
        """Read a single .xye file and return an xarray Dataset and metadata.

        Args:
            filepath: Path to the .xye file

        Returns:
            Tuple of (Dataset, metadata_dict)
        """
        # Parse header metadata
        metadata = self._parse_header(filepath)

        # Read data columns
        try:
            df = pd.read_csv(
                filepath,
                comment="#",
                sep=r"\s+",
                names=["2theta", "intensity", "intensity_error"],
                engine="python",
            )
        except Exception as e:
            logger.error("Error reading %s: %s", filepath, e)
            raise

        # Create xarray Dataset
        ds = self._create_dataset(df, metadata)

        return ds, metadata

    def _create_dataset(self, df: pd.DataFrame, metadata: dict[str, Any]) -> xr.Dataset:
        """Create a standardized xarray.Dataset from DataFrame and metadata.

        Args:
            df: DataFrame with columns ['2theta', 'intensity', 'intensity_error']
            metadata: Metadata dictionary from file header

        Returns:
            xarray.Dataset with data variables and coordinates
        """
        # Create Dataset with data variables
        ds = xr.Dataset(
            {
                "intensity": (("2theta",), df["intensity"].values),
                "intensity_error": (("2theta",), df["intensity_error"].values),
            },
            coords={"2theta": df["2theta"].values},
        )

        # Apply standard attributes
        apply_standard_attrs_xrd(ds)

        # Calculate and add d-spacing if wavelength is available
        wave_to_use = self._get_wave_to_use(metadata)
        if wave_to_use is not None:
            d_spacing = self.calculate_d_spacing(ds["2theta"].values, wave_to_use)
            ds = ds.assign_coords(d_spacing=(("2theta",), d_spacing))
            ds.d_spacing.attrs.update({"units": "Å", "long_name": "d-spacing"})

        return ds

    @staticmethod
    def _parse_header(filepath: Path) -> dict[str, Any]:
        """Parse header lines from .xye file for metadata.

        Args:
            filepath: Path to the .xye file

        Returns:
            Dictionary with extracted metadata
        """
        metadata: dict[str, Any] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                # Extract Wave
                if "Wave =" in line and (match := re.search(r"Wave\s*=\s*([\d\.]+)", line)):
                    metadata["Wave"] = float(match.group(1))
                # Extract Date
                if "Date =" in line and (match := re.search(r"Date\s*=\s*([\d\-_:]+)", line)):
                    metadata["Date"] = match.group(1)
        return metadata

    def _get_wave_to_use(self, metadata: dict[str, Any]) -> float | None:
        """Determine the wavelength to use for d-spacing calculation.

        Args:
            metadata: Metadata dictionary from file

        Returns:
            Wavelength in Angstroms, or None if not available
        """
        if self.wave_number:
            try:
                return float(self.wave_number)
            except ValueError:
                logger.warning("Invalid wave_number trait: %s", self.wave_number)
        return metadata.get("Wave")

    @staticmethod
    def calculate_d_spacing(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
        """Calculate d-spacing from 2theta and wavelength using Bragg's Law.

        Bragg's Law: nλ = 2d sinθ
        For n=1: d = λ / (2 sinθ)

        Args:
            two_theta: Array of 2θ values in degrees
            wavelength: X-ray wavelength in Angstroms

        Returns:
            Array of d-spacing values in Angstroms
        """
        theta_rad = np.deg2rad(two_theta / 2.0)
        return wavelength / (2.0 * np.sin(theta_rad))

    @classmethod
    def parse_date(cls, date_str: str) -> datetime:
        """Parse MSPD date string format 'YYYY-MM-DD_HH:MM:SS' to datetime object.

        Args:
            date_str: Date string in MSPD format

        Returns:
            datetime object

        Raises:
            ValueError: If date string is empty or cannot be parsed
        """
        if not date_str:
            raise ValueError("Empty date string")
        return datetime.strptime(date_str, cls.DATE_FORMAT)

    @staticmethod
    def _get_file_extension() -> str:
        """Get the file extension for this reader.

        Returns:
            File extension including the dot
        """
        return ".xye"
