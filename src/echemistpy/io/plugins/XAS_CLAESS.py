# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""XAS Data Reader for ALBA CLAESS beamline files."""

from __future__ import annotations

import contextlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

from echemistpy.io.base_reader import BaseReader
from echemistpy.io.reader_utils import apply_standard_attrs_xas
from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class CLAESSReader(BaseReader):
    """Reader for CLAESS XAS .dat files.

    Supports reading a single .dat file (which may contain multiple scans)
    or a directory containing multiple files.
    Filters files to only include those without digits in their names.
    """

    # --- Constants ---
    DEFAULT_TECHNIQUE: ClassVar[list[str]] = ["xas", "in_situ"]
    INSTRUMENT_NAME: ClassVar[str] = "CLAESS"
    DEFAULT_COLUMNS: ClassVar[list[str]] = [
        "energyc",
        "a_i0_1",
        "a_i0_2",
        "a_i1_1",
        "a_i1_2",
        "absorption",
    ]
    DATE_FORMAT: ClassVar[str] = "%a %b %d %H:%M:%S %Y"

    # --- Loader Metadata ---
    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "alba_claess"

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """Initialize the CLAESS reader.

        Args:
            filepath: Path to .dat file or directory
            **kwargs: Additional metadata overrides
        """
        # Set default technique
        if "technique" not in kwargs:
            kwargs["technique"] = self.DEFAULT_TECHNIQUE
        super().__init__(filepath, **kwargs)

    def load(self, edges: list[str] | None = None, **kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Load CLAESS file(s) and return RawData and RawDataInfo.

        Args:
            edges: Optional list of absorption edges to filter by
            **kwargs: Additional arguments

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        if not self.filepath:
            raise ValueError("filepath must be set before calling load()")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return self._load_single_file(path, **kwargs)
        if path.is_dir():
            return self._load_directory(path, edges=edges)

        raise ValueError(f"Path is neither a file nor a directory: {path}")

    def _load_single_file(self, path: Path, **_kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Internal method to load a single CLAESS file.

        Args:
            path: Path to the .dat file
            **_kwargs: Additional arguments (unused, prefixed with _ to silence linter)

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        if path.suffix.lower() != ".dat":
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        data_obj, metadata = self._read_spec_file(path)

        # Automatically clean data
        data = self._clean_data(data_obj)

        # Determine number of records
        n_records = (len(data.record) if "record" in data.dims else 1) if isinstance(data, xr.Dataset) else len(data.children)

        # Add metadata to Xarray object
        data.attrs.update({"file_name": [path.stem], "n_files": n_records})

        # Add units and long names if it's a Dataset
        if isinstance(data, xr.Dataset):
            apply_standard_attrs_xas(data)

        raw_info = RawDataInfo(
            sample_name=self.sample_name or path.stem,
            technique=list(self.technique),
            instrument=self.instrument,
            start_time=metadata.get("start_time"),
            others={"sample_names": [self.sample_name or path.stem], "n_files": n_records},
        )

        return RawData(data=data), raw_info

    def _clean_data(self, data: xr.Dataset | xr.DataTree) -> xr.Dataset | xr.DataTree:
        """Keep only specific columns defined in selected_columns.

        Args:
            data: xarray Dataset or DataTree

        Returns:
            Filtered Dataset or DataTree
        """
        if isinstance(data, xr.Dataset):
            existing_cols = [c for c in self.DEFAULT_COLUMNS if c in data.data_vars or c in data.coords]
            result = data[existing_cols]
            return result.to_dataset() if isinstance(result, xr.DataArray) else result

        if isinstance(data, xr.DataTree):
            new_dict: dict[str, xr.Dataset] = {}
            for node in data.subtree:
                if node.dataset is not None:
                    existing_cols = [c for c in self.DEFAULT_COLUMNS if c in node.dataset.data_vars or c in node.dataset.coords]
                    result = node.dataset[existing_cols]
                    new_dict[str(node.path)] = result.to_dataset() if isinstance(result, xr.DataArray) else result
            return xr.DataTree.from_dict(new_dict, name=data.name)

        return data

    def _read_spec_file(self, path: Path) -> tuple[xr.Dataset | xr.DataTree, dict[str, Any]]:
        """Parse a SPEC-like .dat file with multiple scans.

        Args:
            path: Path to the .dat file

        Returns:
            Tuple of (merged_dataset, metadata_dict)
        """
        datasets, scan_times, header = self._parse_spec_file(path)
        merged = self._merge_scans(datasets, scan_times, path.stem)

        if merged is None:
            raise ValueError(f"Failed to merge scans in {path}")

        start_time = merged.attrs.get("start_time") if isinstance(merged, xr.Dataset) else None

        return merged, {"header": header, "start_time": start_time}

    def _parse_spec_file(self, path: Path) -> tuple[dict[str, xr.Dataset], dict[str, datetime], str]:
        """Internal method to parse SPEC file into raw datasets and times.

        Args:
            path: Path to the .dat file

        Returns:
            Tuple of (datasets_dict, scan_times_dict, header)
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split by #S lines
        scans_raw = re.split(r"^#S\s+", content, flags=re.MULTILINE)
        header = scans_raw[0]
        scans_raw = scans_raw[1:]

        if not scans_raw:
            return self._parse_simple_table(path, header)

        datasets = {}
        scan_times = {}
        for scan_content in scans_raw:
            scan_id, ds, scan_time = self._parse_single_scan(scan_content, path)
            if ds is not None:
                datasets[f"scan_{scan_id}"] = ds
                if scan_time:
                    scan_times[f"scan_{scan_id}"] = scan_time

        return datasets, scan_times, header

    @staticmethod
    def _parse_simple_table(path: Path, header: str) -> tuple[dict[str, xr.Dataset], dict[str, datetime], str]:
        """Parse a simple table file without SPEC headers.

        Args:
            path: Path to the file
            header: Header string

        Returns:
            Tuple of (datasets_dict, scan_times_dict, header)
        """
        try:
            df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
            ds = df.to_xarray().rename({"index": "point"})
            return {"scan_1": ds}, {}, header
        except Exception as e:
            raise ValueError(f"No scans found and failed to read as table in {path}: {e}") from e

    def _parse_single_scan(self, scan_content: str, path: Path) -> tuple[str, xr.Dataset | None, datetime | None]:
        """Parse a single scan block from a SPEC file.

        Args:
            scan_content: Scan content string
            path: File path

        Returns:
            Tuple of (scan_id, dataset, scan_time)
        """
        lines = scan_content.splitlines()
        if not lines:
            return "unknown", None, None

        scan_id = lines[0].split()[0] if lines[0].split() else "unknown"
        data_lines: list[list[Any]] = []
        columns: list[str] = []
        scan_time: datetime | None = None

        for line in lines[1:]:
            if line.startswith("#L"):
                columns = line[3:].strip().split()
            elif line.startswith("#D"):
                with contextlib.suppress(Exception):
                    scan_time = self.parse_date(line[3:].strip())
            elif not line.startswith("#") and line.strip():
                data_lines.append(line.split())

        if not data_lines:
            return scan_id, None, None

        if not columns:
            columns = [f"col_{i}" for i in range(len(data_lines[0]))]

        try:
            df = pd.DataFrame(data_lines, columns=list(columns)).astype(float)  # type: ignore
            # Calculate absorption if possible
            if all(c in df.columns for c in ["a_i0_1", "a_i0_2", "a_i1_1", "a_i1_2"]):
                ratio = (df["a_i0_1"] + df["a_i0_2"]) / (df["a_i1_1"] + df["a_i1_2"])
                df["absorption"] = np.log(ratio.where(ratio > 0))

            if "energyc" in df.columns:
                df = df.drop_duplicates(subset=["energyc"]).set_index("energyc")

            return scan_id, df.to_xarray(), scan_time
        except Exception as e:
            logger.warning("Failed to parse scan %s in %s: %s", scan_id, path, e)
            return scan_id, None, None

    @staticmethod
    def _interpolate_datasets(ds_list: list[xr.Dataset]) -> list[xr.Dataset]:
        """Interpolate multiple datasets onto a common energy grid.

        Args:
            ds_list: List of xarray Datasets

        Returns:
            List of interpolated Datasets
        """
        all_energies = [ds.energyc.values for ds in ds_list if "energyc" in ds.coords]
        if not all_energies:
            return ds_list

        ref_energy = max(all_energies, key=len)
        interpolated_list = []
        for ds in ds_list:
            if "energyc" in ds.coords:
                new_vars: dict[str, tuple[list[str], np.ndarray]] = {}
                for var in ds.data_vars:
                    f = interp1d(ds.energyc.values, ds[var].values, bounds_error=False, fill_value=np.nan)
                    new_vars[str(var)] = (["energyc"], f(ref_energy))
                interpolated_list.append(xr.Dataset(new_vars, coords={"energyc": ref_energy}))
            else:
                interpolated_list.append(ds)
        return interpolated_list

    @staticmethod
    def _calculate_scan_times(combined: xr.Dataset, scan_ids: list[str], scan_times: dict[str, datetime]) -> xr.Dataset:
        """Calculate and add systime and time_s to the combined dataset.

        Args:
            combined: Combined xarray Dataset
            scan_ids: List of scan IDs
            scan_times: Dictionary of scan times

        Returns:
            Modified Dataset with time coordinates
        """
        if not scan_times:
            return combined

        systimes = pd.to_datetime([scan_times.get(sid) for sid in scan_ids])
        combined.coords["systime"] = ("record", systimes)

        valid_times = systimes[systimes.notnull()]
        if not valid_times.empty:
            t0 = valid_times[0]
            combined.coords["time_s"] = ("record", (systimes - t0).total_seconds())
        return combined

    def _merge_scans(self, datasets: dict[str, xr.Dataset], scan_times: dict[str, datetime], name: str) -> xr.Dataset | xr.DataTree | None:
        """Internal method to merge multiple scan datasets into one.

        Args:
            datasets: Dictionary of datasets
            scan_times: Dictionary of scan times
            name: Name for the merged dataset

        Returns:
            Merged Dataset or DataTree, or None if no datasets
        """
        if not datasets:
            return None

        first_scan_id = next(iter(datasets.keys()))
        start_time = scan_times[first_scan_id].strftime("%Y-%m-%d %H:%M:%S") if first_scan_id in scan_times else None

        if len(datasets) > 1:
            try:
                ds_list = self._interpolate_datasets(list(datasets.values()))
                combined = xr.concat(ds_list, dim="record")
                scan_ids = list(datasets.keys())
                combined = combined.assign_coords(record=np.arange(1, len(datasets) + 1))
                combined = combined.assign_coords(file_name=("record", [name] * len(datasets)))
                combined = self._calculate_scan_times(combined, scan_ids, scan_times)
                combined.attrs["start_time"] = start_time
                return combined
            except Exception as e:
                logger.warning("Failed to merge scans for %s: %s. Returning DataTree instead.", name, e)

        if len(datasets) == 1:
            ds = next(iter(datasets.values()))
            ds = ds.expand_dims("record").assign_coords(record=[1], file_name=("record", [name]))
            sid = next(iter(datasets.keys()))
            if sid in scan_times:
                ds.coords["systime"] = ("record", pd.to_datetime([scan_times[sid]]))
                ds.coords["time_s"] = ("record", [0.0])
            ds.attrs["start_time"] = start_time
            return ds

        return xr.DataTree.from_dict(datasets, name=name)

    @classmethod
    def parse_date(cls, date_str: str) -> datetime:
        """Parse SPEC date format: Thu Dec 11 12:52:40 2025.

        Args:
            date_str: Date string in SPEC format

        Returns:
            datetime object
        """
        return datetime.strptime(date_str, cls.DATE_FORMAT)

    def _load_directory(self, path: Path, edges: list[str] | None = None, **_kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Load all relevant files in a directory.

        Args:
            path: Path to the directory
            edges: Optional list of absorption edges to filter by
            **_kwargs: Additional arguments (unused, prefixed with _ to silence linter)

        Returns:
            Tuple of (RawData with DataTree, merged RawDataInfo)
        """
        all_files = list(path.rglob("*.dat"))
        relevant_files = [f for f in all_files if not re.search(r"_\d{3}$", f.stem)]

        if not relevant_files:
            raise FileNotFoundError(f"No relevant .dat files found in {path}")

        if edges is None:
            edges = self._auto_detect_edges_from_folders(relevant_files)
            if edges:
                logger.info("Auto-detected edges from folder names: %s", edges)

        tree = xr.DataTree(name=path.name)
        all_infos: list[RawDataInfo] = []

        if edges:
            groups = self._group_files_by_edge(relevant_files, edges, path)
            for (edge, clean_rel_path), files in groups.items():
                all_infos.extend(self._process_edge_group(edge, clean_rel_path, files, tree))
        else:
            for f in sorted(relevant_files):
                try:
                    raw_data, raw_info = self._load_single_file(f)
                    rel_path = f.relative_to(path)
                    node_name = str(rel_path.with_suffix("")).replace("\\", "_").replace("/", "_")

                    if isinstance(raw_data.data, xr.DataTree):
                        for name_path, child in raw_data.data.children.items():
                            tree[f"{node_name}/{name_path}"] = child
                    else:
                        tree[node_name] = raw_data.data
                    all_infos.append(raw_info)
                except Exception as e:
                    logger.warning("Failed to load file %s: %s", f, e)

        if not tree.children and not tree.has_data:
            raise RuntimeError(f"Failed to load any relevant files from {path}")

        root_info = self._merge_infos(all_infos, path)
        tree.attrs = {"file_name": [info.sample_name for info in all_infos], "n_files": root_info.others.get("n_files")}
        return RawData(data=tree), root_info

    @staticmethod
    def _auto_detect_edges_from_folders(files: list[Path]) -> list[str] | None:
        """Auto-detect edges from the parent folder names of the files.

        Args:
            files: List of relevant .dat files

        Returns:
            Detected edge list (sorted by length descending) or None
        """
        # Periodic table elements
        elements = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
        ]

        detected = set()
        for f in files:
            # Get the parent folder name
            folder_name = f.parent.name

            # Look for element symbols in the folder name
            for el in elements:
                # Pattern:
                # 1. Start or non-alphanumeric before
                # 2. Uppercase letter (e.g., MnFoil) or non-alphanumeric or end after
                pattern = rf"(^|[^a-zA-Z0-9]){el}([A-Z]|[^a-zA-Z0-9]|$)"
                if re.search(pattern, folder_name):
                    detected.add(el)

        if detected:
            # Sort by length descending to prioritize longer symbols (e.g., Mn over N)
            return sorted(detected, key=lambda x: (-len(x), x))

        return None

    @staticmethod
    def _group_files_by_edge(files: list[Path], edges: list[str], root_path: Path) -> dict[tuple[str, str], list[Path]]:
        """Group files by edge and clean relative path.

        Args:
            files: List of .dat files
            edges: List of edge names
            root_path: Root path

        Returns:
            Dictionary mapping (edge, clean_rel_path) to file list
        """
        groups: dict[tuple[str, str], list[Path]] = {}
        for f in files:
            rel_parent = f.parent.relative_to(root_path)
            matched_edge: str | None = None

            folder_name = f.parent.name
            for edge in edges:
                pattern = rf"(^|[^a-zA-Z0-9]){edge}([A-Z]|[^a-zA-Z0-9]|$)"
                if re.search(pattern, folder_name):
                    matched_edge = edge
                    break

            if matched_edge:
                parts = [p for p in rel_parent.parts if p.lower() != matched_edge.lower()]
                clean_rel_path = "/".join(parts)
                key = (matched_edge, clean_rel_path)
                if key not in groups:
                    groups[key] = []
                groups[key].append(f)
        return groups

    def _process_edge_group(self, edge: str, clean_rel_path: str, files: list[Path], tree: xr.DataTree) -> list[RawDataInfo]:
        """Process a group of files for a specific edge and path.

        Args:
            edge: Edge name
            clean_rel_path: Clean relative path
            files: List of files
            tree: DataTree to populate

        Returns:
            List of RawDataInfo objects
        """
        all_datasets: dict[str, xr.Dataset] = {}
        all_scan_times: dict[str, datetime] = {}
        for f in sorted(files):
            try:
                ds_dict, st_dict, _ = self._parse_spec_file(f)
                for k, v in ds_dict.items():
                    all_datasets[f"{f.stem}_{k}"] = v
                for k, v in st_dict.items():
                    all_scan_times[f"{f.stem}_{k}"] = v
            except Exception as e:
                logger.warning("Failed to parse %s: %s", f, e)

        if not all_datasets:
            return []

        merged_ds = self._merge_scans(all_datasets, all_scan_times, edge)
        if merged_ds is None:
            return []
        merged_ds = self._clean_data(merged_ds)

        # Determine a name for this dataset node
        stems = [f.stem for f in files]
        ds_name = stems[0] if len(files) == 1 else os.path.commonprefix(stems).rstrip("_")
        if not ds_name or len(ds_name) <= 2:
            ds_name = "merged_data"

        # Build node path
        full_node_name = self._get_node_path(edge, clean_rel_path, ds_name)

        try:
            tree[full_node_name] = merged_ds
        except Exception as e:
            logger.warning("Alignment failed for %s: %s. Renaming.", full_node_name, e)
            with contextlib.suppress(Exception):
                tree[f"{full_node_name}_data"] = merged_ds

        n_records = len(merged_ds.record) if "record" in merged_ds.dims else 1
        return [
            RawDataInfo(
                sample_name=ds_name,
                technique=list(self.technique),
                instrument=self.instrument,
                start_time=merged_ds.attrs.get("start_time"),
                others={"n_files": n_records},
            )
        ]

    @staticmethod
    def _get_node_path(edge: str, clean_rel_path: str, ds_name: str) -> str:
        """Determine the tree node path for a dataset.

        Args:
            edge: Edge name
            clean_rel_path: Clean relative path
            ds_name: Dataset name

        Returns:
            Node path string
        """
        if not clean_rel_path:
            return f"{edge}/{ds_name}"

        clean_parts = clean_rel_path.split("/")
        last_clean_part = clean_parts[-1]

        # Check if ds_name is redundant with last_clean_part
        should_merge = ds_name == f"{last_clean_part}_{edge}" or (ds_name.endswith(f"_{edge}") and ds_name[: -len(edge) - 1] == last_clean_part)

        if should_merge:
            return f"{edge}/{'/'.join(clean_parts[:-1])}/{ds_name}" if len(clean_parts) > 1 else f"{edge}/{ds_name}"

        return f"{edge}/{clean_rel_path}/{ds_name}"

    def _merge_infos(self, infos: list[RawDataInfo], root_path: Path) -> RawDataInfo:
        """Merge multiple RawDataInfo objects into one.

        Args:
            infos: List of RawDataInfo objects
            root_path: Root path

        Returns:
            Merged RawDataInfo
        """
        if not infos:
            return RawDataInfo()

        base = infos[0]
        all_techs = set()
        total_files = 0
        sample_names = []

        for info in infos:
            for t in info.technique:
                all_techs.add(t)
            n = info.others.get("n_files")
            total_files += n if n is not None else 1
            sample_names.append(info.sample_name)

        return RawDataInfo(
            sample_name=self.sample_name or root_path.name,
            technique=list(all_techs),
            instrument=base.instrument,
            start_time=base.start_time,
            others={"n_files": total_files, "root_path": str(root_path), "sample_names": sample_names},
        )

    @staticmethod
    def _get_file_extension() -> str:
        """Get the file extension for this reader.

        Returns:
            File extension including the dot
        """
        return ".dat"
