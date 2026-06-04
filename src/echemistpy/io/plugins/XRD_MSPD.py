# -*- coding: utf-8 -*-
"""MSPD .xye 格式 XRD 数据读取器。"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.data.models import DataBundle, Metadata
from echemistpy.data.utils import apply_standard_attrs_xrd, merge_metadata
from echemistpy.io.base_reader import BaseReader
from echemistpy.io.contracts import ReaderSpec

logger = logging.getLogger(__name__)


class MSPDReader(BaseReader):
    """MSPD XRD .xye 文件读取器。

    支持读取单个 .xye 文件或包含多个 .xye 文件的目录；目录读取结果组织为 xarray.DataTree。
    """

    # --- 解析常量 ---
    DATE_FORMAT: ClassVar[str] = "%Y-%m-%d_%H:%M:%S"
    DEFAULT_TECHNIQUE_SINGLE: ClassVar[list[str]] = ["xrd", "in_situ"]
    DEFAULT_TECHNIQUE_DIR: ClassVar[list[str]] = ["xrd", "operando"]
    INSTRUMENT_NAME: ClassVar[str] = "ALBA_MSPD"

    # --- reader 能力声明 ---
    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "alba_mspd"
    spec: ClassVar[ReaderSpec] = ReaderSpec(
        name="mspd_xye",
        extensions=(".xye",),
        instruments=("alba_mspd",),
        techniques=("xrd",),
        supports_directory=True,
        description="ALBA MSPD XRD XYE files",
    )

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """初始化 MSPD reader。

        Args:
            filepath: .xye 文件或目录路径
            **kwargs: 额外元数据覆盖项
        """
        # 技术类型会根据文件或目录读取方式决定。
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> DataBundle:
        """加载单个 MSPD .xye 文件。

        Args:
            path: .xye 文件路径
            **kwargs: 额外参数（未使用）

        Returns:
            DataBundle 数据包
        """
        ds, metadata = self._read_single_xye(path)

        # 清理并补充元数据。
        metadata["file_path"] = str(path)
        cleaned_metadata = self._clean_metadata(metadata, path)

        # 提取波长。
        wave_val = self._extract_wave_number(cleaned_metadata)

        # 单文件默认视为 in situ 数据。
        technique = list(self.technique) if self.technique != ["unknown"] else self.DEFAULT_TECHNIQUE_SINGLE

        raw_info = Metadata(
            sample_name=self.sample_name or cleaned_metadata.get("sample_name", path.stem),
            start_time=self.start_time or cleaned_metadata.get("start_time"),
            technique=technique,
            instrument=self.instrument,
            operator=self.operator or cleaned_metadata.get("operator"),
            active_material_mass=self.active_material_mass or cleaned_metadata.get("active_material_mass"),
            wave_number=wave_val,
            raw_metadata=cleaned_metadata,
        )

        return DataBundle(data=ds, meta=raw_info)

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

    def _load_directory(self, path: Path, **_kwargs: Any) -> DataBundle:
        """Load all MSPD .xye files in a directory into a DataTree.

        Args:
            path: Path to the directory
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (DataBundle with DataTree, merged Metadata)
        """
        xye_files = sorted(path.rglob("*.xye"))
        if not xye_files:
            raise FileNotFoundError(f"No .xye files found in {path}")

        # Group files by parent directory
        groups: dict[Path, list[Path]] = {}
        for f in xye_files:
            groups.setdefault(f.parent, []).append(f)

        tree_dict: dict[str, xr.Dataset] = {}
        all_infos: list[Metadata] = []

        for parent, files in groups.items():
            try:
                group_result = self._build_directory_node(parent, files, path)
            except Exception as e:
                logger.error("加载或合并 %s 中的文件失败: %s", parent, e)
                continue

            if group_result is None:
                continue

            node_path, merged_ds, node_info = group_result
            tree_dict[node_path] = merged_ds
            merged_ds.attrs.update(node_info.to_dict())
            all_infos.append(node_info)

        if not tree_dict:
            raise RuntimeError(f"Failed to load any valid .xye files from {path}")

        # Create DataTree from dictionary
        tree = xr.DataTree.from_dict(tree_dict, name=path.name)

        # Merge all infos for the root
        root_info = self._merge_metadata(all_infos, path)
        tree.attrs.update(root_info.to_dict())

        return DataBundle(data=tree, meta=root_info)

    def _build_directory_node(self, parent: Path, files: list[Path], root: Path) -> tuple[str, xr.Dataset, Metadata] | None:
        """读取一个目录分组并返回 DataTree 节点数据。"""
        merged_ds, node_infos = self._process_directory_group(files)
        if merged_ds is None:
            return None

        node_path = "/" if parent == root else "/" + "/".join(parent.relative_to(root).parts)
        node_info = self._merge_node_infos(node_infos, parent)
        return node_path, merged_ds, node_info

    def _process_directory_group(self, files: list[Path]) -> tuple[xr.Dataset | None, list[Metadata]]:
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
                bundle = self._load_single_file(f)
                datasets.append(bundle.data)
                infos.append(bundle.meta)
            except Exception as e:
                logger.warning("跳过文件 %s，原因: %s", f, e)

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

    def _merge_node_infos(self, infos: list[Metadata], parent_path: Path) -> Metadata:
        """Merge Metadata objects for a single directory node.

        Args:
            infos: List of Metadata objects from files in the same directory
            parent_path: Parent directory path

        Returns:
            Merged Metadata
        """
        if not infos:
            return Metadata()

        # Collect sample names (one per file)
        sample_names = [info.sample_name for info in infos if info.sample_name]

        # Collect unique operators and start times
        operators = sorted({info.operator for info in infos if info.operator})
        start_times = sorted({info.start_time for info in infos if info.start_time})
        masses = sorted({info.active_material_mass for info in infos if info.active_material_mass})
        wave_numbers = sorted({info.wave_number for info in infos if info.wave_number})

        # Use directory-specific technique
        technique = list(self.technique) if self.technique != ["unknown"] else self.DEFAULT_TECHNIQUE_DIR

        # 构建节点级 raw_metadata。
        raw_metadata = {
            "n_files": len(infos),
            "sample_names": sample_names,
            "filenames": [Path(info.get("file_path", "")).name for info in infos if info.get("file_path")],
        }

        # 多个不同值以列表形式保存。
        if len(operators) > 1:
            raw_metadata["all_operators"] = operators
        if len(masses) > 1:
            raw_metadata["all_active_material_masses"] = masses
        if len(wave_numbers) > 1:
            raw_metadata["all_wave_numbers"] = wave_numbers

        return Metadata(
            sample_name=self.sample_name or parent_path.name,
            technique=technique,
            instrument=self.instrument,
            operator=self.operator or (operators[0] if len(operators) == 1 else None),
            start_time=self.start_time or (start_times[0] if len(start_times) == 1 else None),
            active_material_mass=self.active_material_mass or (masses[0] if len(masses) == 1 else None),
            wave_number=self.wave_number or (wave_numbers[0] if len(wave_numbers) == 1 else None),
            raw_metadata=raw_metadata,
        )

    def _merge_metadata(self, infos: list[Metadata], root_path: Path) -> Metadata:
        """Merge multiple Metadata objects from different directories.

        Args:
            infos: List of Metadata objects from subdirectories
            root_path: Root path for determining folder name

        Returns:
            Merged Metadata
        """
        if not infos:
            return Metadata()

        # Collect all sample names from all subdirectories
        all_sample_names = []
        for info in infos:
            # 每个节点的 raw_metadata 中可能包含多个 sample_names。
            if "sample_names" in info.raw_metadata:
                all_sample_names.extend(info.raw_metadata["sample_names"])
            elif info.sample_name:
                all_sample_names.append(info.sample_name)

        # Use directory-specific technique
        technique = list(self.technique) if self.technique != ["unknown"] else self.DEFAULT_TECHNIQUE_DIR

        # Calculate total files
        total_files = sum(info.raw_metadata.get("n_files", 1) for info in infos)

        # Use merge_metadata from utils for common logic
        merged_info = merge_metadata(
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
        merged_info.raw_metadata["sample_names"] = all_sample_names
        merged_info.raw_metadata["n_files"] = total_files

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
            logger.error("读取 %s 出错: %s", filepath, e)
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
            metadata: 文件中的元数据字典

        Returns:
            埃为单位的波长；无法获取时返回 None
        """
        if self.wave_number:
            try:
                return float(self.wave_number)
            except ValueError:
                logger.warning("无效 wave_number 配置: %s", self.wave_number)
        return metadata.get("Wave")

    @staticmethod
    def calculate_d_spacing(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
        """根据 2theta 和波长使用布拉格定律计算 d-spacing。

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
