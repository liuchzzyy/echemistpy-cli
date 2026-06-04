# -*- coding: utf-8 -*-
"""MSPD .xye 格式原位/operando XRD 数据读取器。"""

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

    支持读取单个 .xye 文件，也支持读取目录中的 .xye 序列并按父目录合并为
    operando XRD 数据。目录读取只处理 .xye，不读取同目录中的 .dat 原始日志或
    电化学文件。
    """

    DATE_FORMAT: ClassVar[str] = "%Y-%m-%d_%H:%M:%S"
    DATE_FORMATS: ClassVar[tuple[str, ...]] = (
        DATE_FORMAT,
        "%Y-%m-%d %H:%M:%S",
        "%a %b %d %H:%M:%S %Y",
    )
    DEFAULT_TECHNIQUE_SINGLE: ClassVar[list[str]] = ["xrd", "in_situ"]
    DEFAULT_TECHNIQUE_DIR: ClassVar[list[str]] = ["xrd", "operando"]
    INSTRUMENT_NAME: ClassVar[str] = "ALBA_MSPD"
    COMMENT_CHARS: ClassVar[tuple[str, ...]] = ("#", "!", ";", "'")
    FLOAT_PATTERN: ClassVar[str] = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
    DATA_COLUMNS: ClassVar[tuple[str, ...]] = ("2theta", "intensity", "intensity_error")

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
        """初始化 MSPD reader。"""
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> DataBundle:
        """加载单个 MSPD .xye 文件。"""
        if path.suffix.lower() != ".xye":
            raise ValueError(f"MSPDReader 只支持 .xye 文件: {path}")

        ds, metadata = self._read_single_xye(path)
        metadata["file_path"] = str(path)
        cleaned_metadata = self._clean_metadata(metadata, path)
        wave_val = self._extract_wave_number(cleaned_metadata)
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

        ds.attrs.update({"source_file": path.name, "instrument": self.instrument})
        return DataBundle(
            data=ds,
            meta=raw_info,
            provenance={"source_path": str(path), "reader": self.__class__.__name__},
        )

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any], filepath: Path) -> dict[str, Any]:
        """清理并标准化 MSPD .xye 头部元数据。"""
        cleaned = dict(metadata)

        date_value = metadata.get("Date") or metadata.get("date")
        if date_value:
            try:
                dt = MSPDReader.parse_date(str(date_value))
                cleaned["start_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                cleaned["start_time"] = str(date_value)

        wavelength = metadata.get("Wave") or metadata.get("wavelength")
        if wavelength is not None:
            cleaned["wavelength"] = float(wavelength)
            cleaned["wave_number"] = str(wavelength)

        numeric_aliases = {
            "Dt": "exposure_time_s",
            "Bin": "bin_width_deg",
            "imon_mean": "monitor_mean",
            "imon_std": "monitor_std",
            "imon_min": "monitor_min",
            "imon_max": "monitor_max",
        }
        for source_key, target_key in numeric_aliases.items():
            if source_key in metadata:
                cleaned[target_key] = metadata[source_key]

        scan_values = metadata.get("scan_values")
        if isinstance(scan_values, dict):
            for key in ("icurr", "imon", "i15", "i7", "mocoIn", "mocoOut"):
                if key in scan_values:
                    cleaned[key] = scan_values[key]

        cleaned["sample_name"] = filepath.stem
        cleaned["file_path"] = str(filepath)
        return cleaned

    def _extract_wave_number(self, metadata: dict[str, Any]) -> str | None:
        """从元数据或 reader 覆盖项中提取 X 射线波长。"""
        if self.wave_number:
            return self.wave_number
        wavelength = metadata.get("wavelength")
        return str(wavelength) if wavelength is not None else None

    def _load_directory(self, path: Path, **_kwargs: Any) -> DataBundle:
        """加载目录中所有 .xye 文件，并按父目录合并为 DataTree。"""
        xye_files = self._find_xye_files(path)
        if not xye_files:
            raise FileNotFoundError(f"在 {path} 中未找到 .xye 文件。")

        groups: dict[Path, list[Path]] = {}
        for file_path in xye_files:
            groups.setdefault(file_path.parent, []).append(file_path)

        tree_dict: dict[str, xr.Dataset] = {}
        all_infos: list[Metadata] = []

        for parent in sorted(groups, key=lambda item: item.relative_to(path).parts if item != path else ()):
            try:
                group_result = self._build_directory_node(parent, groups[parent], path)
            except Exception as exc:
                logger.error("加载或合并 %s 中的 .xye 文件失败: %s", parent, exc)
                continue

            if group_result is None:
                continue

            node_path, merged_ds, node_info = group_result
            tree_dict[node_path] = merged_ds
            merged_ds.attrs.update(node_info.to_dict())
            all_infos.append(node_info)

        if not tree_dict:
            raise RuntimeError(f"未能从 {path} 加载任何有效 .xye 文件。")

        tree = xr.DataTree.from_dict(tree_dict, name=path.name)
        root_info = self._merge_metadata(all_infos, path)
        tree.attrs.update(root_info.to_dict())

        return DataBundle(
            data=tree,
            meta=root_info,
            provenance={"source_path": str(path), "reader": self.__class__.__name__},
        )

    @classmethod
    def _find_xye_files(cls, path: Path) -> list[Path]:
        """查找目录中的 .xye 文件，大小写不敏感。"""
        return sorted((file_path for file_path in path.rglob("*") if file_path.is_file() and file_path.suffix.lower() == ".xye"), key=cls._file_sort_key)

    def _build_directory_node(self, parent: Path, files: list[Path], root: Path) -> tuple[str, xr.Dataset, Metadata] | None:
        """读取一个目录分组并返回 DataTree 节点数据。"""
        merged_ds, node_infos = self._process_directory_group(sorted(files, key=self._file_sort_key))
        if merged_ds is None:
            return None

        node_path = "/" if parent == root else "/" + "/".join(parent.relative_to(root).parts)
        node_info = self._merge_node_infos(node_infos, parent)
        return node_path, merged_ds, node_info

    def _process_directory_group(self, files: list[Path]) -> tuple[xr.Dataset | None, list[Metadata]]:
        """读取并合并同一目录下的一组 .xye 谱线。"""
        datasets: list[xr.Dataset] = []
        infos: list[Metadata] = []

        for file_path in files:
            try:
                bundle = self._load_single_file(file_path)
            except Exception as exc:
                logger.warning("跳过 .xye 文件 %s，原因: %s", file_path, exc)
                continue

            if not isinstance(bundle.data, xr.Dataset):
                raise ValueError(f"单个 .xye 文件应读取为 xarray.Dataset: {file_path}")
            datasets.append(bundle.data)
            infos.append(bundle.meta)

        if not datasets:
            return None, []

        merged_ds = self._concat_operando_datasets(datasets)
        merged_ds = self._assign_record_coordinates(merged_ds, infos)
        return merged_ds, infos

    @staticmethod
    def _concat_operando_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
        """沿 record 维度合并谱线；优先要求 2theta 网格完全一致。"""
        record_index = pd.Index(np.arange(len(datasets)), name="record")
        try:
            return xr.concat(datasets, dim=record_index, join="exact", combine_attrs="drop_conflicts")
        except ValueError:
            logger.warning("部分 .xye 文件的 2theta 网格不完全一致，使用 outer join 合并。")
            return xr.concat(datasets, dim=record_index, join="outer", combine_attrs="drop_conflicts")

    def _assign_record_coordinates(self, ds: xr.Dataset, infos: list[Metadata]) -> xr.Dataset:
        """为 operando 谱序列添加每条谱对应的时间和采集条件坐标。"""
        raw_items = [info.raw_metadata for info in infos]
        systimes = pd.to_datetime([info.start_time for info in infos], errors="coerce")
        coords: dict[str, Any] = {
            "record": np.arange(len(infos)),
            "filename": ("record", [Path(item.get("file_path", "")).name for item in raw_items]),
            "systime": ("record", systimes),
            "time_s": ("record", self._relative_seconds(systimes)),
        }

        numeric_coords = {
            "wavelength_angstrom": "wavelength",
            "exposure_time_s": "exposure_time_s",
            "bin_width_deg": "bin_width_deg",
            "monitor_mean": "monitor_mean",
            "monitor_std": "monitor_std",
            "monitor_min": "monitor_min",
            "monitor_max": "monitor_max",
            "icurr": "icurr",
            "imon": "imon",
        }
        for coord_name, metadata_key in numeric_coords.items():
            values = self._numeric_metadata_array(raw_items, metadata_key)
            if values is not None:
                coords[coord_name] = ("record", values)

        ds = ds.assign_coords(**coords)
        self._apply_record_coordinate_attrs(ds)
        return ds

    @staticmethod
    def _apply_record_coordinate_attrs(ds: xr.Dataset) -> None:
        """写入 operando record 坐标属性。"""
        attr_map = {
            "time_s": {"units": "s", "long_name": "Relative Time"},
            "systime": {"long_name": "System Time"},
            "wavelength_angstrom": {"units": "angstrom", "long_name": "Wavelength"},
            "exposure_time_s": {"units": "s", "long_name": "Exposure Time"},
            "bin_width_deg": {"units": "degree", "long_name": "2theta Bin Width"},
            "monitor_mean": {"units": "counts", "long_name": "Incident Monitor Mean"},
            "monitor_std": {"units": "counts", "long_name": "Incident Monitor Std"},
            "monitor_min": {"units": "counts", "long_name": "Incident Monitor Min"},
            "monitor_max": {"units": "counts", "long_name": "Incident Monitor Max"},
            "icurr": {"units": "mA", "long_name": "Storage Ring Current"},
            "imon": {"units": "counts", "long_name": "Incident Monitor"},
        }
        for coord_name, attrs in attr_map.items():
            if coord_name in ds.coords:
                ds.coords[coord_name].attrs.update(attrs)

    @staticmethod
    def _numeric_metadata_array(raw_items: list[dict[str, Any]], key: str) -> np.ndarray | None:
        """从文件级元数据中提取数值坐标数组；全缺失时返回 None。"""
        values: list[float] = []
        has_value = False

        for item in raw_items:
            value = item.get(key)
            if value is None:
                values.append(np.nan)
                continue
            try:
                values.append(float(value))
                has_value = True
            except (TypeError, ValueError):
                values.append(np.nan)

        return np.asarray(values, dtype=float) if has_value else None

    @staticmethod
    def _relative_seconds(systimes: pd.DatetimeIndex) -> np.ndarray:
        """将绝对时间转换为相对首个有效时间的秒数。"""
        anchor = next((timestamp for timestamp in systimes if not pd.isna(timestamp)), None)
        if anchor is None:
            return np.full(len(systimes), np.nan, dtype=float)
        return (systimes - anchor).total_seconds().to_numpy(dtype=float)

    def _merge_node_infos(self, infos: list[Metadata], parent_path: Path) -> Metadata:
        """合并单个目录节点的 Metadata。"""
        if not infos:
            return Metadata()

        sample_names = [info.sample_name for info in infos if info.sample_name]
        operators = sorted({info.operator for info in infos if info.operator})
        start_times = sorted({info.start_time for info in infos if info.start_time})
        masses = sorted({info.active_material_mass for info in infos if info.active_material_mass})
        wave_numbers = sorted({info.wave_number for info in infos if info.wave_number})
        technique = list(self.technique) if self.technique != ["unknown"] else self.DEFAULT_TECHNIQUE_DIR

        raw_metadata: dict[str, Any] = {
            "n_files": len(infos),
            "sample_names": sample_names,
            "filenames": [Path(info.get("file_path", "")).name for info in infos if info.get("file_path")],
        }
        if start_times:
            raw_metadata["start_time"] = start_times[0]
            raw_metadata["end_time"] = start_times[-1]
            raw_metadata["duration_s"] = self._duration_seconds(start_times)
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
            start_time=self.start_time or (start_times[0] if start_times else None),
            active_material_mass=self.active_material_mass or (masses[0] if len(masses) == 1 else None),
            wave_number=self.wave_number or (wave_numbers[0] if len(wave_numbers) == 1 else None),
            raw_metadata=raw_metadata,
        )

    def _merge_metadata(self, infos: list[Metadata], root_path: Path) -> Metadata:
        """合并多个目录节点的 Metadata。"""
        if not infos:
            return Metadata()

        all_sample_names: list[str] = []
        for info in infos:
            node_sample_names = info.raw_metadata.get("sample_names")
            if isinstance(node_sample_names, list):
                all_sample_names.extend(str(name) for name in node_sample_names)
            elif info.sample_name:
                all_sample_names.append(info.sample_name)

        technique = list(self.technique) if self.technique != ["unknown"] else self.DEFAULT_TECHNIQUE_DIR
        total_files = sum(int(info.raw_metadata.get("n_files", 1)) for info in infos)
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

        start_times = sorted({info.start_time for info in infos if info.start_time})
        if start_times and not self.start_time:
            merged_info.start_time = start_times[0]
            merged_info.raw_metadata["start_time"] = start_times[0]
            merged_info.raw_metadata["end_time"] = start_times[-1]
            merged_info.raw_metadata["duration_s"] = self._duration_seconds(start_times)

        merged_info.raw_metadata["sample_names"] = all_sample_names
        merged_info.raw_metadata["n_files"] = total_files
        return merged_info

    def _read_single_xye(self, filepath: Path) -> tuple[xr.Dataset, dict[str, Any]]:
        """读取单个 .xye 文件并返回 Dataset 和头部元数据。"""
        header_lines, skiprows = self._collect_header_lines(filepath)
        metadata = self._parse_header_lines(header_lines)
        metadata["header_line_count"] = skiprows

        df = self._read_xye_dataframe(filepath, skiprows)
        expected_points = metadata.get("n_points")
        if isinstance(expected_points, int) and expected_points != len(df):
            logger.warning("文件 %s 头部声明 %s 个点，实际读取 %s 个点。", filepath, expected_points, len(df))

        return self._create_dataset(df, metadata), metadata

    @classmethod
    def _collect_header_lines(cls, filepath: Path) -> tuple[list[str], int]:
        """收集 .xye 头部行，并返回数据区起始行号。"""
        header_lines: list[str] = []
        skiprows = 0

        with filepath.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                stripped = line.strip()
                if cls._is_header_line(stripped):
                    if stripped:
                        header_lines.append(stripped)
                    skiprows += 1
                    continue
                break

        return header_lines, skiprows

    @classmethod
    def _is_header_line(cls, stripped_line: str) -> bool:
        """判断一行是否属于头部而不是数值数据区。"""
        if not stripped_line:
            return True
        if stripped_line[0] in cls.COMMENT_CHARS:
            return True
        first_token = stripped_line.replace(",", " ").split()[0]
        return not cls._is_numeric_token(first_token)

    @classmethod
    def _is_numeric_token(cls, token: str) -> bool:
        """判断字符串 token 是否为数值。"""
        return bool(re.fullmatch(cls.FLOAT_PATTERN, token))

    @classmethod
    def _read_xye_dataframe(cls, filepath: Path, skiprows: int) -> pd.DataFrame:
        """读取 .xye 数据区的 2theta、intensity 和 error 列。"""
        try:
            df = pd.read_csv(
                filepath,
                skiprows=skiprows,
                comment="#",
                sep=r"\s+|,",
                header=None,
                engine="python",
                dtype=float,
            )
        except Exception as exc:
            logger.error("读取 .xye 数据区失败 %s: %s", filepath, exc)
            raise

        df = df.dropna(how="all")
        if df.shape[1] < 2:
            raise ValueError(f".xye 文件至少需要 2 列数值数据: {filepath}")

        df = df.iloc[:, : min(3, df.shape[1])].copy()
        df.columns = list(cls.DATA_COLUMNS[: df.shape[1]])
        df = df.dropna(subset=["2theta", "intensity"])
        if df.empty:
            raise ValueError(f".xye 文件没有可读取的数值数据: {filepath}")
        return df

    def _create_dataset(self, df: pd.DataFrame, metadata: dict[str, Any]) -> xr.Dataset:
        """从 .xye 数值表构造 XRD Dataset。"""
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
            "intensity": (("2theta",), df["intensity"].to_numpy(dtype=float)),
        }
        if "intensity_error" in df.columns and not df["intensity_error"].isna().all():
            data_vars["intensity_error"] = (("2theta",), df["intensity_error"].to_numpy(dtype=float))

        ds = xr.Dataset(data_vars, coords={"2theta": df["2theta"].to_numpy(dtype=float)})
        apply_standard_attrs_xrd(ds)

        wave_to_use = self._get_wave_to_use(metadata)
        if wave_to_use is not None and wave_to_use > 0:
            d_spacing = self.calculate_d_spacing(ds["2theta"].values, wave_to_use)
            ds = ds.assign_coords(d_spacing=(("2theta",), d_spacing))
            ds.d_spacing.attrs.update({"units": "Å", "long_name": "d-spacing"})
            ds.attrs["wavelength_angstrom"] = float(wave_to_use)

        return ds

    @classmethod
    def _parse_header_lines(cls, header_lines: list[str]) -> dict[str, Any]:
        """解析 .xye 头部元数据。"""
        cleaned_lines = [cls._strip_header_marker(line) for line in header_lines]
        metadata: dict[str, Any] = {"raw_header": cleaned_lines}

        for index, line in enumerate(cleaned_lines):
            cls._parse_processing_line(index, line, metadata)
            cls._parse_core_header_values(line, metadata)
            cls._parse_monitor_statistics(line, metadata)
            cls._parse_monitor_line(line, metadata)
            cls._parse_expected_points(line, metadata)

        return metadata

    @classmethod
    def _parse_processing_line(cls, index: int, line: str, metadata: dict[str, Any]) -> None:
        """解析第一行中的处理命令和源文件路径。"""
        if index != 0 or ":" not in line:
            return
        processed_file, command = line.split(":", 1)
        if processed_file.strip():
            metadata["processed_file"] = processed_file.strip()
        if command.strip():
            metadata["processing_command"] = command.strip()
        input_match = re.search(r"(\S+\.dat)\b", command)
        if input_match:
            metadata["input_data_file"] = input_match.group(1)

    @classmethod
    def _parse_core_header_values(cls, line: str, metadata: dict[str, Any]) -> None:
        """解析波长、日期、曝光时间、bin 宽度等核心头部字段。"""
        for label in ("Wave", "Dt", "Bin"):
            value = cls._number_after_equals(line, label)
            if value is not None:
                metadata[label] = value

        date_match = re.search(r"\bDate\s*=\s*(\S+)", line)
        if date_match:
            metadata["Date"] = date_match.group(1)

        cal_match = re.search(r"\bCalPath\s+(\S+)", line)
        if cal_match:
            metadata["CalPath"] = cal_match.group(1)

        list_fields = {"IsMon": "IsMon", "IsPos": "IsPos"}
        for label, key in list_fields.items():
            list_match = re.search(rf"\b{label}\s*=\s*\[([^\]]+)\]", line)
            if list_match:
                metadata[key] = cls._parse_number_list(list_match.group(1))

    @classmethod
    def _parse_monitor_statistics(cls, line: str, metadata: dict[str, Any]) -> None:
        """解析 MSPD 头部的 <imon> 均值、标准差和范围。"""
        pattern = rf"<imon>\s*=\s*({cls.FLOAT_PATTERN})(?:\s*\+/-\s*({cls.FLOAT_PATTERN}))?(?:.*?\(Min/Max\)\s*({cls.FLOAT_PATTERN})\s*/\s*({cls.FLOAT_PATTERN}))?"
        match = re.search(pattern, line)
        if not match:
            return

        keys = ("imon_mean", "imon_std", "imon_min", "imon_max")
        for key, value in zip(keys, match.groups(), strict=True):
            if value is not None:
                metadata[key] = float(value)

    @classmethod
    def _parse_monitor_line(cls, line: str, metadata: dict[str, Any]) -> None:
        """解析不带等号的 MSPD monitor 键值行。"""
        if "=" in line:
            return

        values = cls._numeric_key_value_pairs(line)
        if values:
            metadata.setdefault("scan_values", {}).update(values)

    @classmethod
    def _parse_expected_points(cls, line: str, metadata: dict[str, Any]) -> None:
        """解析仅包含点数的头部行。"""
        token = line.strip()
        if token.isdigit():
            metadata["n_points"] = int(token)

    @classmethod
    def _number_after_equals(cls, line: str, label: str) -> float | None:
        """读取形如 ``Label = number`` 的字段。"""
        match = re.search(rf"\b{re.escape(label)}\s*=\s*({cls.FLOAT_PATTERN})", line)
        return float(match.group(1)) if match else None

    @classmethod
    def _numeric_key_value_pairs(cls, line: str) -> dict[str, float]:
        """读取形如 ``key value key value`` 的数值键值对。"""
        pairs: dict[str, float] = {}
        pattern = rf"(?<![/\w])([A-Za-z_][A-Za-z0-9_<>]*)\s+({cls.FLOAT_PATTERN})(?=\s|$)"
        for match in re.finditer(pattern, line):
            pairs[match.group(1)] = float(match.group(2))
        return pairs

    @classmethod
    def _parse_number_list(cls, text: str) -> list[float]:
        """解析 MSPD 头部方括号中的数值列表。"""
        return [float(value) for value in re.findall(cls.FLOAT_PATTERN, text)]

    @classmethod
    def _strip_header_marker(cls, line: str) -> str:
        """去除头部注释符。"""
        stripped = line.strip()
        return stripped[1:].strip() if stripped and stripped[0] in cls.COMMENT_CHARS else stripped

    def _get_wave_to_use(self, metadata: dict[str, Any]) -> float | None:
        """确定用于 d-spacing 计算的波长，单位为 Å。"""
        if self.wave_number:
            try:
                return float(self.wave_number)
            except ValueError:
                logger.warning("无效 wave_number 配置: %s", self.wave_number)

        for key in ("wavelength", "Wave", "wave_number"):
            value = metadata.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning("无效波长元数据 %s=%s", key, value)
        return None

    @staticmethod
    def calculate_d_spacing(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
        """根据 2theta 和波长使用布拉格定律计算 d-spacing。"""
        theta_rad = np.deg2rad(two_theta / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return wavelength / (2.0 * np.sin(theta_rad))

    @classmethod
    def parse_date(cls, date_str: str) -> datetime:
        """解析 MSPD 常见日期格式。"""
        if not date_str:
            raise ValueError("日期字符串为空。")

        for date_format in cls.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, date_format)
            except ValueError:
                continue
        raise ValueError(f"无法解析 MSPD 日期: {date_str}")

    @staticmethod
    def _duration_seconds(time_strings: list[str]) -> float | None:
        """根据一组时间字符串计算起止时间差。"""
        systimes = pd.to_datetime(time_strings, errors="coerce")
        valid_times = systimes[~pd.isna(systimes)]
        if len(valid_times) < 2:
            return None
        return float((valid_times[-1] - valid_times[0]).total_seconds())

    @staticmethod
    def _file_sort_key(path: Path) -> tuple[tuple[int, int | str], ...]:
        """按文件名中的数字自然排序。"""
        parts = re.split(r"(\d+)", path.name.lower())
        return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)

    @staticmethod
    def _get_file_extension() -> str:
        """返回默认文件扩展名。"""
        return ".xye"
