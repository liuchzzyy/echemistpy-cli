# -*- coding: utf-8 -*-
"""Bio-Logic MPT 文件读取器。

主要类：
- BiologicMPTReader：读取 MPT 文件并提取元数据

MPT 列类型识别和文本头解析实现参考并受启发于 galvani 项目的 ``galvani/BioLogic.py``：
https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py
当前实现为 echemistpy 内置解析器，不依赖 galvani 包。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.data.models import DataBundle, Metadata
from echemistpy.io.base_reader import BaseReader
from echemistpy.io.contracts import ReaderSpec

logger = logging.getLogger(__name__)

# --- 列解析相关常量 ---

UNKNOWN_COLUMN_TYPE_HIERARCHY = ("<f8", "<f4", "<u4", "<u2", "<u1")

BOOL_COLUMNS = {
    "ox/red",
    "error",
    "control changes",
    "Ns changes",
    "counter inc.",
}

INT_COLUMNS = {"cycle number", "I Range", "Ns", "half cycle", "z cycle"}

FLOAT_COLUMNS = {
    "time/s",
    "P/W",
    "(Q-Qo)/mA.h",
    "x",
    "control/V",
    "control/mA",
    "control/V/mA",
    "(Q-Qo)/C",
    "dQ/C",
    "freq/Hz",
    "|Ewe|/V",
    "|I|/A",
    "Phase(Z)/deg",
    "|Z|/Ohm",
    "Re(Z)/Ohm",
    "-Im(Z)/Ohm",
    "Re(M)",
    "Im(M)",
    "|M|",
    "Re(Permittivity)",
    "Im(Permittivity)",
    "|Permittivity|",
    "Tan(Delta)",
    "Q charge/discharge/mA.h",
    "step time/s",
    "Q charge/mA.h",
    "Q discharge/mA.h",
    "Temperature/°C",
    "Efficiency/%",
    "Capacity/mA.h",
}

FLOAT_SUFFIXES = (
    "/s",
    "/Hz",
    "/deg",
    "/W",
    "/mW",
    "/W.h",
    "/mW.h",
    "/A",
    "/mA",
    "/A.h",
    "/mA.h",
    "/V",
    "/mV",
    "/F",
    "/mF",
    "/uF",
    "/µF",
    "/nF",
    "/C",
    "/Ohm",
    "/Ohm-1",
    "/Ohm.cm",
    "/mS/cm",
    "/%",
)

SPECIAL_MAPPINGS = {
    "dq/mA.h": ("dQ/mA.h", np.float64),
    "dQ/mA.h": ("dQ/mA.h", np.float64),
    "I/mA": ("I/mA", np.float64),
    "<I>/mA": ("I/mA", np.float64),
    "Ewe/V": ("Ewe/V", np.float64),
    "<Ewe>/V": ("Ewe/V", np.float64),
    "Ecell/V": ("Ewe/V", np.float64),
    "<Ewe/V>": ("Ewe/V", np.float64),
}


def _get_dtype_from_column_type(fieldname: str) -> Any:
    """根据列分类获取 dtype。

    Args:
        fieldname: 列名

    Returns:
        numpy dtype 或 None
    """
    if fieldname in BOOL_COLUMNS:
        return np.bool_
    if fieldname in INT_COLUMNS:
        return np.int_
    if fieldname in FLOAT_COLUMNS:
        return np.float64
    if fieldname.endswith(FLOAT_SUFFIXES) or fieldname.startswith("empty_column_"):
        return np.float64
    return None


def fieldname_to_dtype(fieldname: str) -> tuple[str, Any]:
    """将 MPT 列头转换为 (name, dtype) 元组。

    Args:
        fieldname: MPT 文件列头

    Returns:
        (name, dtype) 元组

    Raises:
        ValueError: 列头无效
    """
    if fieldname == "mode":
        return ("mode", np.uint8)

    if fieldname in SPECIAL_MAPPINGS:
        return SPECIAL_MAPPINGS[fieldname]

    dtype = _get_dtype_from_column_type(fieldname)
    if dtype is not None:
        return (fieldname, dtype)

    raise ValueError(f"无效列头: {fieldname}")


def _calculate_systime(acq_start: str, relative_times: np.ndarray) -> pd.Series:
    """根据采集开始时间和相对时间计算绝对系统时间。

    Args:
        acq_start: 采集开始时间字符串
        relative_times: 秒为单位的相对时间数组

    Returns:
        datetime 对象序列
    """
    try:
        # BioLogic 格式：MM/DD/YYYY HH:MM:SS.ffffff
        start_dt = datetime.strptime(acq_start, "%m/%d/%Y %H:%M:%S.%f")
        start_ts = start_dt.timestamp()
        return pd.Series(pd.to_datetime(start_ts + relative_times, unit="s"))
    except Exception as e:
        logger.debug("解析采集开始时间 '%s' 失败: %s", acq_start, e)
        return pd.Series(relative_times)


def _read_mpt_content(mpt_file: Any, encoding: str = "latin1") -> tuple[np.ndarray, list[bytes]]:
    """从文件对象读取 MPT 内容。

    Args:
        mpt_file: 待读取的文件对象
        encoding: 文件编码

    Returns:
        (numpy 数组, 注释列表) 元组

    Raises:
        ValueError: 文件格式无效
    """
    magic = next(mpt_file).strip()
    if magic not in {b"EC-Lab ASCII FILE", b"BT-Lab ASCII FILE"}:
        raise ValueError(f"文件首行无效: {magic!r}")

    nb_headers_match = re.match(rb"Nb header lines : (\d+)\s*$", next(mpt_file))
    if not nb_headers_match:
        raise ValueError("头部行格式无效。")
    nb_headers = int(nb_headers_match.group(1))
    if nb_headers < 3:
        raise ValueError(f"头部行过少: {nb_headers}")

    comments = [next(mpt_file) for _ in range(nb_headers - 3)]

    fieldnames_raw = next(mpt_file).decode(encoding).strip()
    fieldnames = fieldnames_raw.split("\t")

    current_pos = mpt_file.tell()
    first_data_line = next(mpt_file).decode(encoding).strip()
    mpt_file.seek(current_pos)
    data_column_count = len(first_data_line.split("\t"))

    if len(fieldnames) > data_column_count:
        fieldnames = fieldnames[:data_column_count]

    for i, fn in enumerate(fieldnames):
        if not fn or not fn.strip():
            fieldnames[i] = f"empty_column_{i}"

    dtype_list = []
    for fn in fieldnames:
        if fn == "time/s":
            dtype_list.append((fn, "U30"))
        else:
            dtype_list.append(fieldname_to_dtype(fn))
    record_type = np.dtype(dtype_list)

    def str_to_float(s: str) -> float:
        if not s:
            return np.nan
        return float(s.replace(",", "."))

    converter_dict = {}
    for i, fn in enumerate(fieldnames):
        if fn == "time/s":
            converter_dict[i] = lambda s: s
        else:
            converter_dict[i] = str_to_float

    mpt_array = np.loadtxt(mpt_file, dtype=record_type, converters=cast(Any, converter_dict), delimiter="\t")

    return mpt_array, comments


class BiologicMPTReader(BaseReader):
    """BioLogic MPT 文件读取器。"""

    # --- 解析常量 ---
    INSTRUMENT_NAME: ClassVar[str] = "BioLogic"
    DEFAULT_TECHNIQUE: ClassVar[list[str]] = ["echem"]
    MASS_REGEX: ClassVar[re.Pattern[str]] = re.compile(r"(\d+\.?\d*)\s*(mg|g)", re.IGNORECASE)

    # --- reader 能力声明 ---
    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "biologic"
    spec: ClassVar[ReaderSpec] = ReaderSpec(
        name="biologic_mpt",
        extensions=(".mpt",),
        instruments=("biologic",),
        techniques=("echem",),
        supports_directory=True,
        description="BioLogic EC-Lab ASCII MPT files",
    )

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """初始化 BioLogic reader。

        Args:
            filepath: MPT 文件或目录路径
            **kwargs: 额外元数据覆盖项
        """
        # 设置默认技术类型。
        if "technique" not in kwargs:
            kwargs["technique"] = self.DEFAULT_TECHNIQUE
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> DataBundle:
        """加载单个 BioLogic MPT 文件。

        Args:
            path: MPT 文件路径
            **_kwargs: 额外参数（未使用）

        Returns:
            DataBundle 数据包
        """
        with open(path, "rb") as f:
            mpt_array, comments = _read_mpt_content(f)

        # 解析元数据。
        file_info = self._parse_mpt_metadata(list(comments))
        metadata = {
            "file_info": file_info,
            "file_type": "MPT",
            "file_path": str(path),
        }
        cleaned_metadata = self._clean_metadata(metadata)

        # 解析活性物质质量。
        mass = self._extract_mass(cleaned_metadata)

        # 识别技术类型。
        tech_list = self._detect_techniques(cleaned_metadata, mpt_array)

        # 创建 Dataset。
        ds = self._create_dataset(mpt_array, cleaned_metadata, mass)

        metadata = Metadata(
            sample_name=self.sample_name or str(cleaned_metadata.get("sample_name", "Unknown")),
            start_time=self.start_time or cleaned_metadata.get("start_time"),
            operator=self.operator or cleaned_metadata.get("operator"),
            technique=self.technique if self.technique != self.DEFAULT_TECHNIQUE else tech_list,
            instrument=self.instrument,
            active_material_mass=self.active_material_mass or cleaned_metadata.get("active_material_mass"),
            wave_number=self.wave_number,
            raw_metadata=cleaned_metadata,
        )

        return DataBundle(data=ds, meta=metadata, provenance={"source_path": str(path), "reader": self.__class__.__name__})

    def _extract_mass(self, metadata: dict[str, Any]) -> float | None:
        """从元数据或 reader 配置中提取克为单位的质量。

        Args:
            metadata: 元数据字典

        Returns:
            克为单位的质量，无法解析时返回 None
        """
        mass_str = self.active_material_mass or metadata.get("active_material_mass")
        if not mass_str:
            return None

        match = self.MASS_REGEX.search(str(mass_str))
        if match:
            val, unit = float(match.group(1)), match.group(2).lower()
            return val * 0.001 if unit == "mg" else val
        return None

    def _create_dataset(self, mpt_array: np.ndarray, metadata: dict[str, Any], mass: float | None) -> xr.Dataset:
        """根据 MPT 数组创建标准化 xarray.Dataset。

        Args:
            mpt_array: MPT 文件读取出的 NumPy 数组
            metadata: 元数据字典
            mass: 克为单位的活性物质质量

        Returns:
            xarray Dataset
        """
        names = list(mpt_array.dtype.names or [])
        n_records = len(mpt_array)

        # 识别技术类型以确定列顺序。
        tech_str = metadata.get("file_info", {}).get("technique", "")
        is_peis = "Electrochemical Impedance" in tech_str or "PEIS" in tech_str or "freq/Hz" in names
        is_gpcl = "Galvanostatic Cycling" in tech_str or "GPCL" in tech_str
        is_ocv = "Open Circuit Voltage" in tech_str or "OCV" in tech_str

        if is_peis:
            ordered_cols = ["cycle number", "freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm", "|Z|/Ohm", "Phase(Z)/deg"]
        elif is_gpcl:
            ordered_cols = ["time/s", "systime", "cycle number", "Ewe/V", "Ece/V", "voltage/V", "SpeCap_cal/mAh/g", "I/mA", "Capacity/mA.h"]
        elif is_ocv:
            ordered_cols = ["time/s", "systime", "cycle number", "Ewe/V", "Ece/V", "voltage/V"]
        else:
            ordered_cols = names

        data_vars = {col: (["record"], mpt_array[col]) for col in ordered_cols if col in names}
        coords = {"record": np.arange(1, n_records + 1)}

        # 添加计算列。
        extra_vars, extra_coords = self._compute_extra_columns(mpt_array, metadata, mass)
        data_vars.update(extra_vars)
        coords.update(extra_coords)

        # 按列顺序构建最终 Dataset。
        ds = xr.Dataset({k: data_vars[k] for k in ordered_cols if k in data_vars}, coords=coords)
        self._apply_standard_attrs(ds)
        return ds

    @staticmethod
    def _compute_extra_columns(mpt_array: np.ndarray, metadata: dict, mass: float | None) -> tuple[dict[str, Any], dict[str, Any]]:
        """计算电压、系统时间和比容量等附加列。

        Args:
            mpt_array: MPT 文件读取出的 NumPy 数组
            metadata: 元数据字典
            mass: 克为单位的活性物质质量

        Returns:
            (附加变量, 附加坐标) 元组
        """
        extra_vars: dict[str, Any] = {}
        extra_coords: dict[str, Any] = {}
        names = mpt_array.dtype.names or []
        n_records = len(mpt_array)

        # 电压列：缺失 Ewe/V 或 Ece/V 时用 0 占位。
        if "Ewe/V" in names:
            ewe = mpt_array["Ewe/V"]
        else:
            ewe = np.zeros(n_records)
            extra_vars["Ewe/V"] = (["record"], ewe)

        if "Ece/V" in names:
            ece = mpt_array["Ece/V"]
        else:
            ece = np.zeros(n_records)
            extra_vars["Ece/V"] = (["record"], ece)

        extra_vars["voltage/V"] = (["record"], ewe - ece)

        # 时间列。
        acq_start = metadata.get("file_info", {}).get("Acquisition started on", "")
        if "time/s" in names:
            time_data = mpt_array["time/s"]
            if time_data.dtype.kind in {"S", "U", "O"}:
                # 字符串时间通常表示绝对日期。
                try:
                    systimes = pd.to_datetime(time_data)
                except Exception:
                    systimes = pd.to_datetime(time_data, errors="coerce")

                extra_coords["systime"] = (["record"], systimes)
                # 计算从起点开始的秒数。
                if not systimes.empty:
                    extra_coords["time_s"] = (["record"], (systimes - systimes[0]).total_seconds())
            elif acq_start:
                systimes = _calculate_systime(acq_start, time_data)
                extra_coords["systime"] = (["record"], systimes)
                extra_coords["time_s"] = (["record"], (systimes - systimes[0]).dt.total_seconds())

        # 比容量。
        if mass and "Capacity/mA.h" in names:
            extra_vars["SpeCap_cal/mAh/g"] = (["record"], mpt_array["Capacity/mA.h"] / mass)

        return extra_vars, extra_coords

    @staticmethod
    def _apply_standard_attrs(ds: xr.Dataset) -> None:
        """应用标准单位和长名称属性。

        Args:
            ds: 原地修改的 xarray Dataset
        """
        attr_map = {
            "time/s": {"units": "s", "long_name": "Time"},
            "Ewe/V": {"units": "V", "long_name": "Working Electrode Potential"},
            "Ece/V": {"units": "V", "long_name": "Counter Electrode Potential"},
            "I/mA": {"units": "mA", "long_name": "Current"},
            "voltage/V": {"units": "V", "long_name": "Cell Voltage"},
            "Capacity/mA.h": {"units": "mAh", "long_name": "Capacity"},
            "SpeCap_cal/mAh/g": {"units": "mAh/g", "long_name": "Specific Capacity"},
            "freq/Hz": {"units": "Hz", "long_name": "Frequency"},
            "Re(Z)/Ohm": {"units": "Ohm", "long_name": "Real Impedance"},
            "-Im(Z)/Ohm": {"units": "Ohm", "long_name": "Imaginary Impedance"},
        }
        for var, attrs in attr_map.items():
            if var in ds:
                ds[var].attrs.update(attrs)

    def _detect_techniques(self, cleaned_metadata: dict, mpt_array: np.ndarray) -> list[str]:
        """识别具体电化学技术类型。

        Args:
            cleaned_metadata: 清理后的元数据字典
            mpt_array: MPT 文件读取出的 NumPy 数组

        Returns:
            识别到的技术类型列表
        """
        tech_str = cleaned_metadata.get("file_info", {}).get("technique", "")
        names = mpt_array.dtype.names or []
        tech_list = list(self.technique)

        if "Electrochemical Impedance" in tech_str or "PEIS" in tech_str or "freq/Hz" in names:
            tech_list.append("peis")
        if "Galvanostatic Cycling" in tech_str or "GPCL" in tech_str:
            tech_list.append("gpcl")
        if "Open Circuit Voltage" in tech_str or "OCV" in tech_str:
            tech_list.append("ocv")

        return list(set(tech_list))

    @staticmethod
    def _parse_mpt_metadata(comments: list[bytes | str]) -> dict[str, Any]:
        """将 MPT 文件注释解析为结构化元数据。

        Args:
            comments: MPT 文件注释行列表

        Returns:
            解析后的元数据字典
        """
        meta: dict[str, Any] = {}
        state: dict[str, Any] = {"current_section": None, "in_parameters": False, "work_mode_list": []}

        for line in comments:
            text = line.decode("latin1") if isinstance(line, bytes) else line
            text = text.rstrip("\r\n")
            if not text.strip():
                state["in_parameters"] = False
                continue

            BiologicMPTReader._handle_mpt_line(text, meta, state)

        if state["work_mode_list"]:
            meta["work_mode"] = state["work_mode_list"]
        return meta

    @staticmethod
    def _handle_mpt_line(text: str, meta: dict, state: dict) -> None:
        """处理单行 MPT 元数据。

        Args:
            text: MPT 文件中的一行文本
            meta: 待更新的元数据字典
            state: 解析状态字典
        """
        indent = len(text) - len(text.lstrip())
        content = text.strip()

        def split_kv(c: str) -> tuple[str, str] | None:
            for sep in (" : ", ":"):
                if sep in c:
                    k, v = c.split(sep, 1)
                    return k.strip(), v.strip()
            return None

        def add_val(d: dict, key: str, val: Any) -> None:
            if key not in d:
                d[key] = val
            else:
                existing = d[key]
                if isinstance(existing, list):
                    existing.append(val)
                else:
                    d[key] = [existing, val]

        if indent > 0 and state["current_section"] is not None:
            kv = split_kv(content)
            if kv:
                add_val(state["current_section"], *kv)
        elif "technique" not in meta and any(kw in content.lower() for kw in ["electrochemical", "impedance", "spectroscopy", "potentio", "galvano", "open circuit", "ocv"]):
            meta["technique"] = content
        elif content.startswith("Cycle Definition"):
            state["in_parameters"] = True
            kv = split_kv(content)
            state["current_section"] = {"cycle_definition": kv[1]} if kv else {}
            state["work_mode_list"].append(state["current_section"])
        elif state["in_parameters"] and state["current_section"] is not None:
            match = re.match(r"(.+?)\s{2,}(.+)", text)
            if match:
                add_val(state["current_section"], match.group(1).strip(), match.group(2).strip())
            else:
                add_val(state["current_section"], content, "")
        else:
            kv = split_kv(content)
            if kv:
                k, v = kv
                if not v:
                    state["current_section"] = {}
                    meta[k] = state["current_section"]
                else:
                    add_val(meta, k, v)
                    state["current_section"] = None

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """清理元数据，只保留核心字段和原始 file_info。

        Args:
            metadata: 原始元数据字典

        Returns:
            清理后的元数据字典
        """
        cleaned: dict[str, Any] = {}
        file_info = metadata.get("file_info", {})

        test_keys = ["technique", "Electrode material", "Electrolyte", "Mass of active material", "Reference electrode", "Acquisition started on", "Operator"]
        test_info = {k: file_info[k] for k in test_keys if k in file_info}

        if "Saved on" in file_info:
            saved = file_info["Saved on"]
            if isinstance(saved, dict):
                if "File" in saved:
                    test_info["name"] = saved["File"]
                if "Directory" in saved:
                    test_info["file_path"] = saved["Directory"]

        if test_info:
            cleaned.update({
                "sample_name": test_info.get("name"),
                "start_time": test_info.get("Acquisition started on"),
                "operator": test_info.get("Operator"),
                "active_material_mass": test_info.get("Mass of active material"),
            })

        proc_keys = ["Run on channel", "Ewe Ctrl range", "Electrode surface area", "Characteristic mass"]
        proc_info = {k: file_info[k] for k in proc_keys if k in file_info}
        if "Characteristic mass" in proc_info:
            cleaned.setdefault("active_material_mass", proc_info["Characteristic mass"])

        return {**cleaned, **metadata}
