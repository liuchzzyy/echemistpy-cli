"""BioLogic MPR 二进制文件读取器。

MPR 二进制格式解析实现参考并受启发于 galvani 项目的 ``galvani/BioLogic.py``：
https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py
galvani 项目许可证为 GPL-3.0-or-later；本文件只保留 echemistpy 读取 BioLogic MPR 所需的最小格式解析逻辑。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, cast

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.data.models import DataBundle, Metadata
from echemistpy.io.base_reader import BaseReader
from echemistpy.io.contracts import ReaderSpec

logger = logging.getLogger(__name__)

MPR_MAGIC = b"BIO-LOGIC MODULAR FILE\x1a".ljust(48) + b"\x00\x00\x00\x00"
MODULE_MAGIC = b"MODULE"
OLE_BASE = datetime(1899, 12, 30)
OLE_TIMESTAMP_OFFSETS = (465, 469, 473, 585)

VMP_MODULE_HEADER_V1 = np.dtype(
    [
        ("shortname", "S10"),
        ("longname", "S25"),
        ("length", "<u4"),
        ("version", "<u4"),
        ("date", "S8"),
    ]
)

VMP_MODULE_HEADER_V2 = np.dtype(
    [
        ("shortname", "S10"),
        ("longname", "S25"),
        ("max_length", "<u4"),
        ("length", "<u4"),
        ("version", "<u4"),
        ("unknown2", "<u4"),
        ("date", "S8"),
    ]
)

# BioLogic data module column ID 到列名和二进制 dtype 的映射。
MPR_COLUMN_DTYPE_MAP: dict[int, tuple[str, str]] = {
    4: ("time/s", "<f8"),
    5: ("control/V/mA", "<f4"),
    6: ("Ewe/V", "<f4"),
    7: ("dq/mA.h", "<f8"),
    8: ("I/mA", "<f4"),
    9: ("Ece/V", "<f4"),
    11: ("<I>/mA", "<f8"),
    13: ("(Q-Qo)/mA.h", "<f8"),
    16: ("Analog IN 1/V", "<f4"),
    17: ("Analog IN 2/V", "<f4"),
    19: ("control/V", "<f4"),
    20: ("control/mA", "<f4"),
    23: ("dQ/mA.h", "<f8"),
    24: ("cycle number", "<f8"),
    26: ("Rapp/Ohm", "<f4"),
    27: ("Ewe-Ece/V", "<f4"),
    32: ("freq/Hz", "<f4"),
    33: ("|Ewe|/V", "<f4"),
    34: ("|I|/A", "<f4"),
    35: ("Phase(Z)/deg", "<f4"),
    36: ("|Z|/Ohm", "<f4"),
    37: ("Re(Z)/Ohm", "<f4"),
    38: ("-Im(Z)/Ohm", "<f4"),
    39: ("I Range", "<u2"),
    69: ("R/Ohm", "<f4"),
    70: ("P/W", "<f4"),
    74: ("|Energy|/W.h", "<f8"),
    75: ("Analog OUT/V", "<f4"),
    76: ("<I>/mA", "<f4"),
    77: ("<Ewe>/V", "<f4"),
    78: ("Cs-2/µF-2", "<f4"),
    96: ("|Ece|/V", "<f4"),
    98: ("Phase(Zce)/deg", "<f4"),
    99: ("|Zce|/Ohm", "<f4"),
    100: ("Re(Zce)/Ohm", "<f4"),
    101: ("-Im(Zce)/Ohm", "<f4"),
    123: ("Energy charge/W.h", "<f8"),
    124: ("Energy discharge/W.h", "<f8"),
    125: ("Capacitance charge/µF", "<f8"),
    126: ("Capacitance discharge/µF", "<f8"),
    131: ("Ns", "<u2"),
    163: ("|Estack|/V", "<f4"),
    168: ("Rcmp/Ohm", "<f4"),
    169: ("Cs/µF", "<f4"),
    172: ("Cp/µF", "<f4"),
    173: ("Cp-2/µF-2", "<f4"),
    174: ("<Ewe>/V", "<f4"),
    178: ("(Q-Qo)/C", "<f4"),
    179: ("dQ/C", "<f4"),
    182: ("step time/s", "<f8"),
    211: ("Q charge/discharge/mA.h", "<f8"),
    212: ("half cycle", "<u4"),
    213: ("z cycle", "<u4"),
    217: ("THD Ewe/%", "<f4"),
    218: ("THD I/%", "<f4"),
    220: ("NSD Ewe/%", "<f4"),
    221: ("NSD I/%", "<f4"),
    223: ("NSR Ewe/%", "<f4"),
    224: ("NSR I/%", "<f4"),
    230: ("|Ewe h2|/V", "<f4"),
    231: ("|Ewe h3|/V", "<f4"),
    232: ("|Ewe h4|/V", "<f4"),
    233: ("|Ewe h5|/V", "<f4"),
    234: ("|Ewe h6|/V", "<f4"),
    235: ("|Ewe h7|/V", "<f4"),
    236: ("|I h2|/A", "<f4"),
    237: ("|I h3|/A", "<f4"),
    238: ("|I h4|/A", "<f4"),
    239: ("|I h5|/A", "<f4"),
    240: ("|I h6|/A", "<f4"),
    241: ("|I h7|/A", "<f4"),
    242: ("|E2|/V", "<f4"),
    271: ("Phase(Z1) / deg", "<f4"),
    272: ("Phase(Z2) / deg", "<f4"),
    301: ("|Z1|/Ohm", "<f4"),
    302: ("|Z2|/Ohm", "<f4"),
    331: ("Re(Z1)/Ohm", "<f4"),
    332: ("Re(Z2)/Ohm", "<f4"),
    361: ("-Im(Z1)/Ohm", "<f4"),
    362: ("-Im(Z2)/Ohm", "<f4"),
    391: ("<E1>/V", "<f4"),
    392: ("<E2>/V", "<f4"),
    422: ("Phase(Zstack)/deg", "<f4"),
    423: ("|Zstack|/Ohm", "<f4"),
    424: ("Re(Zstack)/Ohm", "<f4"),
    425: ("-Im(Zstack)/Ohm", "<f4"),
    426: ("<Estack>/V", "<f4"),
    430: ("Phase(Zwe-ce)/deg", "<f4"),
    431: ("|Zwe-ce|/Ohm", "<f4"),
    432: ("Re(Zwe-ce)/Ohm", "<f4"),
    433: ("-Im(Zwe-ce)/Ohm", "<f4"),
    434: ("(Q-Qo)/C", "<f4"),
    435: ("dQ/C", "<f4"),
    438: ("step time/s", "<f8"),
    441: ("<Ecv>/V", "<f4"),
    462: ("Temperature/°C", "<f4"),
    467: ("Q charge/discharge/mA.h", "<f8"),
    468: ("half cycle", "<u4"),
    469: ("z cycle", "<u4"),
    471: ("<Ece>/V", "<f4"),
    473: ("THD Ewe/%", "<f4"),
    474: ("THD I/%", "<f4"),
    476: ("NSD Ewe/%", "<f4"),
    477: ("NSD I/%", "<f4"),
    479: ("NSR Ewe/%", "<f4"),
    480: ("NSR I/%", "<f4"),
    486: ("|Ewe h2|/V", "<f4"),
    487: ("|Ewe h3|/V", "<f4"),
    488: ("|Ewe h4|/V", "<f4"),
    489: ("|Ewe h5|/V", "<f4"),
    490: ("|Ewe h6|/V", "<f4"),
    491: ("|Ewe h7|/V", "<f4"),
    492: ("|I h2|/A", "<f4"),
    493: ("|I h3|/A", "<f4"),
    494: ("|I h4|/A", "<f4"),
    495: ("|I h5|/A", "<f4"),
    496: ("|I h6|/A", "<f4"),
    497: ("|I h7|/A", "<f4"),
    498: ("Q charge/mA.h", "<f8"),
    499: ("Q discharge/mA.h", "<f8"),
    500: ("step time/s", "<f8"),
    501: ("Efficiency/%", "<f8"),
    502: ("Capacity/mA.h", "<f8"),
    505: ("Rdc/Ohm", "<f4"),
    509: ("Acir/Dcir Control", "<u1"),
}

MPR_FLAG_COLUMN_MAP: dict[int, tuple[str, int, Any]] = {
    1: ("mode", 0x03, np.uint8),
    2: ("ox/red", 0x04, np.bool_),
    3: ("error", 0x08, np.bool_),
    21: ("control changes", 0x10, np.bool_),
    31: ("Ns changes", 0x20, np.bool_),
    65: ("counter inc.", 0x80, np.bool_),
}


@dataclass(frozen=True)
class MprModule:
    """MPR 文件中的一个 VMP 模块。"""

    shortname: bytes
    longname: bytes
    length: int
    version: int
    date: bytes
    data: bytes


@dataclass(frozen=True)
class MprContent:
    """解析后的 MPR 主数据和元数据。"""

    data: np.ndarray
    cols: np.ndarray
    version: int
    npts: int
    startdate: date | None
    enddate: date | None
    timestamp: datetime | None
    modules: tuple[MprModule, ...]


class BiologicMprReader(BaseReader):
    """BioLogic EC-Lab MPR 二进制文件读取器。"""

    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "biologic"
    spec: ClassVar[ReaderSpec] = ReaderSpec(
        name="biologic_mpr",
        extensions=(".mpr",),
        instruments=("biologic",),
        techniques=("echem",),
        supports_directory=True,
        description="BioLogic EC-Lab 二进制 MPR 文件",
    )

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """初始化 BioLogic MPR reader。"""
        if "technique" not in kwargs:
            kwargs["technique"] = ["echem"]
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> DataBundle:
        """加载单个 BioLogic MPR 文件。"""
        mpr = _read_mpr_file(path)
        dataset = self._create_dataset(mpr)
        source_metadata = self._metadata(path, mpr)
        metadata = Metadata(
            sample_name=self.sample_name or path.stem,
            start_time=self.start_time or source_metadata.get("start_time"),
            operator=self.operator,
            technique=self._techniques(mpr.data.dtype.names or ()),
            instrument=self.instrument,
            active_material_mass=self.active_material_mass,
            wave_number=self.wave_number,
            raw_metadata=source_metadata,
        )
        return DataBundle(data=dataset, meta=metadata, provenance={"source_path": str(path), "reader": self.__class__.__name__})

    @staticmethod
    def _create_dataset(mpr: MprContent) -> xr.Dataset:
        """将 MPR 主数据转换为 xarray Dataset。"""
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

            if mpr.timestamp is not None:
                systime = pd.to_datetime(mpr.timestamp) + pd.to_timedelta(time_values, unit="s")
                dataset = dataset.assign_coords(systime=(("record",), systime))
                dataset.systime.attrs.update({"long_name": "System Time"})

        dataset.attrs["source_format"] = "mpr"
        dataset.attrs["reader"] = BiologicMprReader.__name__
        return dataset

    @staticmethod
    def _metadata(path: Path, mpr: MprContent) -> dict[str, Any]:
        timestamp = mpr.timestamp
        return {
            "file_path": str(path),
            "source_format": "mpr",
            "reader": BiologicMprReader.__name__,
            "start_time": None if timestamp is None else timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "column_ids": [int(column_id) for column_id in mpr.cols],
            "version": int(mpr.version),
            "n_records": int(mpr.npts),
            "module_names": [_decode_module_name(module.shortname) for module in mpr.modules],
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


def _read_mpr_file(path: Path) -> MprContent:
    """读取 MPR 文件并解析主数据模块。"""
    with path.open("rb") as mpr_file:
        magic = mpr_file.read(len(MPR_MAGIC))
        if magic != MPR_MAGIC:
            raise ValueError(f"BioLogic MPR 文件头无效: {magic!r}")

        modules = tuple(_read_mpr_modules(mpr_file))

    settings_module = _single_module(modules, b"VMP Set   ")
    data_module = _single_module(modules, b"VMP data  ")
    log_module = _optional_module(modules, b"VMP LOG   ")

    data, cols = _parse_data_module(data_module)
    timestamp, enddate = _parse_log_module(log_module)
    startdate = _parse_biologic_date(settings_module.date)

    if startdate is not None and timestamp is not None and startdate != timestamp.date():
        logger.warning("MPR 文件日期不一致: startdate=%s, timestamp=%s", startdate, timestamp.date())

    return MprContent(
        data=data,
        cols=cols,
        version=data_module.version,
        npts=len(data),
        startdate=startdate,
        enddate=enddate,
        timestamp=timestamp,
        modules=modules,
    )


def _read_mpr_modules(fileobj: BinaryIO) -> list[MprModule]:
    """读取 MPR 文件中的所有模块。"""
    modules: list[MprModule] = []
    while True:
        module_magic = fileobj.read(len(MODULE_MAGIC))
        if len(module_magic) == 0:
            break
        if module_magic != MODULE_MAGIC:
            raise ValueError(f"期望读取 MODULE 标记，但实际为: {module_magic!r}")

        header = _read_module_header(fileobj)
        payload = fileobj.read(header["length"])
        if len(payload) != header["length"]:
            raise OSError(f"读取模块 {header['longname']!r} 失败，期望 {header['length']} 字节，实际 {len(payload)} 字节。")

        modules.append(
            MprModule(
                shortname=header["shortname"],
                longname=header["longname"],
                length=header["length"],
                version=header["version"],
                date=header["date"],
                data=payload,
            )
        )
    return modules


def _read_module_header(fileobj: BinaryIO) -> dict[str, Any]:
    """读取单个 VMP 模块头。"""
    header_bytes = fileobj.read(VMP_MODULE_HEADER_V1.itemsize)
    if len(header_bytes) < VMP_MODULE_HEADER_V1.itemsize:
        raise OSError("读取 MPR 模块头时提前到达文件末尾。")

    header_dtype = VMP_MODULE_HEADER_V1
    if header_bytes[35:39] == b"\xff\xff\xff\xff":
        header_dtype = VMP_MODULE_HEADER_V2
        header_bytes += fileobj.read(VMP_MODULE_HEADER_V2.itemsize - VMP_MODULE_HEADER_V1.itemsize)

    header = cast(Any, np.frombuffer(header_bytes, dtype=header_dtype, count=1))
    return {name: _plain_header_value(header[name][0]) for name in header_dtype.names or ()}


def _parse_data_module(data_module: MprModule) -> tuple[np.ndarray, np.ndarray]:
    """解析 VMP data 模块，返回记录数组和列 ID。"""
    n_data_points = _read_scalar(data_module.data[:4], "<u4")
    n_columns = _read_scalar(data_module.data[4:5], "u1")
    column_ids, main_data = _extract_column_ids_and_payload(data_module, n_columns)
    record_dtype = _dtype_from_column_ids(column_ids)
    data = np.frombuffer(main_data, dtype=record_dtype)

    if len(data) != n_data_points:
        raise ValueError(f"MPR 记录数不匹配: header={n_data_points}, data={len(data)}")
    return data, column_ids


def _extract_column_ids_and_payload(data_module: MprModule, n_columns: int) -> tuple[np.ndarray, bytes]:
    """按 data module 版本提取列 ID 和主数据 payload。"""
    data = data_module.data
    if data_module.version == 0:
        if _read_scalar(data[5:6], "u1") != 0:
            column_ids = np.frombuffer(data[5:], dtype="u1", count=n_columns)
            payload = data[100:]
        else:
            raw_column_ids = np.frombuffer(data[5:], dtype="u1", count=n_columns * 2)
            column_ids = raw_column_ids[1::2]
            payload = data[1007:]
    elif data_module.version in {2, 3}:
        column_ids = np.frombuffer(data[5:], dtype="<u2", count=n_columns)
        payload_offset = 406 if data_module.version == 3 else 405
        payload = data[payload_offset:]
    else:
        raise ValueError(f"不支持的 BioLogic MPR data module 版本: {data_module.version}")
    return column_ids, payload


def _dtype_from_column_ids(column_ids: np.ndarray) -> np.dtype:
    """根据 BioLogic 列 ID 构造 NumPy 结构化 dtype。"""
    dtype_fields: list[tuple[str, str]] = []
    field_name_counts: defaultdict[str, int] = defaultdict(int)
    has_flags = False

    for raw_column_id in column_ids:
        column_id = int(raw_column_id)
        if column_id in MPR_FLAG_COLUMN_MAP:
            if not has_flags:
                dtype_fields.append(("flags", "u1"))
                has_flags = True
            continue

        if column_id not in MPR_COLUMN_DTYPE_MAP:
            previous = dtype_fields[-1][0] if dtype_fields else "<none>"
            raise NotImplementedError(f"未知 BioLogic MPR 列 ID: {column_id}，前一列: {previous}")

        field_name, field_type = MPR_COLUMN_DTYPE_MAP[column_id]
        field_name_counts[field_name] += 1
        count = field_name_counts[field_name]
        dtype_fields.append((field_name if count == 1 else f"{field_name} {count}", field_type))

    return np.dtype(dtype_fields)


def _parse_log_module(log_module: MprModule | None) -> tuple[datetime | None, date | None]:
    """从 LOG 模块提取 OLE Automation 时间戳。"""
    if log_module is None:
        return None, None

    enddate = _parse_biologic_date(log_module.date)
    for offset in OLE_TIMESTAMP_OFFSETS:
        if offset + 8 > len(log_module.data):
            continue
        ole_timestamp = _read_scalar(log_module.data[offset : offset + 8], "<f8")
        if 40_000 < ole_timestamp < 50_000:
            return OLE_BASE + timedelta(days=float(ole_timestamp)), enddate

    logger.warning("未能在 BioLogic MPR LOG 模块中找到有效 OLE 时间戳。")
    return None, enddate


def _parse_biologic_date(date_value: bytes | str) -> date | None:
    """解析 BioLogic 模块日期。"""
    date_text = date_value.decode("ascii", errors="ignore") if isinstance(date_value, bytes) else date_value
    date_text = date_text.strip("\x00 ")
    if not date_text:
        return None

    for date_format in ("%m/%d/%y", "%m-%d-%y", "%m.%d.%y"):
        try:
            parsed = time.strptime(date_text, date_format)
            return date(parsed.tm_year, parsed.tm_mon, parsed.tm_mday)
        except ValueError:
            continue
    logger.debug("无法解析 BioLogic 日期: %s", date_text)
    return None


def _single_module(modules: tuple[MprModule, ...], shortname: bytes) -> MprModule:
    """按 shortname 获取唯一模块。"""
    matches = [module for module in modules if module.shortname == shortname]
    if len(matches) != 1:
        raise ValueError(f"期望找到唯一 MPR 模块 {shortname!r}，实际数量为 {len(matches)}。")
    return matches[0]


def _optional_module(modules: tuple[MprModule, ...], shortname: bytes) -> MprModule | None:
    """按 shortname 获取可选模块。"""
    matches = [module for module in modules if module.shortname == shortname]
    if len(matches) > 1:
        raise ValueError(f"期望 MPR 模块 {shortname!r} 最多出现一次，实际数量为 {len(matches)}。")
    return matches[0] if matches else None


def _read_scalar(buffer: bytes, dtype: str) -> Any:
    """从二进制 buffer 中读取单个 NumPy 标量。"""
    return np.frombuffer(buffer, dtype=dtype, count=1).item()


def _plain_header_value(value: Any) -> Any:
    """将模块头中的 NumPy 标量转为普通 Python 值。"""
    if isinstance(value, bytes):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def _decode_module_name(value: bytes) -> str:
    """解码模块名称。"""
    return value.decode("ascii", errors="replace").strip("\x00 ")


def _plain_array(values: np.ndarray) -> np.ndarray:
    """返回包含元数据安全标量值的数组。"""
    if values.dtype.kind == "S":
        return values.astype(str)
    return values


__all__ = ["BiologicMprReader"]
