"""数据层通用工具函数。

本模块包含数据层通用辅助函数，用于 xarray 名称清理、元数据合并和标准属性写入。

主要功能：
- 变量名清理：为 DataTree 兼容性清理变量名（替换 '/'）
- 元数据合并：合并多个 Metadata 对象
- 标准属性应用：为不同技术类型应用标准单位属性
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import xarray as xr

from echemistpy.data.models import Metadata

logger = logging.getLogger(__name__)


def sanitize_variable_names(obj: xr.Dataset | xr.DataTree | dict[str, Any]) -> xr.Dataset | xr.DataTree | dict[str, Any]:
    """清理变量名，将 '/' 替换为 '_' 以兼容 DataTree。

    注意：如果目标名称已存在，则添加后缀以避免冲突。
    对于 DataTree，递归处理每个节点。

    Args:
        obj: xarray Dataset, DataTree 或字典进行清理

    Returns:
        清理后的对象
    """
    if isinstance(obj, xr.DataTree):
        # 对 DataTree 的每个节点进行处理
        def _sanitize_node(ds: xr.Dataset) -> xr.Dataset:
            # 递归调用处理 Dataset
            result = sanitize_variable_names(ds)
            return result if isinstance(result, xr.Dataset) else ds

        return obj.map_over_datasets(_sanitize_node)

    if isinstance(obj, xr.Dataset):
        rename_dict = {}
        all_names = list(obj.data_vars) + list(obj.coords)

        for name in all_names:
            name_str = str(name)
            if "/" in name_str:
                new_name = name_str.replace("/", "_")
                # 如果目标名称已存在，添加后缀避免冲突
                if new_name in all_names and new_name != name_str:
                    suffix = 1
                    while f"{new_name}_{suffix}" in all_names:
                        suffix += 1
                    new_name = f"{new_name}_{suffix}"
                if new_name != name_str:
                    rename_dict[name_str] = new_name

        return obj.rename(rename_dict) if rename_dict else obj

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            new_key = k.replace("/", "_")
            # 如果新键已存在，添加后缀避免冲突
            if new_key in result and new_key != k:
                suffix = 1
                while f"{new_key}_{suffix}" in result:
                    suffix += 1
                new_key = f"{new_key}_{suffix}"
            result[new_key] = v
        return result

    return obj


def merge_metadata(  # noqa: PLR0914
    metadata_items: list[Metadata],
    root_path: Path,
    **overrides: Any,
) -> Metadata:
    """合并多个 Metadata 对象。

    多文件目录读取时使用。标准字段取唯一值或覆盖值，文件级信息写入
    ``raw_metadata``。

    Args:
        metadata_items: 元数据对象列表
        root_path: 目录根路径
        **overrides: 可选覆盖项

    Returns:
        合并后的 Metadata 对象
    """
    # 提取覆盖项。
    sample_name_override = overrides.get("sample_name_override")
    operator_override = overrides.get("operator_override")
    start_time_override = overrides.get("start_time_override")
    active_material_mass_override = overrides.get("active_material_mass_override")
    wave_number_override = overrides.get("wave_number_override")
    technique = overrides.get("technique")
    instrument = overrides.get("instrument")
    if not metadata_items:
        return Metadata()

    # 汇总所有文件的技术类型；echem 只保留一次，子技术按文件保留。
    all_techs = []
    seen_echem = False
    for meta in metadata_items:
        for tech in meta.technique:
            if tech.lower() == "echem":
                if not seen_echem:
                    all_techs.append(tech)
                    seen_echem = True
            else:
                all_techs.append(tech)

    # 汇总各标准字段；sample_names 保持原始顺序，不去重。
    sample_names = [meta.sample_name for meta in metadata_items if meta.sample_name]
    operators = sorted({meta.operator for meta in metadata_items if meta.operator})
    start_times = sorted({meta.start_time for meta in metadata_items if meta.start_time})
    masses = sorted({meta.active_material_mass for meta in metadata_items if meta.active_material_mass})
    wave_numbers = sorted({meta.wave_number for meta in metadata_items if meta.wave_number})

    # 将文件级汇总信息写入 raw_metadata。
    combined_raw_metadata: dict[str, Any] = {
        "n_files": len(metadata_items),
        "sample_names": sample_names,
    }

    # 多个不同值以列表形式保存。
    if len(operators) > 1:
        combined_raw_metadata["all_operators"] = operators
    if len(masses) > 1:
        combined_raw_metadata["all_active_material_masses"] = masses
    if len(wave_numbers) > 1:
        combined_raw_metadata["all_wave_numbers"] = wave_numbers

    # 确定文件夹名称：对于相对路径（如 '.' 或 '..'）使用解析后的绝对路径名称
    # 注意：Path('.').name 返回空字符串 ''，而非 '.'
    folder_name = root_path.resolve().name if not root_path.name or root_path.name in {".", ".."} else root_path.name

    return Metadata(
        sample_name=sample_name_override or folder_name,
        technique=technique or all_techs,
        instrument=instrument,
        operator=operator_override or (operators[0] if len(operators) == 1 else None),
        start_time=start_time_override or (start_times[0] if len(start_times) == 1 else None),
        active_material_mass=active_material_mass_override or (masses[0] if len(masses) == 1 else None),
        wave_number=wave_number_override or (wave_numbers[0] if len(wave_numbers) == 1 else None),
        raw_metadata=combined_raw_metadata,
    )


def apply_standard_attrs_echem(ds: xr.Dataset) -> None:
    """写入电化学数据的标准单位和长名称。

    Args:
        ds: 需要原地修改的 xarray Dataset
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


def apply_standard_attrs_xrd(ds: xr.Dataset) -> None:
    """写入 XRD 数据的标准单位和长名称。

    Args:
        ds: 需要原地修改的 xarray Dataset
    """
    attr_map = {
        "intensity": {"units": "counts", "long_name": "Intensity"},
        "intensity_error": {"units": "counts", "long_name": "Intensity Error"},
        "2theta": {"units": "degree", "long_name": "2-Theta"},
    }
    for var, attrs in attr_map.items():
        if var in ds:
            ds[var].attrs.update(attrs)
        if var in ds.coords:
            ds.coords[var].attrs.update(attrs)


def apply_standard_attrs_xas(ds: xr.Dataset) -> None:
    """写入 XAS 数据的标准单位和长名称。

    Args:
        ds: 需要原地修改的 xarray Dataset
    """
    if "energyc" in ds:
        ds.energyc.attrs.update({"units": "eV", "long_name": "Energy"})
    if "absorption" in ds:
        ds.absorption.attrs.update({"units": "a.u.", "long_name": "Absorption"})
    if "time_s" in ds.coords:
        ds.time_s.attrs.update({"units": "s", "long_name": "Relative Time"})
    if "systime" in ds.coords:
        ds.systime.attrs.update({"long_name": "System Time"})


def apply_standard_attrs_txm(ds: xr.Dataset) -> None:
    """写入 TXM 数据的标准单位和长名称。

    Args:
        ds: 需要原地修改的 xarray Dataset
    """
    if "energy" in ds.coords:
        ds.energy.attrs.update({"units": "eV", "long_name": "Energy"})
    if "x" in ds.coords:
        ds.x.attrs.update({"units": "nm", "long_name": "X Position"})
    if "y" in ds.coords:
        ds.y.attrs.update({"units": "nm", "long_name": "Y Position"})
    if "transmission" in ds:
        ds.transmission.attrs.update({"units": "a.u.", "long_name": "Transmission"})
    if "optical_density" in ds:
        ds.optical_density.attrs.update({"units": "a.u.", "long_name": "Optical Density"})
    if "rotation_angle" in ds:
        ds.rotation_angle.attrs.update({"units": "deg", "long_name": "Rotation Angle"})


__all__ = [
    "apply_standard_attrs_echem",
    "apply_standard_attrs_txm",
    "apply_standard_attrs_xas",
    "apply_standard_attrs_xrd",
    "merge_metadata",
    "sanitize_variable_names",
]
