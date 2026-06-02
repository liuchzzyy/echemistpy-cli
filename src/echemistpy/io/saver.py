"""科学数据的统一文件保存接口。

本模块提供简化的数据保存接口。支持常见的格式，
如 CSV、JSON 和 NetCDF/HDF5。

主要功能：
- 数据保存：支持 .nc（NetCDF）和 .csv 格式
- 元数据保存：将 Info 对象保存为 .dat（JSON 格式）
- 组合保存：将数据和元数据一起保存到单个文件
- 自定义编码：处理 datetime、Path 等特殊类型
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd
import xarray as xr

from echemistpy.io.structures import (
    BaseData,
    BaseInfo,
)


class MetadataEncoder(json.JSONEncoder):
    """Custom JSON encoder for metadata containing datetime objects."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _to_list(obj: Any) -> list[Any]:
    """Convert object to list if it's not already a sequence."""
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _sanitize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Sanitize dataset variable and coordinate names for NetCDF saving.
    Replaces '/' with '_' as NetCDF/HDF5 doesn't allow '/' in names.
    Also handles timedelta64 encoding conflicts.
    """
    # 1. Rename variables with '/'
    rename_dict = {str(name): str(name).replace("/", "_") for name in list(ds.data_vars) + list(ds.coords) if "/" in str(name)}
    if rename_dict:
        ds = ds.rename(rename_dict)

    # 2. Handle timedelta64 units conflict
    for var_name in list(ds.data_vars) + list(ds.coords):
        if ds[var_name].dtype.kind == "m" and "units" in ds[var_name].attrs:
            # Remove units attribute to let xarray handle encoding
            del ds[var_name].attrs["units"]

    return ds


def save_info(
    info: Union[BaseInfo, Sequence[BaseInfo]],
    path: str | Path,
) -> None:
    """Save one or more Info objects to a .dat file.

    Args:
        info: Single or sequence of Info objects (RawDataInfo, ResultsDataInfo).
        path: Destination path (should end in .dat).
    """
    path = Path(path)
    info_list = _to_list(info)

    # Convert to dicts
    data_to_save = [i.to_dict() for i in info_list]

    # If only one, save as dict, otherwise as list of dicts
    output = data_to_save[0] if len(data_to_save) == 1 else data_to_save

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False, cls=MetadataEncoder)


def save_data(
    data: Union[BaseData, Sequence[BaseData]],
    path: str | Path,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save one or more Data objects to .nc or .csv.

    Args:
        data: Single or sequence of Data objects (RawData, ResultsData).
        path: Destination path.
        fmt: Optional format override ('nc', 'csv').
        **kwargs: Additional arguments for xarray.to_netcdf or pandas.to_csv.
    """
    path = Path(path)
    ext = fmt.lower() if fmt else path.suffix.lower().lstrip(".")
    data_list = _to_list(data)
    datasets = [d.data for d in data_list]

    if ext in {"nc", "netcdf"}:
        kwargs.setdefault("engine", "h5netcdf")
        sanitized_datasets = [_sanitize_dataset(ds) for ds in datasets]
        if len(sanitized_datasets) == 1:
            sanitized_datasets[0].to_netcdf(path, **kwargs)
        else:
            # Merge datasets if possible
            combined = xr.merge(sanitized_datasets)
            combined.to_netcdf(path, **kwargs)

    elif ext == "csv":
        dfs = []
        for ds in datasets:
            df = ds.to_dataframe()
            # Reset index if it's a standard record/row dimension
            if "record" in df.index.names or "row" in df.index.names:
                df = df.reset_index(drop=True)
            dfs.append(df)

        final_df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

        final_df.to_csv(path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use 'nc' or 'csv'.")


def save_combined(
    data: Union[BaseData, Sequence[BaseData]],
    info: Union[BaseInfo, Sequence[BaseInfo]],
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Save combined data and info to a .nc file.

    Args:
        data: Single or sequence of Data objects (RawData, ResultsData).
        info: Single or sequence of Info objects (RawDataInfo, ResultsDataInfo).
        path: Destination path.
        **kwargs: Additional arguments for xarray.to_netcdf.
    """
    path = Path(path)
    data_list = _to_list(data)
    info_list = _to_list(info)

    if len(data_list) != len(info_list) and len(info_list) != 1:
        raise ValueError("Number of data objects and info objects must match, or provide a single info object.")

    combined_datasets = []
    for i, d in enumerate(data_list):
        ds = d.data.copy()
        # Use the corresponding info or the only info provided
        current_info = info_list[i] if i < len(info_list) else info_list[0]
        # Update attributes with metadata, filtering out None values and converting complex types to JSON strings
        info_dict = {}
        for k, v in current_info.to_dict().items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)) or (isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v)):
                info_dict[k] = v
            else:
                # Convert dicts and nested structures to JSON strings
                info_dict[k] = json.dumps(v, ensure_ascii=False, cls=MetadataEncoder)
        ds.attrs.update(info_dict)
        combined_datasets.append(_sanitize_dataset(ds))

    kwargs.setdefault("engine", "h5netcdf")
    if len(combined_datasets) == 1:
        combined_datasets[0].to_netcdf(path, **kwargs)
    else:
        combined = xr.merge(combined_datasets)
        combined.to_netcdf(path, **kwargs)


__all__ = [
    "save_combined",
    "save_data",
    "save_info",
]
