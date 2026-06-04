"""科学数据的统一文件保存接口。

本模块提供简化的数据保存接口。支持常见的格式，
如 CSV、JSON 和 NetCDF/HDF5。

主要功能：
- 数据保存：支持 .nc（NetCDF）和 .csv 格式
- 元数据保存：将 bundle 元数据保存为 .dat（JSON 格式）
- 组合保存：将数据和元数据一起保存到单个文件
- 自定义编码：处理 datetime、Path 等特殊类型
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import pandas as pd
import xarray as xr

from echemistpy.data.models import (
    AnalysisBundle,
    DataBundle,
    Metadata,
)

Bundle = DataBundle | AnalysisBundle


class MetadataEncoder(json.JSONEncoder):
    """处理 datetime 和 Path 的 JSON 编码器。"""

    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _to_list(obj: Any) -> list[Any]:
    """将单个对象转为列表。"""
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _sanitize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """清理 Dataset 名称和 attrs，避免 NetCDF 写出冲突。"""
    # NetCDF/HDF5 不允许变量名中包含 "/"。
    rename_dict = {str(name): str(name).replace("/", "_") for name in list(ds.data_vars) + list(ds.coords) if "/" in str(name)}
    if rename_dict:
        ds = ds.rename(rename_dict)

    # timedelta64 的 units 由 xarray 编码层处理。
    for var_name in list(ds.data_vars) + list(ds.coords):
        if ds[var_name].dtype.kind == "m" and "units" in ds[var_name].attrs:
            del ds[var_name].attrs["units"]

    return ds


def _sanitize_data(data: xr.Dataset | xr.DataTree) -> xr.Dataset | xr.DataTree:
    """清理 Dataset 或 DataTree 内的所有 Dataset 节点。"""
    if isinstance(data, xr.DataTree):
        return data.map_over_datasets(_sanitize_dataset)
    return _sanitize_dataset(data)


def _metadata_to_attrs(metadata: Metadata) -> dict[str, Any]:
    """将 Metadata 转为 NetCDF 友好的 attrs。"""
    attrs: dict[str, Any] = {}
    for k, v in metadata.to_dict().items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)) or (isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v)):
            attrs[k] = v
        else:
            attrs[k] = json.dumps(v, ensure_ascii=False, cls=MetadataEncoder)
    return attrs


def _bundle_to_attrs(bundle: Bundle) -> dict[str, Any]:
    """将 bundle 级信息转为 NetCDF 友好的 attrs。"""
    attrs: dict[str, Any] = {"schema": bundle.schema}
    if bundle.provenance:
        attrs["provenance"] = json.dumps(bundle.provenance, ensure_ascii=False, cls=MetadataEncoder)
    if bundle.warnings:
        attrs["warnings"] = json.dumps(bundle.warnings, ensure_ascii=False, cls=MetadataEncoder)
    if isinstance(bundle, AnalysisBundle) and bundle.parameters:
        attrs["analysis_parameters"] = json.dumps(bundle.parameters, ensure_ascii=False, cls=MetadataEncoder)
    return attrs


def _with_attrs(data: xr.Dataset | xr.DataTree, attrs: dict[str, Any]) -> xr.Dataset | xr.DataTree:
    """复制数据并将 attrs 写入根对象。"""
    data = data.copy(deep=True)
    data.attrs.update(attrs)
    return data


def _dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """将单个 Dataset 转为扁平表。"""
    df = ds.to_dataframe()
    if "record" in df.index.names or "row" in df.index.names:
        df = df.reset_index(drop=True)
    return df


def _tree_to_dataframes(tree: xr.DataTree, node_path: str = "") -> list[pd.DataFrame]:
    """将 DataTree 中每个有数据的节点转为表。"""
    frames: list[pd.DataFrame] = []
    if tree.dataset is not None and (tree.dataset.data_vars or tree.dataset.sizes):
        df = _dataset_to_dataframe(tree.dataset)
        df.insert(0, "__node__", node_path or "/")
        frames.append(df)

    for name, child in tree.children.items():
        child_path = f"{node_path}/{name}" if node_path else f"/{name}"
        frames.extend(_tree_to_dataframes(child, child_path))
    return frames


def _data_to_dataframes(data: xr.Dataset | xr.DataTree) -> list[pd.DataFrame]:
    """将 Dataset 或 DataTree 转为一个或多个表。"""
    if isinstance(data, xr.DataTree):
        return _tree_to_dataframes(data)
    return [_dataset_to_dataframe(data)]


def save_info(
    bundle: Bundle | Sequence[Bundle],
    path: str | Path,
) -> None:
    """将一个或多个 bundle 的元数据保存到 .dat 文件。

    Args:
        bundle: 单个或多个数据包
        path: 输出路径
    """
    path = Path(path)
    bundle_list = _to_list(bundle)

    data_to_save = []
    for item in bundle_list:
        payload = item.meta.to_dict()
        payload.update(_bundle_to_attrs(item))
        data_to_save.append(payload)

    output = data_to_save[0] if len(data_to_save) == 1 else data_to_save

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False, cls=MetadataEncoder)


def save_data(
    bundle: Bundle | Sequence[Bundle],
    path: str | Path,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """保存一个或多个 bundle 的数据部分。

    Args:
        bundle: 单个或多个数据包
        path: 输出路径
        fmt: 输出格式覆盖，如 ``nc`` 或 ``csv``
        **kwargs: 传递给底层写出函数的参数
    """
    path = Path(path)
    ext = fmt.lower() if fmt else path.suffix.lower().lstrip(".")
    bundle_list = _to_list(bundle)
    datasets = [item.data for item in bundle_list]

    if ext in {"nc", "netcdf"}:
        kwargs.setdefault("engine", "h5netcdf")
        sanitized_datasets = [_sanitize_data(ds) for ds in datasets]
        if len(sanitized_datasets) == 1:
            sanitized_datasets[0].to_netcdf(path, **kwargs)
        else:
            if any(isinstance(ds, xr.DataTree) for ds in sanitized_datasets):
                raise ValueError("暂不支持将多个 DataTree 写入同一个 NetCDF 文件。")
            # 多个 Dataset 可合并为一个 NetCDF。
            dataset_list = [cast(xr.Dataset, ds) for ds in sanitized_datasets]
            combined = xr.merge(dataset_list)
            combined.to_netcdf(path, **kwargs)

    elif ext == "csv":
        dfs = []
        for ds in datasets:
            dfs.extend(_data_to_dataframes(ds))

        final_df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

        final_df.to_csv(path, index=False, **kwargs)
    else:
        raise ValueError(f"不支持的格式: {ext}。请使用 'nc' 或 'csv'。")


def save_combined(
    bundle: Bundle | Sequence[Bundle],
    path: str | Path,
    **kwargs: Any,
) -> None:
    """将数据和元数据保存到同一个 NetCDF 文件。

    Args:
        bundle: 单个或多个数据包
        path: 输出路径
        **kwargs: 传递给 xarray.to_netcdf 的参数
    """
    path = Path(path)
    bundle_list = _to_list(bundle)

    combined_datasets = []
    for item in bundle_list:
        attrs = _metadata_to_attrs(item.meta)
        attrs.update(_bundle_to_attrs(item))
        combined_datasets.append(_sanitize_data(_with_attrs(item.data, attrs)))

    kwargs.setdefault("engine", "h5netcdf")
    if len(combined_datasets) == 1:
        combined_datasets[0].to_netcdf(path, **kwargs)
    else:
        if any(isinstance(ds, xr.DataTree) for ds in combined_datasets):
            raise ValueError("暂不支持将多个 DataTree 写入同一个 NetCDF 文件。")
        dataset_list = [cast(xr.Dataset, ds) for ds in combined_datasets]
        combined = xr.merge(dataset_list)
        combined.to_netcdf(path, **kwargs)


def save_bundle(
    bundle: Bundle,
    path: str | Path,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """按输出扩展名保存 DataBundle 或 AnalysisBundle。"""
    path = Path(path)
    ext = fmt.lower() if fmt else path.suffix.lower().lstrip(".")
    if ext in {"nc", "netcdf"}:
        save_combined(bundle, path, **kwargs)
    elif ext in {"dat", "json"}:
        save_info(bundle, path)
    else:
        save_data(bundle, path, fmt=fmt, **kwargs)


__all__ = [
    "save_bundle",
    "save_combined",
    "save_data",
    "save_info",
]
