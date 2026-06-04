"""XAS 分析结果可视化模块。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_echem_xas(  # noqa: PLR0913, PLR0917
    echem_data: xr.Dataset,
    xas_data: xr.Dataset | xr.DataTree,
    time_col: str = "systime",
    voltage_col: str = "ewe_v",
    current_col: str | None = "current_ma",
    xas_time_col: str = "systime",
    group_by: str = "file_name",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """绘制电化学时间轴和 XAS 扫描时间点的同步概览图。"""
    fig, ax = plt.subplots(figsize=figsize)

    echem_sorted, times, volts, voltage_da = _prepare_echem_data(echem_data, time_col, voltage_col)
    _plot_voltage(ax, times, volts, voltage_da)
    ax2 = _plot_current(ax, echem_sorted, current_col, times)

    xas_points = _collect_xas_points(xas_data, xas_time_col=xas_time_col, group_by=group_by)
    if not xas_points:
        logger.warning("未找到匹配 '%s' 的 XAS 时间戳。", xas_time_col)
        return fig

    df_xas = _interpolate_xas_voltage(xas_points, times, volts)
    if df_xas is None:
        return fig

    _plot_xas_markers(ax, df_xas)
    _finish_figure(fig, ax, ax2, output_path)

    return fig


def _prepare_echem_data(echem_data: xr.Dataset, time_col: str, voltage_col: str) -> tuple[xr.Dataset, Any, np.ndarray, xr.DataArray]:
    """校验并排序电化学时间和电压数据。"""
    time_da = _get_array(echem_data, time_col)
    voltage_da = _get_array(echem_data, voltage_col)
    if time_da is None:
        raise ValueError(f"电化学数据缺少时间列: {time_col}")
    if voltage_da is None:
        raise ValueError(f"电化学数据缺少电压列: {voltage_col}")

    echem_sorted = echem_data.sortby(time_col)
    sorted_time_da = _get_array(echem_sorted, time_col)
    sorted_voltage_da = _get_array(echem_sorted, voltage_col)
    if sorted_time_da is None or sorted_voltage_da is None:
        raise ValueError("电化学数据排序后缺少时间或电压列。")
    times = pd.to_datetime(sorted_time_da.values)
    volts = sorted_voltage_da.values.astype(float)
    return echem_sorted, times, volts, voltage_da


def _plot_voltage(ax: Axes, times: Any, volts: np.ndarray, voltage_da: xr.DataArray) -> None:
    """绘制电压曲线。"""
    ax.plot(times, volts, color="gray", alpha=0.6, linewidth=1.5, label="Voltage Profile")
    ax.set_ylabel(f"Voltage ({voltage_da.attrs.get('units', 'V')})")
    ax.set_xlabel("Time")


def _plot_current(ax: Axes, echem_sorted: xr.Dataset, current_col: str | None, times: Any) -> Axes | None:
    """绘制可选电流曲线。"""
    current_da = _get_array(echem_sorted, current_col) if current_col else None
    if current_da is None:
        return None

    ax2 = ax.twinx()
    curr = current_da.values.astype(float)
    ax2.plot(
        times,
        curr,
        color="lightblue",
        alpha=0.3,
        linewidth=1,
        linestyle="--",
        label="Current",
    )
    ax2.set_ylabel(f"Current ({current_da.attrs.get('units', 'mA')})")
    return ax2


def _interpolate_xas_voltage(xas_points: list[dict[str, Any]], times: Any, volts: np.ndarray) -> pd.DataFrame | None:
    """将 XAS 时间点插值到电化学电压曲线上。"""
    df_xas = pd.DataFrame(xas_points)

    try:
        df_xas["time"] = pd.to_datetime(df_xas["time"])
        t_echem_nums = mdates.date2num(times)
        t_xas_nums = mdates.date2num(df_xas["time"])
        df_xas["voltage"] = np.interp(t_xas_nums, t_echem_nums, volts)
    except Exception as exc:
        logger.error("为 XAS 点插值电压失败: %s", exc)
        return None
    return df_xas


def _plot_xas_markers(ax: Axes, df_xas: pd.DataFrame) -> None:
    """按标签绘制 XAS 扫描时间点。"""
    groups = df_xas.groupby("label")
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(10)]

    for i, (label, group) in enumerate(groups):
        ax.scatter(
            group["time"],
            group["voltage"],
            marker=markers[i % len(markers)],
            s=50,
            color=colors[i % len(colors)],
            edgecolor="k",
            label=f"XAS: {label}",
            zorder=10,
        )


def _finish_figure(fig: Figure, ax: Axes, ax2: Axes | None, output_path: str | Path | None) -> None:
    """设置图例、网格和输出路径。"""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles.extend(handles2)
        labels.extend(labels2)
    ax.legend(handles, labels, loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Operando Synchronization: Echem & XAS")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        logger.info("LC 图已保存到 %s", output_path)


def _get_array(ds: xr.Dataset, name: str) -> xr.DataArray | None:
    """从 Dataset 变量或坐标中读取 DataArray。"""
    if name in ds.data_vars:
        return ds[name]
    if name in ds.coords:
        return ds.coords[name]
    return None


def _collect_xas_points(xas_data: xr.Dataset | xr.DataTree, *, xas_time_col: str, group_by: str) -> list[dict[str, Any]]:
    """从 Dataset/DataTree 中收集 XAS 时间点和分组标签。"""
    points: list[dict[str, Any]] = []

    if isinstance(xas_data, xr.Dataset):
        _append_dataset_points(points, xas_data, xas_time_col=xas_time_col, group_by=group_by, label_prefix="Scan")
        return points

    for node in xas_data.subtree:
        if node.dataset is None or not node.dataset.data_vars:
            continue
        label_prefix = node.path.strip("/") or node.name or "Node"
        _append_dataset_points(points, node.dataset, xas_time_col=xas_time_col, group_by=group_by, label_prefix=label_prefix)
    return points


def _append_dataset_points(points: list[dict[str, Any]], ds: xr.Dataset, *, xas_time_col: str, group_by: str, label_prefix: str) -> None:
    """追加单个 Dataset 的 XAS 时间点。"""
    times = _values_from_dataset(ds, xas_time_col)
    if not times:
        return

    labels = _values_from_dataset(ds, group_by)
    if not labels:
        labels = [label_prefix]
    if len(labels) == 1 and len(times) > 1:
        labels *= len(times)
    if len(labels) != len(times):
        logger.warning("XAS 标签数量 %s 与时间点数量 %s 不一致，使用节点名。", len(labels), len(times))
        labels = [label_prefix] * len(times)

    for time_value, label in zip(times, labels, strict=True):
        points.append({"time": time_value, "label": str(label)})


def _values_from_dataset(ds: xr.Dataset, name: str) -> list[Any]:
    """从 Dataset 坐标、变量或属性中读取一维值列表。"""
    array = _get_array(ds, name)
    if array is not None:
        values = np.asarray(array.values)
        if values.ndim == 0:
            return [values.item()]
        return list(values.reshape(-1))
    if name in ds.attrs:
        return [ds.attrs[name]]
    return []
