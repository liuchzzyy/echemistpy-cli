"""Visualization module for XAS analysis."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_echem_xas(
    echem_data: xr.Dataset,
    xas_data: Union[xr.Dataset, xr.DataTree],
    time_col: str = "abs_time",
    voltage_col: str = "Ewe/V",
    current_col: str | None = "I/mA",
    xas_time_col: str = "systime",
    group_by: str = "file_name",
    output_path: Optional[str] = None,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """
    Plot Time-Electrochemistry-Spectrum sequence summary (LC Plot).

    X-axis: Absolute Time
    Y-axis (Left): Voltage (from Echem data)
    Scatter: XAS Scan moments overlaid on Voltage profile.

    Args:
        echem_data: Dataset containing electrochemical data (Time, Voltage).
        xas_data: Dataset or DataTree containing XAS scans with timestamps.
        time_col: Name of absolute time column in echem_data.
        voltage_col: Name of voltage column in echem_data.
        current_col: Name of current column in echem_data (optional, for right axis).
        xas_time_col: Name of timestamp coordinate/attribute in xas_data.
        group_by: Key to group XAS scans by (for color/shape differentiation).
                  Can be a coordinate name or attribute.
        output_path: If provided, save figure to this path.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 1. Plot Echem Data (Voltage vs Time)
    if time_col not in echem_data:
        raise ValueError(f"Echem data missing time column: {time_col}")
    if voltage_col not in echem_data:
        raise ValueError(f"Echem data missing voltage column: {voltage_col}")

    # Ensure time is sorted for plotting
    echem_sorted = echem_data.sortby(time_col)
    times = echem_sorted[time_col].values
    volts = echem_sorted[voltage_col].values

    # Convert to datetime if it's not already, usually xarray handles datetime64
    ax.plot(times, volts, color="gray", alpha=0.6, linewidth=1.5, label="Voltage Profile")
    ax.set_ylabel(f"Voltage ({echem_data[voltage_col].attrs.get('units', 'V')})")
    ax.set_xlabel("Time")

    # Optional: Plot Current on right axis
    if current_col and current_col in echem_sorted:
        ax2 = ax.twinx()
        curr = echem_sorted[current_col].values
        ax2.plot(
            times,
            curr,
            color="lightblue",
            alpha=0.3,
            linewidth=1,
            linestyle="--",
            label="Current",
        )
        ax2.set_ylabel(f"Current ({echem_data[current_col].attrs.get('units', 'mA')})")
        # Add 'Current' to legend manually or skip

    # 2. Extract and Plot XAS Markers
    # We need to gather (time, group_label) tuples
    xas_points: list[dict[str, Any]] = []

    def extract_points(ds: xr.Dataset, label_prefix: str = ""):
        if xas_time_col not in ds.coords and xas_time_col not in ds.attrs:
            return

        # Get time data
        if xas_time_col in ds.coords:
            ts = ds[xas_time_col].values
            # If it's a single value (scalar coordinate), wrap it
            if ts.ndim == 0:
                ts = [ts]
        else:
            # Attribute fallback
            ts = [pd.to_datetime(ds.attrs[xas_time_col])]

        # Get group labels
        if group_by in ds.coords:
            labels = ds[group_by].values
            if labels.ndim == 0:
                labels = [labels]
        elif group_by in ds.attrs:
            labels = [ds.attrs[group_by]] * len(ts)
        else:
            labels = [label_prefix] * len(ts)

        for t, l in zip(ts, labels):
            xas_points.append({"time": t, "label": str(l)})

    if isinstance(xas_data, xr.Dataset):
        extract_points(xas_data, label_prefix="Scan")
    elif isinstance(xas_data, xr.DataTree):
        for node in xas_data.subtree:
            if node.dataset is not None:
                # Use node name as default label if group_by not found
                name_str = node.name if node.name else "Node"
                extract_points(node.dataset, label_prefix=name_str)

    if not xas_points:
        logger.warning("No XAS timestamps found matching '%s'", xas_time_col)
        return fig

    # 3. Interpolate Voltage for XAS points
    # We use numpy interp. Convert times to float seconds for interpolation.
    df_xas = pd.DataFrame(xas_points)

    # Ensure times are compatible (datetime64[ns])
    try:
        df_xas["time"] = pd.to_datetime(df_xas["time"])
        # Echem times conversion
        t_echem_nums = mdates.date2num(times)
        t_xas_nums = mdates.date2num(df_xas["time"])

        # Interpolate voltage
        df_xas["voltage"] = np.interp(t_xas_nums, t_echem_nums, volts)
    except Exception as e:
        logger.error("Failed to interpolate voltage for XAS points: %s", e)
        return fig

    # 4. Scatter Plot by Group
    groups = df_xas.groupby("label")
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]
    colors = plt.cm.tab10.colors  # type: ignore

    for i, (label, group) in enumerate(groups):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        ax.scatter(
            group["time"],
            group["voltage"],
            marker=marker,
            s=50,
            color=color,
            edgecolor="k",
            label=f"XAS: {label}",
            zorder=10,
        )

    # Format Date Axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Operando Synchronization: Echem & XAS")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info("Saved LC plot to %s", output_path)

    return fig
