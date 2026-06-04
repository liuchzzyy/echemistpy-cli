"""XAS 预处理模块。

Handles data cleaning, calibration, and alignment.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter

from echemistpy.analysis.xas.elements import ELEMENT_DB

Group: Any
find_e0: Any
fluo_corr: Any
xray_edge: Any


def _load_larch_symbols() -> tuple[bool, Any, Any, Any, Any]:
    """加载可选的 larch 符号。"""
    try:
        larch_module = importlib.import_module("larch")
        xafs_module = importlib.import_module("larch.xafs")
        xray_module = importlib.import_module("larch.xray")
    except ImportError:
        return False, None, None, None, None
    return True, larch_module.Group, xafs_module.find_e0, xafs_module.fluo_corr, xray_module.xray_edge


HAS_LARCH, Group, find_e0, fluo_corr, xray_edge = _load_larch_symbols()

logger = logging.getLogger(__name__)
ENERGY_ALIASES = ("energy_ev", "energyc", "energy")


def _energy_name(ds: xr.Dataset) -> str:
    """返回 Dataset 中使用的能量坐标名。"""
    for name in ENERGY_ALIASES:
        if name in ds.coords or name in ds.data_vars or name in ds.dims:
            return name
    raise ValueError(f"Dataset 缺少能量坐标，支持名称: {ENERGY_ALIASES}。")


def calibrate_energy(ds: xr.Dataset, element: str, edge: str = "K", reference_e0: Optional[float] = None) -> float:
    """Calculate energy shift required to calibrate the spectrum.

    Uses the first derivative peak position (E0) of the provided dataset
    and compares it to the theoretical edge energy or a provided reference.

    Args:
        ds: Xarray Dataset with 'energyc' and 'absorption'.
        element: Atomic symbol (e.g., 'Fe').
        edge: Edge type (default 'K').
        reference_e0: Manual reference energy. If None, uses theoretical value.

    Returns:
        delta_e (float): Shift to Apply (Ref - Measured).
                         calibrated_E = measured_E + delta_E
    """
    if not HAS_LARCH:
        raise ImportError("larch is required for calibration.")

    energy_name = _energy_name(ds)
    energy = ds[energy_name].values
    mu = ds.absorption.values

    # Handle multiple records: use the average spectrum for calibration
    if "record" in ds.dims:
        mu = np.nanmean(mu, axis=0)

    # Find measured E0
    # Use the constrained derivative method as requested
    if element in ELEMENT_DB and "e0" in ELEMENT_DB[element]:
        theo_lookup = ELEMENT_DB[element]["e0"]
    # If not in DB, try larch or default
    elif reference_e0:
        theo_lookup = reference_e0
    else:
        # 回退到 Larch 的边能量查询。
        ref_data = xray_edge(element, edge)
        theo_lookup = ref_data[0] if isinstance(ref_data, tuple) else ref_data

    # Use reference_e0 as the "theoretical" center if provided, otherwise the DB value
    center_e0 = reference_e0 if reference_e0 is not None else theo_lookup

    measured_e0 = find_e0_by_derivative(energy, mu, center_e0, search_range=50.0)

    # Get reference E0 (target)
    target_e0 = center_e0

    delta_e = target_e0 - measured_e0
    logger.info("能量校准: 实测 E0=%.2f, 参考 E0=%.2f, 偏移=%.2f", measured_e0, target_e0, delta_e)
    return float(delta_e)


def find_e0_by_derivative(
    energy: np.ndarray,
    mu: np.ndarray,
    theoretical_e0: float,
    search_range: float = 50.0,
) -> float:
    """Find E0 using the maximum of the first derivative within a constrained range.

    Args:
        energy: Energy array.
        mu: Absorption array.
        theoretical_e0: Expected E0 position.
        search_range: Search window +/- eV around theoretical E0.

    Returns:
        Found E0 value.
    """
    # Calculate first derivative
    dmu = np.gradient(mu, energy)

    # Define search window
    mask = (energy >= theoretical_e0 - search_range) & (energy <= theoretical_e0 + search_range)

    if not np.any(mask):
        logger.warning("E0 搜索范围 %s +/- %s 内没有数据点，改用全局最大导数。", theoretical_e0, search_range)
        idx = np.argmax(dmu)
    else:
        # Find max derivative in the window
        # Indices of the window
        window_indices = np.where(mask)[0]
        # Max index relative to window
        max_rel_idx = np.argmax(dmu[window_indices])
        # Absolute index
        idx = window_indices[max_rel_idx]

    return float(energy[idx])


def align_spectra(
    ds: xr.Dataset,
    target_energy: Optional[np.ndarray] = None,
    method: str = "linear",
    shift: float = 0.0,
) -> xr.Dataset:
    """Align spectra to a common energy grid and apply energy shift.

    Args:
        ds: Input Dataset (record x energy).
        target_energy: Common energy grid. If None, uses the energy of the first record
                       (or just applies shift if single record).
        method: Interpolation method ('linear', 'cubic', etc.).
        shift: Energy shift to ADD to the current energy axis before interpolation.

    Returns:
        New Dataset with all records interpolated to target_energy.
    """
    new_ds = ds.copy()

    # Apply shift first
    energy_name = _energy_name(ds)
    current_energy = ds[energy_name].values + shift

    # If single spectrum, just update coordinate
    if "record" not in ds.dims:
        new_ds = new_ds.assign_coords({energy_name: current_energy})
        if target_energy is not None:
            # Interpolate single spectrum to target
            f = interp1d(
                current_energy,
                new_ds.absorption.values,
                kind=method,
                bounds_error=False,
                fill_value=np.nan,
            )
            new_mu = f(target_energy)
            new_ds = xr.Dataset({"absorption": ([energy_name], new_mu)}, coords={energy_name: target_energy})
        return new_ds

    # Multi-record case
    if target_energy is None:
        new_ds = new_ds.assign_coords({energy_name: current_energy})
        return new_ds

    # Interpolate all records to target_energy
    records = ds.record.values

    mu_values = ds.absorption.values  # Shape (N_records, N_energy)

    f = interp1d(
        current_energy,
        mu_values,
        kind=method,
        axis=-1,
        bounds_error=False,
        fill_value=np.nan,
    )
    new_mu = f(target_energy)  # Shape (N_records, N_target)

    # Reconstruct Dataset
    new_ds = xr.Dataset(
        data_vars={
            "absorption": (("record", energy_name), new_mu),
        },
        coords={"record": records, energy_name: target_energy},
    )
    # Copy other vars if compatible
    for v in ds.data_vars:
        if v != "absorption" and energy_name not in ds[v].dims:
            new_ds[v] = ds[v]

    return new_ds


def _fill_nans(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in a 1D array."""
    nans = np.isnan(y)
    if not nans.any():
        return y

    if not np.any(~nans):
        return np.zeros_like(y)

    def indices(mask: np.ndarray) -> np.ndarray:
        return mask.nonzero()[0]

    y_out = y.copy()
    y_out[nans] = np.interp(indices(nans), indices(~nans), y[~nans])
    return y_out


def deglitch(ds: xr.Dataset, window: int = 3, threshold: float = 3.0) -> xr.Dataset:
    """Remove spikes from spectra using a median filter.

    Points deviating more than `threshold` * std_dev from the median filtered
    signal are replaced by the median value.

    Args:
        ds: Input Dataset.
        window: Window size for median filter (must be odd).
        threshold: Z-score threshold for outlier detection.

    Returns:
        Cleaned Dataset.
    """
    if window % 2 == 0:
        window += 1

    ds_out = ds.copy()
    mu = ds.absorption.values

    # Handle 1D or 2D
    is_2d = mu.ndim == 2
    if not is_2d:
        mu = mu[np.newaxis, :]

    cleaned_mu = np.zeros_like(mu)

    for i in range(mu.shape[0]):
        y = mu[i]

        # Fill NaNs first to allow median filter to work
        if np.isnan(y).any():
            y = _fill_nans(y)

        # Median filter
        y_med = medfilt(y, kernel_size=window)
        diff = np.abs(y - y_med)
        std = np.nanstd(diff)

        # Avoid division by zero if flat
        if std == 0:
            cleaned_mu[i] = y
            continue

        mask = diff > (threshold * std)
        y_clean = y.copy()
        y_clean[mask] = y_med[mask]
        cleaned_mu[i] = y_clean

        if np.sum(mask) > 0:
            logger.debug("已在记录 %s 中去除 %s 个毛刺点。", i, np.sum(mask))

    if not is_2d:
        cleaned_mu = cleaned_mu[0]

    ds_out["absorption"].values = cleaned_mu
    return ds_out


def smooth(ds: xr.Dataset, window_length: int = 5, polyorder: int = 2) -> xr.Dataset:
    """Apply Savitzky-Golay smoothing to spectra.

    Args:
        ds: Input Dataset.
        window_length: Length of the filter window.
        polyorder: Order of the polynomial used to fit the samples.

    Returns:
        Smoothed Dataset.
    """
    ds_out = ds.copy()
    mu = ds.absorption.values

    # Handle NaNs
    if np.isnan(mu).any():
        if mu.ndim == 2:
            for i in range(mu.shape[0]):
                mu[i] = _fill_nans(mu[i])
        else:
            mu = _fill_nans(mu)

    smoothed_mu = savgol_filter(mu, window_length, polyorder, axis=-1, mode="interp")

    ds_out["absorption"].values = smoothed_mu
    return ds_out


def correct_fluorescence(
    ds: xr.Dataset,
    formula: str,
    edge: str = "K",
    angle_in: float = 45,
    angle_out: float = 45,
) -> xr.Dataset:
    """Apply fluorescence self-absorption correction.

    Wraps `larch.xafs.fluo_corr`.

    Args:
        ds: Input Dataset.
        formula: Chemical formula of the sample (e.g. 'Fe2O3').
        edge: Edge being measured (e.g. 'K', 'L3').
        angle_in: Incident angle (degrees).
        angle_out: Exit angle (degrees).

    Returns:
        Dataset with corrected 'absorption'.
    """
    if not HAS_LARCH:
        raise ImportError("larch is required for fluorescence correction.")

    ds_out = ds.copy()
    energy_name = _energy_name(ds)
    energy = ds[energy_name].values
    mu = ds.absorption.values

    is_2d = mu.ndim == 2
    if not is_2d:
        mu = mu[np.newaxis, :]

    corrected_mu = np.zeros_like(mu)

    for i in range(mu.shape[0]):
        group = Group(energy=energy, mu=mu[i])
        fluo_corr(group, formula=formula, edge=edge, anginp=angle_in, angout=angle_out)

        if hasattr(group, "mu_corr"):
            corrected_mu[i] = group.mu_corr
        else:
            logger.warning("记录 %s 的荧光校正失败，保留原始数据。", i)
            corrected_mu[i] = mu[i]

    if not is_2d:
        corrected_mu = corrected_mu[0]

    ds_out["absorption"].values = corrected_mu
    return ds_out


def check_consistency(ds: xr.Dataset, correlation_threshold: float = 0.95, energy_tolerance: float = 0.5) -> list[int]:
    """Check consistency of spectra in a dataset (e.g. repeated scans).

    Performs two checks:
    1. Energy Grid: Checks if all records share the same energy grid.
    2. Quality: Calculates Pearson correlation of each spectrum with the median spectrum.

    Args:
        ds: Input Dataset with 'record' dimension.
        correlation_threshold: Min correlation coefficient to accept a spectrum.
        energy_tolerance: Allowed deviation in energy start/end points (eV).

    Returns:
        List of indices of "good" spectra.
    """
    if "record" not in ds.dims:
        return [0]

    mu = ds.absorption.values
    n_records = ds.sizes["record"]
    valid_indices = []
    invalid_energy_indices: set[int] = set()

    energy_name = _energy_name(ds)
    energy_values = np.asarray(ds[energy_name].values)
    if energy_values.ndim == 2 and energy_values.shape[0] == n_records:
        reference_energy = energy_values[0]
        for i in range(n_records):
            delta = np.nanmax(np.abs(energy_values[i] - reference_energy))
            if delta > energy_tolerance:
                invalid_energy_indices.add(i)
                logger.warning("记录 %s 被剔除: 能量轴偏差 %.4f eV > %.4f eV。", i, delta, energy_tolerance)

    # Use median to be robust against outliers
    median_spec = np.nanmedian(mu, axis=0)

    # Handle NaNs in median
    if np.isnan(median_spec).any():
        median_spec = _fill_nans(median_spec)

    for i in range(n_records):
        if i in invalid_energy_indices:
            continue

        y = mu[i]
        if np.isnan(y).any():
            y = _fill_nans(y)

        try:
            corr = np.corrcoef(median_spec, y)[0, 1]
        except Exception:
            corr = 0.0

        if corr >= correlation_threshold:
            valid_indices.append(i)
        else:
            logger.warning("记录 %s 被剔除: 相关系数 %.4f < %s。", i, corr, correlation_threshold)

    logger.info("一致性检查: %s/%s 条记录通过。", len(valid_indices), n_records)
    return valid_indices


def merge_spectra(ds: xr.Dataset, indices: Optional[list[int]] = None, method: str = "average") -> xr.Dataset:
    """Merge multiple spectra into a single high-quality spectrum.

    Args:
        ds: Input Dataset (record x energy).
        indices: List of indices to merge. If None, uses all.
        method: 'average' (mean) or 'median'.

    Returns:
        New Dataset with 'record' dimension collapsed (or size 1).
    """
    if "record" not in ds.dims:
        return ds.copy()

    energy_name = _energy_name(ds)

    if indices is None:
        indices = list(range(ds.sizes["record"]))

    if not indices:
        raise ValueError("No indices provided for merging.")

    subset = ds.isel(record=indices)

    # Merge absorption
    mu = subset.absorption.values

    merged_mu = np.nanmedian(mu, axis=0) if method == "median" else np.nanmean(mu, axis=0)

    # Create new dataset
    # Preserve coordinates except record
    coords = dict(ds.coords)
    coords.pop("record", None)

    ds_out = xr.Dataset(data_vars={"absorption": (energy_name, merged_mu)}, coords=coords)

    # Copy attributes
    ds_out.attrs = ds.attrs.copy()
    ds_out.attrs["merged_scans_count"] = len(indices)

    return ds_out
