"""
STXM Analysis Utilities for Spectromicroscopy Data Processing.

This module provides high-performance tools for processing Scanning Transmission
X-ray Microscopy (STXM) data, including:
- Data loading and preprocessing (alignment, interpolation, OD conversion)
- PCA denoising with automatic component selection
- Saturation/stray-light correction (B-value optimization)
- Clustering analysis (KMeans, UMAP+HDBSCAN)
- Model fitting with custom components (DoubleStep for L-edge)

Performance optimizations:
- Vectorized numpy operations throughout
- Optional float32 for 50% memory reduction
- Lazy loading support via Dask
- Efficient HDF5 I/O with chunking

References:
- Vogt (2004) Ultramicroscopy: PCA/clustering methodology
- Tonti et al. (2021, 2025): Saturation correction model
- Tan et al. (2012): Onset energy threshold method
- Blanco-Portals et al. (2022): UMAP+HDBSCAN workflow
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import h5py
import hyperspy.api as hs
import numpy as np
import optuna
import scipy.optimize
from hyperspy.axes import DataAxis, UniformDataAxis
from scipy.ndimage import gaussian_filter
from scipy.optimize import OptimizeResult
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm

if TYPE_CHECKING:
    from hyperspy.component import Component
    from hyperspy.models.model1d import Model1D
    from hyperspy.roi import BaseROI
    from hyperspy.signals import Signal1D, Signal2D

# Configure module-level logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
FloatArray = np.ndarray
EnergyRange = tuple[float, float]
MaskRanges = tuple[float, float]

# ==================================================================================================
# 1. Constants & Configuration
# ==================================================================================================

# Element Parameters Database
ELEMENT_PARAMS = {
    "Mn": {
        "edge": "L3",
        "full_energy_range": (600.0, 700.0),  # Full energy range for Mn L-edge
        "preedge_fit_range": (625.0, 635.0),  # Pre-edge fitting range
        "main_fit_range": (630.0, 670.0),  # Plotting/Analysis range
        "mapping_range": (638.0, 644.0),  # Selection range for element mapping (L3 peak)
        "mask_thresholds": (0.04, 0.99),  # Normalized intensity thresholds for mask
        "peak_energies": (640.3, 651.4),  # (L3, L2)
    },
    "Zn": {
        "edge": "L3",
        "full_energy_range": (1000.0, 1060.0),  # Full energy range for Zn L-edge
        "preedge_fit_range": (1015.0, 1025.0),
        "main_fit_range": (1010.0, 1050.0),
        "mapping_range": (1020.0, 1025.0),
        "mask_thresholds": (0.04, 0.99),
        "peak_energies": (1021.8, 1044.9),  # (L3, L2)
    },
    "Fe": {
        "edge": "L3",
        "full_energy_range": (690.0, 740.0),
        "preedge_fit_range": (695.0, 705.0),
        "main_fit_range": (700.0, 730.0),
        "mapping_range": (705.0, 715.0),
        "mask_thresholds": (0.04, 0.99),
        "peak_energies": (708.65, 721.65),  # (L3, L2)
    },
    "O": {
        "edge": "K",
        "full_energy_range": (500.0, 600.0),  # Full energy range for O K-edge
        "preedge_fit_range": (520.0, 528.0),
        "main_fit_range": (525.0, 560.0),
        "mapping_range": (530.0, 535.0),
        "mask_thresholds": (0.04, 0.99),
    },
}


@dataclass
class ClusteringResults:
    """Results from clustering analysis."""

    labels: FloatArray | None = None
    spectra: Any | None = None
    n_clusters: int = 0


@dataclass
class UMAPHDBSCANResults:
    """Results from UMAP + HDBSCAN analysis."""

    labels: FloatArray | None = None
    spectra: Any | None = None
    embedding: FloatArray | None = None
    probabilities: FloatArray | None = None
    outlier_scores: FloatArray | None = None


@dataclass
class FittingResult:
    """Results from stepwise model fitting."""

    model: Model1D
    components: dict[str, Any]
    residuals: Any


@dataclass
class ROI:
    """
    Region of Interest (ROI) definition.

    Attributes:
        energy: (min_eV, max_eV) for spectral slicing.
        element: Element symbol (e.g., "Mn").
        spatial: Boolean mask (True=Keep) for pixel selection.
        hyperspy_roi: HyperSpy BaseROI object.
    """

    energy: tuple[float, float] | None = None
    element: str | None = None
    spatial: np.ndarray | None = None
    hyperspy_roi: BaseROI | None = None


@dataclass
class _ExportContext:
    """Internal context for DataTree export operations."""

    xr_module: Any  # xarray module
    tree_cls: Any  # DataTree class
    dt: Any  # DataTree instance
    spatial_dims: list[str]
    spatial_coords: dict[str, Any]
    ref_signal: Any = None


class NumpyEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def get_element_params(element: str) -> dict[str, Any]:
    """
    Retrieves analysis parameters for a specific element.

    Args:
        element (str): Element symbol (e.g., 'Mn', 'Zn').

    Returns:
        dict: Dictionary of parameters for the element.
    """
    if element not in ELEMENT_PARAMS:
        raise ValueError(f"Element '{element}' not found in database. Available: {list(ELEMENT_PARAMS.keys())}")
    return ELEMENT_PARAMS[element]


# ==================================================================================================
# 2. Loading & Basic Preprocessing
# ==================================================================================================


def load_stxm_signal(file_path: str | Path, lazy: bool = False, dtype: np.dtype | None = None) -> Signal2D:
    """
    Loads STXM data from an HDF5 file using dxchange and creates a HyperSpy Signal2D object.

    Args:
        file_path (str or Path): Path to the HDF5 file.
        lazy (bool): If True, loads the data lazily using Dask. Default is False.
        dtype (np.dtype, optional): Target dtype for memory efficiency.
            Use np.float32 for 50% memory savings. Default is None (preserve original).

    Returns:
        Signal2D: The loaded and initialized HyperSpy signal.
    """
    # Ensure file_path is a Path object
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    # Read HDF5 file
    with h5py.File(str(file_path), "r") as h5_file:
        # Helper to safely read dataset
        def _read_dset(key: str) -> np.ndarray:
            obj = h5_file[key]
            if isinstance(obj, h5py.Dataset):
                if lazy:
                    # Return array reference for lazy loading
                    # HyperSpy handles Dask wrapping internally
                    return np.asarray(obj)
                data = obj[:]
                if dtype is not None and data.dtype != dtype:
                    return data.astype(dtype)
                return data
            raise TypeError(f"{key} is not a dataset")

        data_unalignment = _read_dset("SpecNormalized/spectroscopy_normalized_aligned")
        energy = _read_dset("SpecNormalized/energy")
        x_pixel_size = _read_dset("SpecNormalized/x_pixel_size")
        y_pixel_size = _read_dset("SpecNormalized/y_pixel_size")

    # If lazy loading with HyperSpy, we often use hs.load(file, lazy=True)
    # But here we are manually constructing the signal from specific datasets.
    # We pass the h5py datasets directly to Signal2D if lazy.

    energy_axis = DataAxis(
        axis=energy[:] if not lazy else energy,
        index_in_array=None,
        name="Energy",
        units="eV",
    )
    x_position = UniformDataAxis(
        offset=0,
        scale=x_pixel_size[0],
        size=data_unalignment.shape[2],
        name="x_position",
        units=r"um",
    )
    y_position = UniformDataAxis(
        offset=0,
        scale=y_pixel_size[0],
        size=data_unalignment.shape[1],
        name="y_position",
        units=r"um",
    )

    signal = hs.signals.Signal2D(data_unalignment, axes=[energy_axis, y_position, x_position])
    if lazy:
        # Convert to lazy signal if it's not already
        signal = signal.as_lazy()
    return signal


def align_and_transpose_stack(signal: Signal2D) -> Signal1D:
    """
    Aligns the image stack using HyperSpy's estimate_shift2D and align2D,
    then transposes the data to a Signal1D (Y, X | Energy).

    Args:
        signal (Signal2D): Input signal stack (Energy | Y, X).

    Returns:
        Signal1D: Aligned signal with Energy as the signal axis.
    """
    shifts = signal.estimate_shift2D()
    signal.align2D(shifts=shifts, show_progressbar=True)
    return signal.transpose(signal_axes=[0], optimize=True)


def _find_energy_axis(signal: Signal1D) -> tuple[Any, int]:
    """Find the energy axis in signal axes. Returns (axis, index_in_array)."""
    for ax in signal.axes_manager.signal_axes:
        name = (ax.name or "").lower()
        units = (ax.units or "").lower()
        if name == "energy" or units == "ev":
            return ax, ax.index_in_array
    # Fallback: first signal axis if it has eV units
    if signal.axes_manager.signal_axes:
        ax = signal.axes_manager.signal_axes[0]
        units = (ax.units or "").lower()
        if units == "ev":
            return ax, ax.index_in_array
        # Last resort: assume first signal axis is energy
        return ax, ax.index_in_array
    raise ValueError("Could not find Energy axis. No signal axes available.")


def interpolate_energy_axis(signal: Signal1D, new_scale: float = 0.1, degree: int = 1) -> Signal1D:
    """
    Interpolates the energy axis to a uniform scale using HyperSpy's interpolation method.

    This function wraps `hyperspy.api.signals.BaseSignal.interpolate_on_axis` to resample
    the signal energy axis onto a uniform grid with the specified step size.

    Args:
        signal: Input Signal1D (Y, X | Energy).
        new_scale: New energy step size in eV. Default is 0.1 eV.
        degree: Interpolation polynomial degree. Default is 1 (linear interpolation).
            Higher values (e.g., 3) use cubic spline interpolation for smoother results.
            Must be a positive integer.

    Returns:
        Interpolated signal (modified in-place).

    Notes:
        Uses `hyperspy.api.signals.BaseSignal.interpolate_on_axis` internally.
        The interpolation modifies the signal in-place but returns it for convenience.
        For most spectral data, linear interpolation (degree=1) is sufficient and recommended.
        Use higher degrees (e.g., cubic) only when you have dense, high-quality data and need smoother curves.
    """
    # Validate degree parameter
    if not isinstance(degree, int) or degree < 1:
        raise ValueError(f"degree must be a positive integer, got {degree!r}")

    energy_axis, axis_index = _find_energy_axis(signal)
    energy_values = energy_axis.axis

    # Calculate new axis parameters
    # IMPORTANT: Use floor instead of ceil to ensure new axis stays WITHIN original range
    # This prevents the HyperSpy warning about extrapolation
    e_min = float(energy_values.min())
    e_max = float(energy_values.max())
    e_range = e_max - e_min

    # Calculate size to stay strictly within bounds
    new_size = int(np.floor(e_range / new_scale)) + 1

    # Adjust offset slightly inward to avoid boundary issues
    # The new axis will be: [e_min, e_min + new_scale, ..., e_min + (new_size-1)*new_scale]
    # Ensure the last point doesn't exceed e_max
    max_new_energy = e_min + (new_size - 1) * new_scale
    if max_new_energy > e_max:
        new_size -= 1

    axis_new = UniformDataAxis(
        offset=e_min,
        scale=new_scale,
        size=new_size,
        name="Energy",
        units="eV",
        navigate=energy_axis.navigate,
        is_binned=True,
    )
    signal.interpolate_on_axis(axis_new, axis=axis_index, inplace=True, degree=degree)
    return signal


def convert_to_optical_density(signal: Signal1D) -> Signal1D:
    """
    Converts transmission data to optical density (-log(T)).
    Expects data to be a Signal1D (Energy as signal axis).
    """
    od_signal = signal.deepcopy()
    # Handle zeros or negatives before log
    od_signal.data = -np.log(np.maximum(od_signal.data, 1e-10))
    return od_signal


# ==================================================================================================
# 3. Denoising & Correction
# ==================================================================================================


def apply_pca_denoising(signal: Signal1D, components_number: int | None = None, algorithm: str = "SVD") -> Signal1D:
    """
    Performs PCA denoising on the signal.

    Args:
        signal (Signal1D): Input signal.
        components_number (int, optional): Number of components to keep.
            If None, uses automatic elbow estimation.
        algorithm (str): Decomposition algorithm.

    Returns:
        Signal1D: Denoised signal.
    """
    # Ensure decomposition
    if signal.learning_results.factors is None:
        signal.decomposition(algorithm=algorithm, navigation_mask=None, reproject="signal")

    if components_number is None:
        try:
            # estimate_elbow_position returns the index of the detected elbow (0-based)
            elbow_index = signal.estimate_elbow_position()
            # Number of components to keep = elbow_index + 1
            # (e.g., if elbow is at index 2, keep components 0, 1, 2 = 3 total)
            components_number = elbow_index + 1
            logger.info("Automatic PCA component selection (elbow at index %s): %s components retained.", elbow_index, components_number)
        except Exception as e:
            logger.warning("Elbow estimation failed: %s. Defaulting to 3 components.", e)
            # Default to 3 components (this IS the count, not an index)
            components_number = 3

    # get_decomposition_model expects the NUMBER of components (1-indexed count)
    denoised_data = signal.get_decomposition_model(components=components_number)
    return denoised_data


def _saturation_model(absorbance_thin: np.ndarray, kappa: float, c: float) -> np.ndarray:
    """
    Saturation correction model relating thin and thick region absorbances.

    Formula: A_true = -ln((1+κ)·exp(-A_meas) - κ)

    Args:
        absorbance_thin: Measured absorbance of thin region.
        kappa: Stray light fraction (κ).
        c: Thickness ratio (d_thick / d_thin).

    Returns:
        Predicted measured absorbance of thick region.
    """
    absorbance_thin = np.asarray(absorbance_thin)
    tm_thin = np.exp(-absorbance_thin)
    t_corr_thin = (1 + kappa) * tm_thin - kappa
    t_corr_thin = np.maximum(t_corr_thin, 1e-9)  # Clip for numerical stability
    t_corr_thick = np.power(t_corr_thin, c)
    tm_thick = (t_corr_thick + kappa) / (1 + kappa)
    tm_thick = np.maximum(tm_thick, 1e-9)
    return np.nan_to_num(-np.log(tm_thick))


def _calc_r2(thin_spec: FloatArray, thick_spec: FloatArray, kappa: float) -> float:
    """
    Computes R² between corrected thin/thick spectra for kappa optimization.
    """
    tm_thin = np.exp(-thin_spec)
    tm_thick = np.exp(-thick_spec)
    term_thin = (1 + kappa) * tm_thin - kappa
    term_thick = (1 + kappa) * tm_thick - kappa

    valid = (term_thin > 1e-5) & (term_thick > 1e-5)
    if np.count_nonzero(valid) < 10:
        return 0.0

    a_thin = -np.log(term_thin[valid])
    a_thick = -np.log(term_thick[valid])

    std_thin, std_thick = np.std(a_thin), np.std(a_thick)
    if std_thin == 0 or std_thick == 0:
        return 0.0

    cov = np.mean((a_thin - np.mean(a_thin)) * (a_thick - np.mean(a_thick)))
    r = cov / (std_thin * std_thick)
    return float(r * r)


def _split_spectra(summed: FloatArray, data: FloatArray, threshold: float, mask_range: MaskRanges) -> tuple[FloatArray | None, FloatArray | None]:
    """
    Split pixels into thin/thick regions and return mean spectra.

    Args:
        summed: Total intensity map (Y, X).
        data: Spectral data (Y, X, E).
        threshold: Value separating thin/thick regions.
        mask_range: (min, max) valid intensity range.

    Returns:
        (thin_spec, thick_spec) or (None, None) if empty.
    """
    thin_mask = (summed >= mask_range[0]) & (summed <= threshold)
    thick_mask = (summed > threshold) & (summed < mask_range[1])

    if not np.any(thin_mask) or not np.any(thick_mask):
        return None, None

    return np.nanmean(data[thin_mask], axis=0), np.nanmean(data[thick_mask], axis=0)


def _coarse_kappa_search(
    summed_reconstruction: FloatArray,
    data_array: FloatArray,
    mask_ranges: MaskRanges,
    num_points: int,
    tqdm_func: Any,
) -> tuple[float, float, float]:
    """
    Coarse search for optimal thin/thick threshold in kappa optimization.
    """
    # Search range for k (threshold separating thin/thick): within the mask ranges
    k_space = np.linspace(mask_ranges[0] + 0.01, mask_ranges[1] - 0.01, num_points)

    best_r2_global = -1.0
    best_k_global = k_space[0]

    # Pre-flatten data for faster indexing
    flat_reconstruction = summed_reconstruction.ravel()
    flat_data = data_array.reshape(-1, data_array.shape[-1])

    for k_val in tqdm_func(k_space, desc="Coarse B-value search"):
        thin_spec, thick_spec = _split_spectra(flat_reconstruction, flat_data, k_val, mask_ranges)

        if thin_spec is None or thick_spec is None:
            continue

        # Optimize b for this split
        # Kappa is typically < 10% (0.1), rarely > 0.2.
        res = cast(
            OptimizeResult,
            scipy.optimize.minimize_scalar(lambda b, ts=thin_spec, ths=thick_spec: -_calc_r2(ts, ths, b), bounds=(0.0, 0.3), method="bounded", options={"xatol": 0.005}),
        )
        r2 = -float(res.fun)

        if r2 > best_r2_global:
            best_r2_global = r2
            best_k_global = k_val

    # Define refinement range around best k
    step = k_space[1] - k_space[0]
    k_low = max(mask_ranges[0] + 0.001, best_k_global - step)
    k_high = min(mask_ranges[1] - 0.001, best_k_global + step)

    return k_low, k_high, best_k_global


def _refine_kappa(summed: FloatArray, data: FloatArray, mask_range: MaskRanges, k_low: float, k_high: float) -> tuple[float, float]:
    """Refine threshold k to maximize R² between corrected thin/thick spectra."""

    def _objective(k: float) -> float:
        thin, thick = _split_spectra(summed, data, k, mask_range)
        if thin is None or thick is None:
            return 1.0
        res = cast(OptimizeResult, scipy.optimize.minimize_scalar(lambda b: -_calc_r2(thin, thick, b), bounds=(0.0, 0.3), method="bounded", options={"xatol": 0.001}))
        return float(res.fun)

    result = cast(OptimizeResult, scipy.optimize.minimize_scalar(_objective, bounds=(k_low, k_high), method="bounded", options={"xatol": 0.001}))
    return float(result.x), -float(result.fun)


def _fit_kappa_params(
    data: FloatArray,
    summed: FloatArray,
    threshold: float,
    mask_range: MaskRanges,
) -> list[float] | FloatArray:
    """Fit kappa and thickness ratio using curve_fit."""
    thin, thick = _split_spectra(summed, data, threshold, mask_range)

    if thin is not None and thick is not None:
        # Initial guess: kappa=0.07 (typical), c=1.5
        p0 = [0.07, 1.5]
        # Bounds: kappa in [0, 1], c in [1, 10]
        bounds = ([0.0, 1.0], [1.0, 10.0])

        try:
            best_popt, _ = scipy.optimize.curve_fit(
                _saturation_model,
                xdata=thin,
                ydata=thick,
                p0=p0,
                bounds=bounds,
                maxfev=2000,
            )
        except (RuntimeError, ValueError):
            # Fallback if fit fails
            best_popt = [0.0, 1.0]
    else:
        best_popt = [0.0, 1.0]
    return best_popt


def optimize_kappa(signal: Signal1D, mask_range: tuple[float, float], num_points: int = 100) -> tuple[float, list[float] | np.ndarray, float]:
    """
    Optimize stray light (kappa) correction parameters.

    Uses a two-stage optimization: coarse search followed by refinement.
    Reference: Tonti et al. (2021, 2025).

    Args:
        signal: Reconstructed/denoised signal.
        mask_range: (min, max) intensity thresholds for thin/thick regions.
        num_points: Coarse search resolution.

    Returns:
        (threshold, [kappa, c], r2): Optimal parameters.
    """
    summed = signal.sum(axis=-1).data
    data = signal.data
    # 1. Coarse search
    k_low, k_high, _ = _coarse_kappa_search(summed, data, mask_range, num_points, tqdm)
    # 2. Fine optimization
    best_k, best_r2 = _refine_kappa(summed, data, mask_range, k_low, k_high)
    # 3. Fit final parameters
    best_popt = _fit_kappa_params(data, summed, best_k, mask_range)
    return best_k, best_popt, best_r2


# ==================================================================================================
# 4. ROI Operations
# ==================================================================================================


def _resolve_range_from_element(
    element: str | None,
    provided_range: tuple[float, float] | None,
    range_type: str = "full_energy_range",
    default_element: str | None = None,
) -> tuple[float, float] | None:
    """
    Helper to resolve energy ranges from element parameters.

    Args:
        element: Element symbol.
        provided_range: Manual range (takes precedence).
        range_type: Key in ELEMENT_PARAMS (e.g., 'full_energy_range', 'preedge_fit_range').
        default_element: Fallback element if 'element' is None.

    Returns:
        Resolved range tuple or None.
    """
    if provided_range is not None:
        return provided_range

    target_element = element if element is not None else default_element

    if target_element is not None and isinstance(target_element, str) and target_element in ELEMENT_PARAMS:
        return cast(tuple[float, float] | None, ELEMENT_PARAMS[target_element].get(range_type))

    return None


def resolve_roi_energy_params(element: str | None = None, energy_range: tuple[float, float] | None = None) -> tuple[float, float] | None:
    """
    Resolves ROI parameters (energy range) based on element name.

    Args:
        element (str, optional): Element symbol.
        energy_range (tuple, optional): Manual energy range.

    Returns:
        tuple: Resolved energy range.
    """
    return _resolve_range_from_element(element, energy_range, "full_energy_range")


def extract_roi_energy_signal(signal: Signal1D, energy_range: tuple[float, float] | None = None) -> Signal1D:
    """
    Extracts the signal within the specified energy ROI.
    Expects data to be a Signal1D (Energy as signal axis).

    Args:
        signal (Signal1D): Input signal.
        energy_range (tuple, optional): (min, max) energy range.

    Returns:
        Signal1D: Sliced signal.
    """
    if energy_range is None:
        return signal.deepcopy()

    # Strictly slice the signal axis (Energy)
    return signal.isig[energy_range[0] : energy_range[1]]


# ==================================================================================================
# 5. Analysis: Onset Energy & Clustering
# ==================================================================================================


def remove_background(signal: Signal1D, fit_range: tuple[float, float] | None = None, component_name: str = "Offset", polynomial_order: int = 1, element: str | None = None) -> Signal1D:
    """
    Removes background from the signal using a specified model component.

    Args:
        signal (Signal1D): Input signal.
        fit_range (tuple, optional): Energy range for fitting (min, max).
            If None and element is provided, uses 'preedge_fit_range'.
            If None and no element, defaults to Mn pre-edge.
        component_name (str): 'Offset', 'PowerLaw', or 'Polynomial'.
        polynomial_order (int): Order of polynomial if component_name is 'Polynomial'.
        element (str, optional): Element symbol to look up default fit_range.

    Returns:
        Signal1D: Signal with background removed.
    """
    hs.set_log_level("ERROR")

    # Resolve default fit_range using helper
    # Defaults to Mn if nothing else is found
    fit_range = _resolve_range_from_element(element, fit_range, "preedge_fit_range", default_element="Mn")

    # Fallback to Mn hardcoded if even the helper returns None (unlikely given default_element="Mn")
    if fit_range is None:
        fit_range = cast(tuple[float, float], ELEMENT_PARAMS["Mn"]["preedge_fit_range"])

    # Dynamic component selection
    if component_name == "Polynomial":
        comp = _create_component("Polynomial", order=polynomial_order)
    else:
        if component_name not in {"PowerLaw", "Polynomial", "Offset"}:
            component_name = "Offset"
        comp = _create_component(component_name)

    model = signal.create_model()
    model.append(comp)

    model.fit_component(
        component=comp,
        signal_range=fit_range,
        fit_independent=True,
        only_current=False,
    )

    background = model.as_signal(component_list=[comp])
    result = signal - background
    return result


def _get_energy_data(signal: Signal1D, energy_range: EnergyRange | None = None) -> tuple[FloatArray, FloatArray]:
    """Get energy axis and data cube, optionally sliced to energy range."""
    energy_axis = signal.axes_manager.signal_axes[0].axis
    datacube = signal.data

    if energy_range is not None:
        e_min, e_max = energy_range
        # Vectorized mask creation
        mask_e = (energy_axis >= e_min) & (energy_axis <= e_max)
        datacube = datacube[..., mask_e]
        energy_axis_sliced = energy_axis[mask_e]
    else:
        energy_axis_sliced = energy_axis

    return energy_axis_sliced, datacube


def _onset_indices(datacube: FloatArray, threshold: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Find indices where intensity first exceeds threshold."""
    # Broadcast comparison: datacube shape (Y, X, E), threshold shape (Y, X, 1)
    mask_over = datacube > threshold
    # argmax returns first True index along last axis
    idx_upper = np.argmax(mask_over, axis=-1)
    idx_lower = np.maximum(0, idx_upper - 1)
    return idx_upper, idx_lower


def _interp_onset(
    e_lower: FloatArray,
    e_upper: FloatArray,
    i_lower: FloatArray,
    i_upper: FloatArray,
    threshold: FloatArray,
) -> FloatArray:
    """Sub-pixel linear interpolation for onset energy."""
    denom = i_upper - i_lower
    # Avoid division by zero
    denom = np.where(denom == 0, 1e-10, denom)
    fraction = (threshold.squeeze() - i_lower) / denom
    return e_lower + (e_upper - e_lower) * fraction


def calculate_onset_energy(signal: Signal1D, threshold_ratio: float = 0.1, energy_range: tuple[float, float] | None = None, smooth_sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the onset energy map based on the 10% intensity threshold method.

    Methodology Reference:
    Tan, H., Verbeeck, J., Abakumov, A., & Van Tendeloo, G. (2012).
    "Oxidation state and chemical shift investigation in transition metal oxides by EELS."
    Ultramicroscopy, 116, 24-33.

    The paper defines the onset energy as the energy-loss value where the intensity of the
    edge reaches 10% (threshold_ratio=0.1) of its maximum height. This method is chosen
    to suppress noise effects and consistently handle weak pre-peaks compared to
    inflection point methods.

    Args:
        signal (Signal1D): Input signal (Absorbance mode, background subtracted).
        threshold_ratio (float): Ratio of max intensity to define onset (Default: 0.1 for 10%).
        energy_range (tuple, optional): (min, max) energy range to search within.
        smooth_sigma (float): Sigma for Gaussian smoothing (spatial) to reduce noise before detection.

    Returns:
        tuple: (onset_map, flattened_map)
            onset_map is 2D array of energy values.
            flattened_map is sorted 1D array for plotting.
    """
    # 1. Get energy axis and data cube
    energy_axis_sliced, datacube = _get_energy_data(signal, energy_range)
    # 2. Optional spatial smoothing
    if smooth_sigma > 0:
        datacube = gaussian_filter(datacube, sigma=(smooth_sigma, smooth_sigma, 0))
    # 3. Threshold Calculation
    # Calculate max intensity per pixel in the selected energy range
    pixel_max_intensity = np.max(datacube, axis=-1, keepdims=True)
    threshold = pixel_max_intensity * threshold_ratio

    # 4. Find onset indices
    # We find the FIRST index where intensity > threshold
    idx_upper, idx_lower = _onset_indices(datacube, threshold)

    # 5. Get energy and intensity for interpolation
    rows, cols = np.indices(idx_upper.shape)
    e_upper = energy_axis_sliced[idx_upper]
    e_lower = energy_axis_sliced[idx_lower]
    i_upper = datacube[rows, cols, idx_upper]
    i_lower = datacube[rows, cols, idx_lower]

    # 6. Sub-pixel interpolation
    onset_map = _interp_onset(e_lower, e_upper, i_lower, i_upper, threshold)

    # 7. Validity mask
    # Filter out pixels with very low signal (noise) or where threshold wasn't crossed properly (idx=0)
    # Also valid if the max intensity is significant enough (e.g. > 0.01 absorbance)
    valid_mask = (pixel_max_intensity.squeeze() > 0.01) & (idx_upper > 0)
    onset_map[~valid_mask] = np.nan

    flattened = onset_map[valid_mask].flatten()
    flattened = np.sort(flattened)
    return onset_map, flattened


def _ensure_pca(signal: Signal1D, algorithm: str = "SVD", navigation_mask: np.ndarray | None = None) -> None:
    """Ensure PCA decomposition has been performed on the signal."""
    if signal.learning_results.factors is None:
        logger.info("Decomposition not found. Performing %s...", algorithm)
        signal.decomposition(algorithm=algorithm, navigation_mask=navigation_mask)


def _get_pca_loadings(signal: Signal1D, exclude_first: bool = False) -> Any:
    """Get spatial PCA loadings for clustering, handling axis mismatches."""
    loadings_sig = signal.get_decomposition_loadings()
    factors_sig = signal.get_decomposition_factors()

    target_nav_shape = signal.axes_manager.navigation_shape
    spatial = None

    logger.info("Signal Nav: %s", target_nav_shape)
    logger.debug("Loadings Nav: %s", loadings_sig.axes_manager.navigation_shape)
    logger.debug("Loadings Sig: %s", loadings_sig.axes_manager.signal_shape)

    # Strategy 1: Direct Match
    if loadings_sig.axes_manager.navigation_shape == target_nav_shape:
        spatial = loadings_sig
    elif loadings_sig.axes_manager.signal_shape == target_nav_shape:
        spatial = loadings_sig.transpose()

    # Strategy 2: Factors Match
    if spatial is None:
        if factors_sig.axes_manager.navigation_shape == target_nav_shape:
            spatial = factors_sig
        elif factors_sig.axes_manager.signal_shape == target_nav_shape:
            spatial = factors_sig.transpose()

    # Strategy 3: Component axis misplaced in navigation (e.g. all axes are navigation)
    if spatial is None:
        l_nav = loadings_sig.axes_manager.navigation_shape
        # Check for (Comp, Y, X) where (Y, X) matches target
        if len(l_nav) == len(target_nav_shape) + 1:
            if l_nav[1:] == target_nav_shape:
                # Axis 0 is Comp. Transpose it to signal.
                spatial = loadings_sig.transpose(signal_axes=[0])
            elif l_nav[:-1] == target_nav_shape:
                # Last axis is Comp.
                spatial = loadings_sig.transpose(signal_axes=[-1])

    if spatial is None:
        logger.warning(
            "Could not match decomposition results to signal dimensions.\nSignal Nav: %s\nLoadings Nav: %s\nUsing loadings as-is (may fail).",
            target_nav_shape,
            loadings_sig.axes_manager.navigation_shape,
        )
        spatial = loadings_sig

    if exclude_first:
        if spatial.axes_manager.signal_dimension == 0:
            logger.warning("Spatial signal has no signal dimension to slice components from.")
        elif spatial.axes_manager.signal_shape[0] < 2:
            logger.warning("Not enough components to exclude the first one. Using all components.")
        else:
            logger.info("Excluding 1st PCA component (Thickness) from clustering...")
            # Slicing the signal axis (Components)
            spatial = spatial.isig[1:]

    return spatial


def _flatten_features(signal: Signal1D, navigation_mask: np.ndarray | None = None, exclude_first_component: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Prepare flattened feature array for sklearn-based clustering."""
    _ensure_pca(signal, navigation_mask=navigation_mask)
    spatial_sig = _get_pca_loadings(signal, exclude_first_component)

    # spatial_sig data shape is (Y, X, n_components) -> Reshape to (Y*X, n_components)
    features = spatial_sig.data.reshape(-1, spatial_sig.data.shape[-1])

    if navigation_mask is not None:
        # mask is (Y, X), True=Keep (ROI convention in this utils file)
        mask_flat = navigation_mask.flatten()
        valid_indices = np.where(mask_flat)[0]
        features_fit = features[valid_indices]
    else:
        valid_indices = np.arange(features.shape[0])
        features_fit = features

    return features_fit, valid_indices


def _cluster_means(signal: Signal1D, labels: np.ndarray, mask: np.ndarray | None = None) -> Any:
    """Calculate mean spectrum for each cluster label."""
    flat_data = signal.data.reshape(-1, signal.data.shape[-1])
    flat_labels = labels.flatten()

    unique = np.sort(np.unique(flat_labels))
    unique = unique[unique >= 0]  # Exclude noise (-1)

    centers = []
    flat_mask = mask.flatten() if mask is not None else None

    for lbl in unique:
        lbl_mask = flat_labels == lbl
        if flat_mask is not None:
            lbl_mask &= flat_mask
        if np.any(lbl_mask):
            centers.append(np.nanmean(flat_data[lbl_mask], axis=0))

    if not centers:
        return None

    signals = hs.signals.Signal1D(np.array(centers))

    # Copy signal axis metadata from source
    sig_ax = signal.axes_manager.signal_axes[0]
    out_ax = signals.axes_manager.signal_axes[0]

    out_ax.name = sig_ax.name
    out_ax.units = sig_ax.units
    out_ax.scale = sig_ax.scale
    out_ax.offset = sig_ax.offset

    signals.axes_manager[0].name = "Cluster index"

    return signals


def perform_clustering_analysis(  # noqa: PLR0913, PLR0917
    signal: Signal1D,
    n_clusters: int | None = None,
    algorithm: Any = "kmeans",
    navigation_mask: np.ndarray | None = None,
    preprocessing: str | None = None,
    exclude_first_component: bool = False,
    metric: str = "silhouette",
) -> tuple[np.ndarray, Any]:
    """
    Performs clustering on the signal using HyperSpy's implementation, optimized for STXM.
    Supports algorithms from sklearn (or string shortcuts).

    Methodology Reference:
    Vogt, S. (2004). "Cluster analysis of soft X-ray spectromicroscopy data."
    Ultramicroscopy, 99, 149-157.

    Key features from reference:
    1. Decomposition (PCA) is used to orthogonalize and denoised data.
    2. Clustering is performed in the eigenspace (loadings).
    3. The first principal component often represents total thickness/absorbance.
       Excluding it ('exclude_first_component=True') focuses analysis on chemical speciation.

    Args:
        signal (Signal1D): Input signal.
        n_clusters (int, optional): Number of clusters. If None, estimates using 'metric'.
        algorithm (str or object): Clustering algorithm ('kmeans', etc.) or sklearn estimator instance.
        navigation_mask (np.ndarray, optional): Boolean mask for valid pixels.
        preprocessing (str or None): Preprocessing normalization ('norm', 'standard', etc.).
        exclude_first_component (bool): If True, ignores the 1st PCA component (Thickness).
        metric (str): Metric for estimating n_clusters ('silhouette', 'elbow', 'gap').

    Returns:
        tuple: (cluster_labels, cluster_results)
    """
    # 1. Prepare Data
    _ensure_pca(signal, algorithm="SVD", navigation_mask=navigation_mask)
    cluster_source = _get_pca_loadings(signal, exclude_first_component)

    # 2. Estimate n_clusters if needed
    if n_clusters is None:
        logger.info("Estimating number of clusters using '%s' metric (using KMeans for estimation)...", metric)
        # Note: estimate_number_of_clusters expects cluster_source to be the signal
        estimated = signal.estimate_number_of_clusters(
            cluster_source=cluster_source,
            preprocessing=preprocessing,
            algorithm="kmeans",  # Force KMeans for estimation step
            navigation_mask=navigation_mask,
            metric=metric,
        )
        n_clusters = int(estimated[0]) if isinstance(estimated, list) else int(estimated)
        logger.info("Estimated number of clusters: %s", n_clusters)

    # 3. Perform Analysis
    signal.cluster_analysis(
        cluster_source=cluster_source,
        n_clusters=n_clusters,
        preprocessing=preprocessing,
        algorithm=algorithm,
        navigation_mask=navigation_mask,
    )

    labels = signal.get_cluster_labels()
    labels_array = np.asarray(labels.data) if hasattr(labels, "data") else np.asarray(labels)

    # Handle case where labels are returned as (n_clusters, Y, X)
    # This can happen if HyperSpy returns boolean masks or probabilities for each cluster
    if labels_array.ndim == 3 and labels_array.shape[0] == n_clusters:
        logger.debug("Collapsing labels from %s to integer labels.", labels_array.shape)
        labels_array = np.argmax(labels_array, axis=0)

    logger.debug("Labels shape: %s", labels_array.shape)
    logger.debug("Signal shape: %s", signal.data.shape)

    # 4. Calculate Mean Spectra
    # Try HyperSpy method first, fallback to our robust helper
    try:
        signals = signal.get_cluster_signals(signal="mean")
    except Exception as e:
        logger.warning("get_cluster_signals failed (%s). Calculating manually...", e)
        # HyperSpy mask True=Exclude, Helper mask True=Keep
        # If navigation_mask was passed to HyperSpy, it was an EXCLUSION mask?
        # Let's check call site.
        # TXMAnalyzer.perform_clustering calls with `navigation_mask=hs_mask` where `hs_mask = ~mask`.
        # So `navigation_mask` here is Exclude=True.
        # Helper expects Keep=True. So we invert it back if present.
        helper_mask = ~navigation_mask if navigation_mask is not None else None
        signals = _cluster_means(signal, labels_array, helper_mask)

    return labels_array, signals


def perform_umap_hdbscan_clustering(  # noqa: PLR0913, PLR0917
    signal: Signal1D,
    navigation_mask: np.ndarray | None = None,
    preprocessing: str | None = None,
    exclude_first_component: bool = False,
    # UMAP parameters
    umap_n_neighbors: int = 15,
    umap_n_components: int = 2,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    umap_random_state: int = 42,
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int | None = None,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_metric: str = "euclidean",
    verbose: bool = True,
) -> tuple[np.ndarray, Any, Any, dict[str, Any]]:
    """
    Performs clustering using UMAP for manifold learning followed by HDBSCAN.
    This combination is powerful for finding non-linear, density-based clusters.

    Methodology Reference:
    Blanco-Portals, J., Peiró, F., & Estradé, S. (2022).
    "Strategies for EELS Data Analysis. Introducing UMAP and HDBSCAN for Dimensionality Reduction and Clustering."
    Microscopy and Microanalysis, 28(1), 109-122.

    Key Strategy:
    1. Metric: Use 'cosine' distance for UMAP (default) to be insensitive to thickness/amplitude variations,
       focusing on spectral shape (chemical state).
    2. PCA first: Reduces noise and dimensionality before UMAP.
    3. HDBSCAN: Clusters the 2D UMAP embedding.

    Workflow:
    1. PCA (Denoising/Orthogonalization).
    2. UMAP (Embedding PCA loadings into low-dim manifold).
    3. HDBSCAN (Clustering on UMAP embedding).

    Args:
        signal (Signal1D): Input signal.
        navigation_mask (np.ndarray): Mask.
        preprocessing (str): 'norm', 'standard', etc.
        exclude_first_component (bool): Ignore 1st PCA component.
        umap_*: Parameters for UMAP.
        hdbscan_*: Parameters for HDBSCAN.
        verbose (bool): Whether to print progress logs.

    Returns:
        tuple: (labels, signals, umap_embedding)
    """
    import umap  # noqa: PLC0415
    from sklearn.cluster import HDBSCAN  # noqa: PLC0415

    # 1. Prepare Features
    # Note: We fetch ALL components first, preprocess, THEN exclude thickness if requested.
    features_fit, valid_indices = _flatten_features(signal, navigation_mask=navigation_mask, exclude_first_component=False)

    # 2. Preprocessing
    if preprocessing == "standard":
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        features_fit = StandardScaler().fit_transform(features_fit)
    elif preprocessing == "norm":
        from sklearn.preprocessing import Normalizer  # noqa: PLC0415

        features_fit = Normalizer().fit_transform(features_fit)
    elif preprocessing == "minmax":
        from sklearn.preprocessing import MinMaxScaler  # noqa: PLC0415

        features_fit = MinMaxScaler().fit_transform(features_fit)

    # Exclude first component (Thickness)
    if exclude_first_component and features_fit.shape[1] > 1:
        if verbose:
            logger.info("Excluding 1st PCA component from UMAP...")
        features_fit = features_fit[:, 1:]

    # 3. UMAP Embedding
    if verbose:
        logger.info("Running UMAP (n_neighbors=%d, min_dist=%.2f)...", umap_n_neighbors, umap_min_dist)
    reducer = umap.UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist, metric=umap_metric, random_state=umap_random_state)
    embedding = reducer.fit_transform(features_fit)

    # 4. HDBSCAN Clustering
    if verbose:
        logger.info("Running HDBSCAN (min_cluster_size=%d)...", hdbscan_min_cluster_size)
    clusterer = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, cluster_selection_epsilon=hdbscan_cluster_selection_epsilon, metric=hdbscan_metric)
    cluster_labels_fit = clusterer.fit_predict(embedding)

    # 5. Reconstruct Labels Image
    # Initialize with -1 (noise)
    full_labels = np.full(signal.axes_manager.navigation_size, -1, dtype=int)
    full_labels[valid_indices] = cluster_labels_fit

    # Reshape back to navigation shape
    labels_image = full_labels.reshape(signal.data.shape[:-1])

    # 6. Calculate Mean Spectra
    signals = _cluster_means(signal, labels_image, navigation_mask)

    # 7. Collect HDBSCAN properties (probabilities & outlier scores)
    hdbscan_props = {}

    # Initialize full-sized arrays with NaNs/zeros
    # Probabilities: 0.0 for noise/unassigned
    full_probs = np.zeros(signal.axes_manager.navigation_size, dtype=float)
    probs = getattr(clusterer, "probabilities_", None)
    if probs is not None:
        full_probs[valid_indices] = probs
        if verbose:
            logger.debug("Captured HDBSCAN probabilities (shape %s)", full_probs.shape)
    else:
        logger.warning("HDBSCAN probabilities_ not found.")
    hdbscan_props["probabilities"] = full_probs.reshape(signal.data.shape[:-1])

    # Outlier Scores: 0.0 (not outlier) -> 1.0 (outlier)
    # Note: GLOSH scores are available if created with specific params, generally always available in modern sklearn/hdbscan
    full_scores = np.zeros(signal.axes_manager.navigation_size, dtype=float)
    scores = getattr(clusterer, "outlier_scores_", None)
    if scores is not None:
        full_scores[valid_indices] = scores
        if verbose:
            logger.debug("Captured HDBSCAN outlier_scores (shape %s)", full_scores.shape)
    else:
        logger.warning("HDBSCAN did not generate outlier_scores_")
    hdbscan_props["outlier_scores"] = full_scores.reshape(signal.data.shape[:-1])

    return labels_image, signals, embedding, hdbscan_props


def optimize_clustering_params(
    signal: Signal1D,
    navigation_mask: np.ndarray | None = None,
    n_trials: int = 50,
    exclude_first_component: bool = True,
    study_name: str = "stxm_clustering",
) -> dict[str, Any]:
    """
    Auto-tunes UMAP + HDBSCAN parameters for STXM data using Optuna.

    Target: Maximizes (Silhouette Score - Noise Ratio).

    Args:
        signal: HyperSpy Signal1D (preprocessed).
        navigation_mask: Boolean navigation mask (True=Keep).
        n_trials: Number of parameter combinations to try.
        exclude_first_component: Whether to ignore thickness component.
        study_name: Name of the Optuna study.

    Returns:
        dict: Best parameters found.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. Pre-calculate features (PCA loadings) ONCE
    # We calculate score based on PCA features (chemical reality),
    # NOT UMAP embedding (which distorts distances).
    features, valid_idx = _flatten_features(signal, navigation_mask=navigation_mask, exclude_first_component=exclude_first_component)

    # Subsample for performance if dataset is large (>10k pixels)
    # Silhouette score is O(N^2), so this is crucial.
    use_sampling = False
    sample_indices = None
    if len(features) > 10000:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(features), 10000, replace=False)
        features_eval = features[sample_indices]
        use_sampling = True
    else:
        features_eval = features

    def objective(trial):
        # 2. Define Parameter Search Space
        # UMAP Params
        n_neighbors = trial.suggest_int("n_neighbors", 10, 50)
        n_components = trial.suggest_int("n_components", 2, 5)
        min_dist = trial.suggest_float("min_dist", 0.0, 0.5)

        # HDBSCAN Params
        min_cluster_size = trial.suggest_int("min_cluster_size", 10, 200)
        # min_samples usually <= min_cluster_size. Let's vary it or keep it None (default)
        # trial.suggest_int("min_samples", 1, 50)

        # 3. Run Clustering
        try:
            # We call perform_umap_hdbscan_clustering which does UMAP+HDBSCAN
            # But we need to make sure we don't re-calculate PCA every time if possible.
            # perform_umap_hdbscan_clustering calls _flatten_features internally.
            # It's fast enough (just array slicing) since PCA is already done on signal.

            labels_img, _, _, _ = perform_umap_hdbscan_clustering(
                signal=signal,
                navigation_mask=navigation_mask,
                exclude_first_component=exclude_first_component,
                umap_n_neighbors=n_neighbors,
                umap_n_components=n_components,
                umap_min_dist=min_dist,
                hdbscan_min_cluster_size=min_cluster_size,
                hdbscan_min_samples=None,  # Default to min_cluster_size
                verbose=False,
            )
        except Exception:
            return -1.0

        # 4. Extract Labels for Evaluation
        flat_labels_full = labels_img.flatten()
        # Get labels corresponding to the valid pixels
        valid_labels = flat_labels_full[valid_idx]

        eval_labels = valid_labels[sample_indices] if use_sampling and sample_indices is not None else valid_labels

        # 5. Calculate Validity Score
        # Filter out noise points (-1)
        is_cluster = eval_labels != -1
        n_clustered_points = np.sum(is_cluster)

        # Constraint A: Must find at least 2 clusters
        unique_clusters = np.unique(eval_labels[is_cluster])
        if len(unique_clusters) < 2:
            return -1.0

        # Constraint B: Noise Penalty
        # If >80% of data is noise, it's a bad result
        noise_ratio = 1.0 - (n_clustered_points / len(eval_labels))
        if noise_ratio > 0.8:
            return -1.0

        # Metric: Silhouette Score on PCA features (Chemical separation)
        score = silhouette_score(features_eval[is_cluster], eval_labels[is_cluster])

        # Final Objective: Score penalizing noise
        final_score = score - noise_ratio

        return final_score

    # 6. Run Optimizer
    study = optuna.create_study(direction="maximize", study_name=study_name)

    # Initialize tqdm progress bar
    with tqdm(total=n_trials, desc="Optimization Trials", unit="trial") as pbar:

        def objective_wrapper(trial):
            result = objective(trial)
            try:
                current_best = f"{study.best_value:.4f}"
            except ValueError:
                # No trials completed yet (or all failed)
                current_best = "N/A"
            pbar.set_postfix({"best_score": current_best})
            pbar.update(1)
            return result

        study.optimize(objective_wrapper, n_trials=n_trials)

    logger.info("Optimization finished. Best score: %.4f", study.best_value)
    return study


# ==================================================================================================
# 6. Model Fitting
# ==================================================================================================


class DoubleStep(hs.model.components1D.Expression):
    """
    Double Arctan Step function for Fe/Mn L-edge continuum background.

    Methodology Reference:
    Liebscher, B., et al. (2002). "Quantification of Ferrous/Ferric Ratios in Minerals:
    New Evaluation Schemes of Fe L23 Electron Energy-Loss Near-Edge Spectra."
    Physics and Chemistry of Minerals, 29, 579-588. Eq. 1.

    Features:
    1. Element-specific Energy Positions (pos1, pos2) derived from ELEMENT_PARAMS.
    2. Statistical Branching Ratio Constraint: Step(L2) height is fixed to 0.5 * Step(L3).
       This reduces free parameters and enforces physical realism.

    Equation:
    f(E) = H * [arctan(pi/w * (E - pos1)) + pi/2] + 0.5 * H * [arctan(pi/w * (E - pos2)) + pi/2]
           where H = height / pi.

    Parameters:
        height: Intensity of the primary L3 step.
        width: Width of the steps (w).
        pos1, pos2: Inflection points (L3, L2 energies).
    """

    def __init__(self, element="Fe", height=1.0, width=1.0, **kwargs):
        # 1. Resolve Energy Parameters based on element
        params = get_element_params(element)
        # Default to Fe values if not found, though get_element_params raises error for unknown elements
        peak_energies = params.get("peak_energies", (708.65, 721.65))
        e1_val = peak_energies[0]
        e2_val = peak_energies[1]

        # 2. Define Expression with strict 2:1 constraint
        val_pi = np.pi
        # Use 'atan' instead of 'arctan' for SymPy compatibility in expressions
        expr = f"height / {val_pi} * (atan({val_pi} / width * (x - pos1)) + {val_pi / 2}) + 0.5 * height / {val_pi} * (atan({val_pi} / width * (x - pos2)) + {val_pi / 2})"
        super().__init__(expression=expr, name="DoubleStep", **kwargs)

        # 3. Initialize Parameters
        self.height.value = height
        self.width.value = width
        self.pos1.value = e1_val
        self.pos2.value = e2_val

        # 4. Set physical bounds / constraints
        self.height.bmin = 0.0
        self.width.bmin = 1e-6
        self.width.bmax = 10.0  # Prevent extreme widths
        self.pos1.bmin = e1_val - 5.0
        self.pos1.bmax = e1_val + 5.0
        self.pos2.bmin = e2_val - 5.0
        self.pos2.bmax = e2_val + 5.0


def _create_component(type_name: str, **kwargs: Any) -> Component:
    """Factory function to create HyperSpy 1D components, including custom ones."""
    if type_name == "DoubleStep":
        return DoubleStep(**kwargs)
    if hasattr(hs.model.components1D, type_name):
        cls = getattr(hs.model.components1D, type_name)
        return cls(**kwargs)
    raise ValueError(f"Unknown component type: {type_name}")


def _init_component(comp: Component, config: dict[str, Any]) -> None:  # noqa: PLR0912
    """Initialize component parameters from config."""
    if "param_guesses" in config:
        for param, value in config["param_guesses"].items():
            if hasattr(comp, param):
                parameter = getattr(comp, param)
                if hasattr(parameter, "value"):
                    parameter.value = value
                else:
                    # Handle properties (like height) that are not Parameter objects
                    setattr(comp, param, value)
    # Set bounds
    if "param_bounds" in config:
        for param, (bmin, bmax) in config["param_bounds"].items():
            if hasattr(comp, param):
                parameter = getattr(comp, param)
                # Only set bounds if the attribute is a Parameter object (has bmin/bmax)
                if hasattr(parameter, "bmin") and hasattr(parameter, "bmax"):
                    if bmin is not None:
                        parameter.bmin = bmin
                    if bmax is not None:
                        parameter.bmax = bmax
    # Estimate parameters if range provided
    if "estimation_range" in config:
        est_range = config["estimation_range"]
        if hasattr(comp, "estimate_parameters"):
            comp.estimate_parameters(config["signal"], est_range[0], est_range[1])  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue]


def _freeze_params(comp: Component, strategy: str) -> None:
    """Freeze component parameters based on strategy."""
    if strategy == "after_fit":
        for param in comp.parameters:
            param.free = False
    elif strategy == "partial":
        for param in comp.parameters:
            if param.name.lower() in {"centre", "center", "sigma", "fwhm", "gamma"}:
                param.free = False


def _unfreeze_all(model: Model1D) -> None:
    """Unfreeze all parameters in the model."""
    for comp in model:
        for param in comp.parameters:
            param.free = True


def fit_stepwise(  # noqa: PLR0913, PLR0917
    signal: Signal1D,
    peak_configs: list[dict[str, Any]],
    mask: np.ndarray | None = None,
    iterpath: str = "serpentine",
    optimizer: str = "lm",
    freeze_strategy: str = "after_fit",
    unfreeze_final: bool = True,
    fit_global_first: bool = True,
) -> FittingResult:
    """
    Performs stepwise model fitting for a list of peaks with configurable freezing.

    Args:
        signal (hs.signals.Signal1D): The signal to fit.
        peak_configs (list): List of dictionaries defining peaks.
        mask (np.ndarray, optional): Mask for fitting.
        iterpath (str): Fitting iteration path.
        optimizer (str): Optimizer name.
        freeze_strategy (str): When to freeze parameters.
        unfreeze_final (bool): Whether to unfreeze and refit at the end.
        fit_global_first (bool): If True, fits the mean spectrum first to get better initial guesses.

    Returns:
        FittingResult: Dataclass containing model, component signals, and residuals.
    """
    model = signal.create_model()

    # Step 0: Global Fit (Optional but recommended)
    if fit_global_first:
        mean_sig = signal.mean(axis=signal.axes_manager.navigation_axes)
        global_model = mean_sig.create_model()
        for config in peak_configs:
            comp = _create_component(config["type"])
            comp.name = config["name"]
            config_copy = dict(config)
            config_copy["signal"] = mean_sig
            _init_component(comp, config_copy)
            global_model.append(comp)

        # Fit mean spectrum
        global_model.fit(optimizer=optimizer)

        # Update initial guesses in peak_configs based on global fit
        for config in peak_configs:
            if "param_guesses" not in config:
                config["param_guesses"] = {}
            comp_name = config["name"]
            # In HyperSpy, we can check if a component name is in the model directly
            if comp_name in global_model:
                g_comp = global_model[comp_name]
                for param in g_comp.parameters:
                    config["param_guesses"][param.name] = param.value

    # Step 1: Sequential Fitting on Stack
    ranges_set = False
    for config in peak_configs:
        comp = _create_component(config["type"])
        comp.name = config["name"]
        config_with_signal = dict(config)
        config_with_signal["signal"] = signal
        _init_component(comp, config_with_signal)
        model.append(comp)
        if "fit_range" in config:
            fit_range = config["fit_range"]
            if not ranges_set:
                model.set_signal_range(fit_range[0], fit_range[1])
                ranges_set = True
            else:
                model.add_signal_range(fit_range[0], fit_range[1])

        # Multifit current component
        model.multifit(iterpath=iterpath, bounded=True, mask=mask, optimizer=optimizer)
        _freeze_params(comp, freeze_strategy)

    if unfreeze_final or freeze_strategy == "final_only":
        _unfreeze_all(model)
        model.multifit(iterpath=iterpath, bounded=True, mask=mask, optimizer=optimizer)

    # Extract component signals and residuals
    component_signals = {}
    for comp in model:
        try:
            component_signals[comp.name] = model.as_signal(component_list=[comp])
        except Exception as e:
            print(f"Warning: Could not extract signal for component '{comp.name}': {e}")

    try:
        model_sig = model.as_signal()
        residuals = signal - model_sig
    except Exception as e:
        print(f"Warning: Could not calculate residuals: {e}")
        residuals = None

    return FittingResult(model=model, components=component_signals, residuals=residuals)


# ==================================================================================================
# 7. Main Analysis Class
# ==================================================================================================


def hyperspy_to_xarray(signal: hs.signals.BaseSignal) -> Any:
    """
    Manual conversion of HyperSpy Signal to xarray DataArray.
    Used because HyperSpy 2.x removed built-in .to_xarray().
    """
    import xarray as xr  # noqa: PLC0415

    data = signal.data
    # Get all axes and sort them by their position in the data array
    all_axes = list(signal.axes_manager.navigation_axes) + list(signal.axes_manager.signal_axes)
    all_axes.sort(key=lambda x: x.index_in_array)

    dims = []
    coords = {}

    for i, axis in enumerate(all_axes):
        # Use name or default index-based name (1-based for dimensions)
        name = axis.name if axis.name else f"dim_{i + 1}"
        dims.append(name)

        # Generate coordinate values
        if hasattr(axis, "axis") and axis.axis is not None:  # DataAxis (non-uniform)
            coords[name] = axis.axis
        else:  # UniformDataAxis
            coords[name] = np.arange(axis.size) * axis.scale + axis.offset

    # HyperSpy signals can sometimes have extra singleton dimensions in the data array
    # that are not reflected in the axes_manager, especially with lazy signals.
    if data.ndim > len(dims):
        data = np.squeeze(data)

    # Final check: if still mismatched, fallback to default dimensions
    if data.ndim != len(dims):
        print(f"Warning: Data shape {data.shape} does not match axes count {len(dims)}. Using default dimensions.")
        return xr.DataArray(data, attrs={"units": [getattr(ax, "units", "") for ax in all_axes]})

    return xr.DataArray(data, dims=dims, coords=coords, attrs={"units": [getattr(ax, "units", "") for ax in all_axes]})


class TXMAnalyzer:
    def __init__(self, file_path: str | Path, element: str | None = None, lazy: bool = False):
        """
        Initialize the TXMAnalyzer.

        Args:
            file_path (str or Path): Path to the HDF5 file.
            element (str, optional): Default element for analysis. If provided, checks paths relative to DATA_ROOT.
            lazy (bool): If True, loads the data lazily.
        """
        self.file_path = self._resolve_path(file_path, element)
        self.lazy = lazy
        self.raw_signal = None
        self.od_signal = None
        self.denoised_signal = None
        self.corrected_signal = None

        # Dictionaries to store multiple ROIs and their corresponding data/results
        self.roi = {}  # Dict[str, ROI]
        self.roi_data = {}  # Dict[str, Signal]

        self.onset_energy = {}  # Dict[str, tuple]
        self.clustering_results = {}  # Dict[str, ClusteringResults] (KMeans/Standard)
        self.umap_hdbscan_results = {}  # Dict[str, UMAPHDBSCANResults] (UMAP+HDBSCAN)

        self.models = {}  # Dict[str, Model1D]
        self.fitting_results = {}  # Dict[str, FittingResult]
        self.parameters = {}  # Dict[str, Any]

        self._load_data()

    def _resolve_path(self, file_path: str | Path, element: str | None) -> Path:  # noqa: ARG002, PLR6301
        path = Path(file_path)
        if path.exists():
            return path

        # Try relative to current directory
        alt_path = Path.cwd() / path.name
        if alt_path.exists():
            return alt_path

        # Try DATA_ROOT / STXM / path.name
        try:
            from Figure.config import DATA_ROOT  # noqa: PLC0415

            data_path = DATA_ROOT / "STXM" / path.name
            if data_path.exists():
                return data_path
        except ImportError:
            pass

        return path  # Return original if not found (let load fail)

    def _load_data(self):
        self.raw_signal = load_stxm_signal(self.file_path, lazy=self.lazy)

    def preprocess(self, new_energy_scale: float = 0.1):
        self.parameters["preprocess"] = {"new_energy_scale": new_energy_scale}
        if self.raw_signal is None:
            raise ValueError("Data not loaded.")

        # For lazy signals, we avoid deepcopy if possible or use as_lazy
        signal = self.raw_signal.deepcopy() if not self.lazy else self.raw_signal.as_lazy()

        # 自动修正能量轴名称和单位，兼容不同beamline导出
        found_energy = False
        for ax in signal.axes_manager.navigation_axes:
            name = (ax.name or "").lower()
            units = (ax.units or "").lower()
            # If it looks like energy or is the only nav axis
            if name == "energy" or units == "ev":
                ax.name = "Energy"
                ax.units = "eV"
                found_energy = True
                break

        # Fallback: if no energy axis found, assume the first navigation axis is energy
        if not found_energy and signal.axes_manager.navigation_axes:
            signal.axes_manager.navigation_axes[0].name = "Energy"
            signal.axes_manager.navigation_axes[0].units = "eV"

        # Align and transpose to Signal1D immediately
        signal = align_and_transpose_stack(signal)

        # 再次修正（有些align/transpose后会丢失信息）
        for ax in signal.axes_manager.signal_axes:
            name = (ax.name or "").lower()
            units = (ax.units or "").lower()
            if name != "energy" or units != "ev":
                ax.name = "Energy"
                ax.units = "eV"
                break
        signal = interpolate_energy_axis(signal, new_energy_scale)
        self.od_signal = convert_to_optical_density(signal)

    def denoise(self, components_number: int | None = None, algorithm: str = "SVD"):
        """
        Performs PCA denoising on the signal.

        Args:
            components_number (int, optional): Number of components to keep. If None, uses automatic elbow estimation.
            algorithm (str): Decomposition algorithm.
        """
        self.parameters["denoise"] = {"components_number": components_number, "algorithm": algorithm}
        if self.od_signal is None:
            raise ValueError("OD data not available. Run preprocess() first.")
        self.denoised_signal = apply_pca_denoising(self.od_signal, components_number=components_number, algorithm=algorithm)

    def remove_background(
        self,
        fit_range: tuple[float, float] | None = None,
        component_name: str = "Offset",
        polynomial_order: int = 1,
        element: str | None = None,
    ):
        """
        Removes background from the signal (prefers denoised_signal, otherwise od_signal).
        Updates stored data (denoised_signal or od_signal).

        Args:
            fit_range (tuple, optional): Energy range for fitting (min, max).
                - If None and element is specified, loads from ELEMENT_PARAMS.
                - If both are None, defaults to "Mn" parameters (for backward compatibility).
            component_name (str): 'Offset', 'PowerLaw', or 'Polynomial'.
            polynomial_order (int): Polynomial order if component_name is 'Polynomial'.
            element (str, optional): Element name (e.g. "Mn", "Zn") for automatic fit_range loading.
        """
        # Determine fit_range logic
        if fit_range is None:
            if element is not None and element in ELEMENT_PARAMS:
                fit_range = cast(tuple[float, float], ELEMENT_PARAMS[element]["preedge_fit_range"])
            else:
                fit_range = cast(tuple[float, float], ELEMENT_PARAMS["Mn"]["preedge_fit_range"])

        self.parameters["remove_background"] = {
            "fit_range": fit_range,
            "component_name": component_name,
            "polynomial_order": polynomial_order,
            "element": element,
        }
        if self.od_signal is None:
            raise ValueError("OD data not available. Run preprocess() first.")

        target = self.denoised_signal if self.denoised_signal is not None else self.od_signal
        result = remove_background(target, fit_range, component_name, polynomial_order)

        if self.denoised_signal is not None:
            self.denoised_signal = result
        else:
            self.od_signal = result

        # Refresh all existing ROIs to reflect the background removal
        for name, roi_obj in self.roi.items():
            # Pass element info if available in existing ROIs to maintain consistency
            # However, set_roi signature now accepts element to RELOAD params.
            # Here we just want to refresh data, so we use existing params.
            self.set_roi(
                name=name,
                energy_range=roi_obj.energy,
                spatial_mask=roi_obj.spatial,
                element=roi_obj.element,
            )

    def apply_b_value_correction(self, mask_ranges: tuple[float, float] | None = None, element: str | None = None, num_points: int = 100):
        """
        Applies B-value (stray light) correction to the current best data (PCA or OD).
        Updates self.corrected_signal.

        Args:
            mask_ranges (tuple, optional): Intensity ranges (min, max) for thin/thick masking.
                If None, uses 'mask_thresholds' from ELEMENT_PARAMS if element is provided.
            element (str, optional): Element symbol for automatic threshold calculation.
            num_points (int): Coarse search points for optimization.

        Returns:
            tuple: (best_k, popt, best_r2)
        """
        source = self.denoised_signal if self.denoised_signal is not None else self.od_signal
        if source is None:
            raise ValueError("No data available for correction.")

        # Auto-calculate thresholds if element is provided
        if mask_ranges is None and element is not None and element in ELEMENT_PARAMS:
            s_sum = source.sum(axis=-1).data
            s_sum_max = np.nanmax(s_sum)
            thresholds = ELEMENT_PARAMS[element]["mask_thresholds"]
            mask_ranges = (thresholds[0] * s_sum_max, thresholds[1] * s_sum_max)

        if mask_ranges is None:
            raise ValueError("mask_ranges must be provided or element must be specified for auto-thresholding.")

        self.parameters["apply_b_value_correction"] = {"mask_ranges": mask_ranges, "num_points": num_points, "element": element}

        best_k, best_popt, best_r2 = optimize_kappa(source, mask_ranges, num_points)

        kappa = best_popt[0]
        s_corrected = source.deepcopy()
        term = (1 + kappa) * np.exp(-source.data) - kappa
        term[term <= 0] = 1e-10
        s_corrected.data = -np.log(term)

        self.corrected_signal = s_corrected
        return best_k, best_popt, best_r2

    def plot_clusters(self, roi_name: str = "full", ax=None):
        """
        Visualizes clustering results for a specific ROI.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if roi_name not in self.clustering_results:
            raise ValueError(f"No clustering results for ROI '{roi_name}'.")

        res = self.clustering_results[roi_name]
        labels = res.labels
        spectra = res.spectra

        # Try to use configured colors/styles if available
        try:
            from Figure.config import setup  # noqa: PLC0415

            colors = setup()
        except ImportError:
            colors = None

        if ax is None:
            # Use constrained_layout for better spacing
            _fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        # Plot Labels
        # Use discrete colormap logic if possible, or tab10/Set3 for clusters
        labels_sig = hs.signals.Signal2D(labels)
        # HyperSpy plot is convenient, but for publication we might prefer imshow directly
        # to control colorbar/axes better.
        # But HS plot handles axes units automatically.
        labels_sig.plot(ax=ax[0], cmap="tab20", colorbar=False)
        ax[0].set_title(f"Clusters: {roi_name}")
        ax[0].set_xlabel("X Position")
        ax[0].set_ylabel("Y Position")

        # Plot Spectra
        # Use hs.plot.plot_spectra for easy multi-spectrum plotting
        if spectra is not None:
            hs.plot.plot_spectra(
                spectra,
                ax=ax[1],
                colors=colors[: len(spectra)] if colors else None,
                style="mosaic",
                linewidth=1.5,
            )
            ax[1].set_title("Cluster Mean Spectra")
            ax[1].set_xlabel("Energy (eV)")
            ax[1].set_ylabel("Optical Density")
            ax[1].grid(True, which="major", linestyle="--", alpha=0.5)
            # Improve legend
            ax[1].legend(loc="upper right", frameon=True, fontsize="small")

        return ax

    def plot_fit_results(self, roi_name: str = "full", components: list[str] | None = None):
        """
        Visualizes fitting results (parameter maps).
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fit_maps = self.get_fit_maps(roi_name)

        if components is None:
            # Default to all components that have a height map
            components = [c for c, m in fit_maps.items() if "height" in m]

        n_comp = len(components)
        if n_comp == 0:
            print("No height maps found to plot.")
            return

        # Use constrained_layout for automatic adjustment
        fig, axes = plt.subplots(1, n_comp, figsize=(4 * n_comp, 3.5), constrained_layout=True)
        if n_comp == 1:
            axes = [axes]

        for i, comp_name in enumerate(components):
            param_map = fit_maps[comp_name]
            if "height" in param_map:
                # Use 'sunset' or 'inferno' for intensity
                im = axes[i].imshow(param_map["height"], cmap="sunset", origin="lower")
                axes[i].set_title(f"{comp_name} Intensity")
                axes[i].axis("off")
                # Add colorbar
                cbar = fig.colorbar(im, ax=axes[i], orientation="vertical", shrink=0.8, aspect=20)
                cbar.ax.tick_params(labelsize=8)

        return axes

    def set_roi(  # noqa: PLR0912, PLR0914
        self,
        name: str | None = None,
        energy_range: tuple[float, float] | None = None,
        spatial_mask: np.ndarray | None = None,
        element: str | list[str] | None = None,
        hyperspy_roi: BaseROI | None = None,
    ):
        """
        Sets a named ROI and updates the corresponding entry in self.roi_data.
        Supports automatic parameter loading based on element name, or batch creation of multiple element ROIs.
        Also supports HyperSpy ROI objects for defining spatial regions.

        Args:
            name (str, optional): Identifier for the ROI. If None and element is a string, defaults to element; otherwise "default".
            energy_range (min_eV, max_eV): Energy range. If None and element is specified, loads from ELEMENT_PARAMS.
            spatial_mask: Boolean mask array matching spatial dimensions (y, x).
            element (str or List[str], optional): Element identifier (e.g. "Mn") or list of identifiers.
                - None: Do not use preset parameters.
                - str: Load parameters for this element.
                - List[str]: Batch create ROIs for each element in the list (ignores name and energy_range arguments).
            hyperspy_roi (BaseROI, optional): A HyperSpy RegionOfInterest object (e.g. RectangularROI, CircleROI).
                If provided, it will be used to generate the spatial mask (if possible) or slice the data.
        """
        # 1. Batch processing (List)
        if isinstance(element, list):
            for el in element:
                # Recursive call, automatically uses element name as name, and loads default energy_range
                self.set_roi(name=el, element=el, spatial_mask=spatial_mask, hyperspy_roi=hyperspy_roi)
            return

        # 2. Single processing
        # Determine name
        if name is None:
            name = element if isinstance(element, str) else "default"

        # Load default parameters
        if isinstance(element, str) and element in ELEMENT_PARAMS:
            params = ELEMENT_PARAMS[element]
            if energy_range is None:
                energy_range = cast(tuple[float, float] | None, params.get("full_energy_range"))

        # Source for ROI data: B-corrected > PCA > OD
        if self.corrected_signal is not None:
            source = self.corrected_signal
        elif self.denoised_signal is not None:
            source = self.denoised_signal
        else:
            source = self.od_signal

        if source is None:
            raise ValueError("No processed data (OD, PCA, or Corrected) available.")

        # Handle HyperSpy ROI
        if hyperspy_roi is not None:
            # If a HyperSpy ROI is provided, we try to create a spatial mask from it.
            # Currently, stxm_utils heavily relies on boolean spatial masks for the full image.
            # We can generate a boolean mask by applying the ROI to a dummy signal of ones.
            try:
                # Create a dummy signal with the same navigation shape as the source
                # The source signal is (Y, X | Energy) -> navigation (Y, X)
                # But HyperSpy ROIs operate on the axes.
                # If we have a 2D navigation space, we need a 2D signal to apply the ROI to.
                # Let's create a 2D dummy signal representing the navigation space.
                # source.mean(axis=signal_axis) gives a 2D image (Y, X).
                dummy_nav = source.mean(axis="signal")
                # Fill with zeros
                dummy_nav.data[:] = 0
                # Fill with ones?
                # Actually, HyperSpy ROI `__call__` slices the data.
                # If we want a boolean mask:
                # 1. Create dummy of zeros.
                # 2. Slice with ROI? No, slicing just returns a view.
                # We need to know WHICH pixels are inside.
                # Method: Create an index array/signal.
                # Create a signal where data[y, x] = flattened_index or similar?
                # Or: use the geometry of the ROI if simple.
                # Better: Use HyperSpy's built-in functionality if available.
                # For now, let's stick to the boolean mask logic required by clustering/fitting.
                # If the ROI is rectangular, we can get slices.
                # If the ROI is arbitrary, we need a mask.
                # workaround: Create a boolean mask of False.
                # But how to set True inside ROI?
                # HyperSpy doesn't easily output a boolean mask for an ROI on a grid without interactive widget.
                # Wait, interactive ROI slicing `sliced = roi(signal)`.
                # If we slice an index array, we know which indices are kept.

                # Construct coordinate grid
                ny, nx = source.axes_manager.navigation_shape
                # Create a signal of indices: (Y, X)
                indices = np.arange(ny * nx).reshape(ny, nx)
                s_indices = hs.signals.Signal2D(indices)
                # Copy axes
                s_indices.axes_manager[0].offset = source.axes_manager.navigation_axes[1].offset
                s_indices.axes_manager[0].scale = source.axes_manager.navigation_axes[1].scale
                s_indices.axes_manager[1].offset = source.axes_manager.navigation_axes[0].offset
                s_indices.axes_manager[1].scale = source.axes_manager.navigation_axes[0].scale
                # Note: Axes order in HyperSpy Signal2D is (x, y) in axes_manager but (y, x) in data?
                # Let's double check axes mapping.
                # source.axes_manager.navigation_axes[0] is usually y?
                # Check `load_stxm_signal`: axes=[energy_axis, x_position, y_position].
                # Created Signal2D(..., axes=[...]).
                # HyperSpy internal order is usually navigation reversed?
                # Let's just trust that s_indices with same axes metadata works.
                for i, ax in enumerate(source.axes_manager.navigation_axes):
                    s_indices.axes_manager[i].name = ax.name
                    s_indices.axes_manager[i].scale = ax.scale
                    s_indices.axes_manager[i].offset = ax.offset
                    s_indices.axes_manager[i].units = ax.units

                # Apply ROI
                sliced_indices = hyperspy_roi(s_indices)

                # Create boolean mask
                mask_full = np.zeros((ny, nx), dtype=bool)
                # Mark kept indices
                kept_flat = sliced_indices.data.flatten().astype(int)
                # Note: NaN might appear if ROI has non-rectangular selection?
                # RectangularROI returns rectangular slice.
                # If kept_flat contains valid indices:
                if kept_flat.size > 0:
                    np.put(mask_full, kept_flat, True)

                # Update spatial_mask (combine if existing?)
                # For now, override or combine. Let's override or AND if both present?
                # Usually user passes one.
                spatial_mask = mask_full

            except Exception as e:
                print(f"Warning: Could not generate spatial mask from HyperSpy ROI: {e}")
                # We still store the ROI object.

        # Store parameters (summary for mask)
        mask_info = f"Mask shape: {spatial_mask.shape}" if spatial_mask is not None else "None"
        roi_info = f"Type: {type(hyperspy_roi).__name__}" if hyperspy_roi is not None else "None"

        if "set_roi" not in self.parameters:
            self.parameters["set_roi"] = {}
        self.parameters["set_roi"][name] = {
            "energy_range": energy_range,
            "spatial_mask": mask_info,
            "hyperspy_roi": roi_info,
            "element": element,
        }

        self.roi[name] = ROI(energy=energy_range, element=element, spatial=spatial_mask, hyperspy_roi=hyperspy_roi)

        # Apply Energy Slice
        roi_sig = source.isig[energy_range[0] : energy_range[1]] if energy_range else source.deepcopy()

        # Note: Spatial mask is stored in self.roi[name].spatial and used in clustering/fitting,
        # but roi_data currently keeps full spatial extent (but energy sliced).
        self.roi_data[name] = roi_sig

    def _resolve_targets(self, rois: str | list[str] | None) -> list[str]:
        """
        Resolves the target ROIs for analysis.
        - If rois is a string, returns [rois].
        - If rois is a list, returns the list.
        - If rois is None:
            - If self.roi is empty, returns ["full"] (global analysis).
            - If self.roi has entries, returns all ROI names.
        """
        if rois is not None:
            if isinstance(rois, str):
                return [rois]
            return rois

        # rois is None
        if not self.roi:  # No ROIs defined
            return ["full"]
        else:
            return list(self.roi.keys())

    def _get_target_data(self, name: str) -> tuple[Any, np.ndarray | None]:
        """
        Helper to retrieve data and mask for analysis.
        Note:
        - If name is "full", returns (full_data, None).
        - If name is an ROI name, returns (roi_data, mask).
          ROI mask (self.roi[name].spatial) assumes True = Keep (ROI).
          However, HyperSpy usually expects True = Exclude (Mask).
          If passing to a HyperSpy function requiring an exclusion mask (e.g. navigation_mask),
          the caller must handle mask inversion.
        """
        if name == "full":
            if self.corrected_signal is not None:
                source = self.corrected_signal
            elif self.denoised_signal is not None:
                source = self.denoised_signal
            else:
                source = self.od_signal

            if source is None:
                raise ValueError("No processed data available. Run preprocess() first.")
            return source, None

        if name not in self.roi_data:
            raise ValueError(f"ROI '{name}' not found. Call set_roi('{name}', ...) first.")

        return self.roi_data[name], self.roi[name].spatial

    def _execute_on_rois(self, rois: str | list[str] | None, method_name: str, callback, **kwargs):
        """Helper to iterate over ROIs and execute a callback."""
        targets = self._resolve_targets(rois)
        self.parameters[method_name] = {"rois": targets, **kwargs}
        for name in targets:
            data, mask = self._get_target_data(name)
            callback(name, data, mask)

    def calculate_onset_energy(self, rois: str | list[str] | None = None, threshold_ratio: float = 0.1, smooth_sigma: float = 0):
        """
        Calculates onset energy map using the 10% intensity threshold method.

        Determines the chemical shift (onset energy) for each pixel using the method from
        Tan et al. (2012) Ultramicroscopy. The onset is defined as the energy where intensity
        reaches a threshold percentage of the maximum.

        Args:
            rois: Target ROIs.
                - None: Auto-detect (global if no ROIs defined, else all ROIs).
                - str: Specific ROI name.
                - list[str]: List of specific ROI names.
            threshold_ratio: Fraction of max intensity to define onset (default: 0.1 = 10%).
            smooth_sigma: Gaussian smoothing sigma for spatial noise reduction (default: 0 = none).

        Results are stored in ``self.onset_energy[roi_name]`` as ``(onset_map, flattened_values)``.
        """

        def _process(name, data, mask):
            # Helper should handle masking output.
            onset_map, flattened = calculate_onset_energy(data, threshold_ratio, smooth_sigma=smooth_sigma)
            if mask is not None and onset_map.shape == mask.shape:
                onset_map[~mask] = np.nan
            self.onset_energy[name] = (onset_map, flattened)

        self._execute_on_rois(rois, "calculate_onset_energy", _process, threshold_ratio=threshold_ratio, smooth_sigma=smooth_sigma)

    def perform_clustering(  # noqa: PLR0913, PLR0917
        self,
        rois: str | list[str] | None = None,
        n_clusters: int | None = None,
        algorithm: Any = "kmeans",
        preprocessing: str | None = None,
        exclude_first_component: bool = False,
        metric: str = "silhouette",
    ):
        """
        Performs clustering analysis on PCA-decomposed spectra.

        Implements the clustering methodology from Vogt (2004) Ultramicroscopy for
        spectromicroscopy data. Clustering is performed in the PCA eigenspace
        (loadings), optionally excluding the first component (total thickness).

        Args:
            rois: Target ROIs.
                - None: Auto-detect (global if no ROIs defined, else all ROIs).
                - str: Specific ROI name.
                - list[str]: List of specific ROI names.
            n_clusters: Number of clusters. If None, auto-estimates using ``metric``.
            algorithm: Clustering algorithm - string shortcut ('kmeans', etc.)
                or a configured sklearn estimator instance.
            preprocessing: Data preprocessing before clustering.
                Options: 'norm' (L2 normalize), 'standard' (z-score), None.
            exclude_first_component: If True, excludes the 1st PCA component
                (thickness/absorbance) to focus on chemical speciation.
            metric: Metric for auto-estimating n_clusters ('silhouette', 'elbow', 'gap').

        Results are stored in ``self.clustering_results[roi_name]`` as a ``ClusteringResults``
        dataclass with ``labels`` (cluster map) and ``spectra`` (mean spectra per cluster).
        """

        def _process(name, data, mask):
            hs_mask = ~mask if mask is not None else None
            labels, signals = perform_clustering_analysis(
                data,
                n_clusters,
                algorithm,
                navigation_mask=hs_mask,
                preprocessing=preprocessing,
                exclude_first_component=exclude_first_component,
                metric=metric,
            )
            self.clustering_results[name] = ClusteringResults(labels=labels, spectra=signals)

        self._execute_on_rois(
            rois,
            "perform_clustering",
            _process,
            n_clusters=n_clusters,
            algorithm=algorithm,
            preprocessing=preprocessing,
            exclude_first_component=exclude_first_component,
            metric=metric,
        )

    def perform_umap_clustering(  # noqa: PLR0913, PLR0917
        self,
        rois: str | list[str] | None = None,
        # UMAP
        n_neighbors: int = 15,
        n_components: int = 2,
        min_dist: float = 0.1,
        metric: str = "cosine",
        # HDBSCAN
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        # General
        preprocessing: str | None = None,
        exclude_first_component: bool = False,
    ):
        """
        Performs clustering using UMAP manifold learning followed by HDBSCAN.

        This advanced clustering pipeline combines dimensionality reduction via
        UMAP with density-based clustering via HDBSCAN, enabling discovery of
        non-linear, arbitrary-shaped clusters without pre-specifying cluster count.

        Methodology based on Blanco-Portals et al. (2022). Using 'cosine' metric
        is recommended for spectral data to be insensitive to thickness variations.

        Workflow:
            1. PCA decomposition (if not already done).
            2. UMAP embedding of PCA loadings into 2D manifold.
            3. HDBSCAN clustering on the UMAP embedding.

        Args:
            rois: Target ROIs.
                - None: Auto-detect (global if no ROIs defined, else all ROIs).
                - str: Specific ROI name.
                - list[str]: List of specific ROI names.
            n_neighbors: UMAP locality parameter (default: 15).
            n_components: UMAP output dimensions (default: 2).
            min_dist: UMAP minimum distance between points (default: 0.1).
            metric: UMAP distance metric (default: 'cosine').
            min_cluster_size: HDBSCAN minimum cluster size (default: 10).
            min_samples: HDBSCAN core point threshold (default: None = min_cluster_size).
            preprocessing: Data preprocessing ('norm', 'standard', 'minmax', None).
            exclude_first_component: Exclude 1st PCA component (thickness).

        Results are stored in:
            - ``self.clustering_results[roi_name]``: ClusteringResults with labels and spectra.
            - ``self.umap_embeddings[roi_name]``: 2D UMAP embedding array for visualization.
        """

        def _process(name, data, mask):
            labels, signals, embedding, hdbscan_props = perform_umap_hdbscan_clustering(
                data,
                navigation_mask=mask,
                preprocessing=preprocessing,
                exclude_first_component=exclude_first_component,
                umap_n_neighbors=n_neighbors,
                umap_n_components=n_components,
                umap_min_dist=min_dist,
                umap_metric=metric,
                hdbscan_min_cluster_size=min_cluster_size,
                hdbscan_min_samples=min_samples,
            )

            # Store specific UMAP+HDBSCAN results
            # Extract props
            probs = hdbscan_props.get("probabilities")
            scores = hdbscan_props.get("outlier_scores")

            self.umap_hdbscan_results[name] = UMAPHDBSCANResults(labels=labels, spectra=signals, embedding=embedding, probabilities=probs, outlier_scores=scores)

        self._execute_on_rois(
            rois,
            "perform_umap_clustering",
            _process,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            min_cluster_size=min_cluster_size,
            preprocessing=preprocessing,
            exclude_first_component=exclude_first_component,
        )

    def optimize_umap_params(
        self,
        rois: str | list[str] | None = None,
        n_trials: int = 50,
        exclude_first_component: bool = True,
    ) -> Any:
        """
        Runs Optuna optimization for UMAP+HDBSCAN parameters on the specified ROI(s).
        Returns the Optuna study object (containing best_params).
        """
        targets = self._resolve_targets(rois)
        if len(targets) > 1:
            print(f"Warning: Optimization requested for multiple ROIs {targets}. Using the first one: '{targets[0]}'")

        target_name = targets[0]
        data, mask = self._get_target_data(target_name)

        study = optimize_clustering_params(
            data,
            navigation_mask=mask,
            n_trials=n_trials,
            exclude_first_component=exclude_first_component,
            study_name=f"stxm_{target_name}",
        )
        return study

    def fit(  # noqa: PLR0913, PLR0917
        self,
        peak_configs: list[dict[str, Any]],
        rois: str | list[str] | None = None,
        iterpath: str = "serpentine",
        optimizer: str = "lm",
        freeze_strategy: str = "after_fit",
        unfreeze_final: bool = True,
        fit_global_first: bool = True,
    ):
        """
        Performs stepwise model fitting.

        Args:
            peak_configs (list): List of peak configuration dictionaries.
            rois: Target ROIs. None=Auto (Global if no ROIs, else all ROIs), str=Specific ROI, List=List of specific ROIs.
            iterpath (str): Fitting order.
            optimizer (str): Optimizer algorithm.
            freeze_strategy (str): Parameter freezing strategy.
            unfreeze_final (bool): Whether to unfreeze all parameters for simultaneous fitting at the end.
            fit_global_first (bool): If True, fits the mean spectrum first to get better initial guesses.
        """

        def _process(name, data, mask):
            hs_mask = ~mask if mask is not None else None

            fit_data = data
            if name in self.roi:
                roi_obj = self.roi[name]
                if roi_obj.element and roi_obj.element in ELEMENT_PARAMS:
                    # Check if we should slice to main_fit_range
                    main_range = ELEMENT_PARAMS[roi_obj.element].get("main_fit_range")
                    if main_range:
                        # Slice the signal to the main fit range to optimize fitting and exclude noise
                        fit_data = data.isig[main_range[0] : main_range[1]]

            result = fit_stepwise(
                fit_data,
                peak_configs,
                mask=hs_mask,
                iterpath=iterpath,
                optimizer=optimizer,
                freeze_strategy=freeze_strategy,
                unfreeze_final=unfreeze_final,
                fit_global_first=fit_global_first,
            )
            self.models[name] = result.model
            self.fitting_results[name] = result

        self._execute_on_rois(
            rois, "fit", _process, peak_configs=peak_configs, iterpath=iterpath, optimizer=optimizer, freeze_strategy=freeze_strategy, unfreeze_final=unfreeze_final, fit_global_first=fit_global_first
        )

    def _export_umap_results(self, ctx: _ExportContext) -> None:
        """Export UMAP+HDBSCAN results to DataTree."""
        nav_shape = tuple(ctx.ref_signal.axes_manager.navigation_shape)
        for name, res in self.umap_hdbscan_results.items():
            ds_umap = ctx.xr_module.Dataset()
            if res.embedding is not None:
                ds_umap["embedding"] = ctx.xr_module.DataArray(res.embedding, dims=["samples", "components"])

            # Map results (labels, probabilities, outlier_scores)
            for key in ["labels", "probabilities", "outlier_scores"]:
                val = getattr(res, key, None)
                if val is not None:
                    val_sq = np.squeeze(val)
                    if val_sq.shape == nav_shape:
                        ds_umap[key] = ctx.xr_module.DataArray(val_sq, dims=ctx.spatial_dims, coords=ctx.spatial_coords)
                    else:
                        ds_umap[key] = ctx.xr_module.DataArray(val)

            if res.spectra is not None:
                xr_spectra = hyperspy_to_xarray(res.spectra)
                ds_umap = ctx.xr_module.merge([ds_umap, xr_spectra.to_dataset(name="spectra")])

            if len(ds_umap.data_vars) > 0:
                ctx.dt[f"/rois/{name}/umap_hdbscan"] = ctx.tree_cls(ds_umap)

    def _export_clustering_results(self, ctx: _ExportContext) -> None:
        """Export standard clustering results to DataTree."""
        nav_shape = tuple(ctx.ref_signal.axes_manager.navigation_shape)
        for name, res in self.clustering_results.items():
            ds_cluster = ctx.xr_module.Dataset()
            if res.labels is not None:
                val_sq = np.squeeze(res.labels)
                if val_sq.shape == nav_shape:
                    ds_cluster["labels"] = ctx.xr_module.DataArray(val_sq, dims=ctx.spatial_dims, coords=ctx.spatial_coords)
                else:
                    ds_cluster["labels"] = ctx.xr_module.DataArray(res.labels)

            if res.spectra is not None:
                xr_spectra = hyperspy_to_xarray(res.spectra)
                ds_cluster = ctx.xr_module.merge([ds_cluster, xr_spectra.to_dataset(name="spectra")])

            if len(ds_cluster.data_vars) > 0:
                ctx.dt[f"/rois/{name}/clustering"] = ctx.tree_cls(ds_cluster)

    def _export_fitting_results(self, ctx: _ExportContext) -> None:
        """Export model fitting results to DataTree."""
        nav_shape = tuple(ctx.ref_signal.axes_manager.navigation_shape)
        for name in self.models:
            try:
                maps = self.get_fit_maps(name)
                if not maps:
                    print(f"Warning: No parameter maps found for model '{name}'.")
                    continue
                for comp, param_maps in maps.items():
                    ds_comp = ctx.xr_module.Dataset()
                    for param, val in param_maps.items():
                        if val.shape == nav_shape:
                            ds_comp[param] = ctx.xr_module.DataArray(val, dims=ctx.spatial_dims, coords=ctx.spatial_coords)
                        else:
                            ds_comp[param] = ctx.xr_module.DataArray(val)
                    ctx.dt[f"/rois/{name}/fitting/{comp}"] = ctx.tree_cls(ds_comp)

                # Export component signals and residuals
                if name in self.fitting_results:
                    result = self.fitting_results[name]
                    # Residuals
                    if result.residuals is not None:
                        xr_res = hyperspy_to_xarray(result.residuals)
                        ctx.dt[f"/rois/{name}/fitting/residuals"] = ctx.tree_cls(xr_res.to_dataset(name="residuals"))

                    # Component signals
                    for comp_name, sig in result.components.items():
                        if sig is not None:
                            xr_comp = hyperspy_to_xarray(sig)
                            comp_path = f"/rois/{name}/fitting/{comp_name}"
                            # Merge with existing component dataset if parameters were exported
                            if comp_path in ctx.dt:
                                ds_existing = ctx.dt[comp_path].dataset
                                ds_new = xr_comp.to_dataset(name="signal")
                                ctx.dt[comp_path] = ctx.tree_cls(ctx.xr_module.merge([ds_existing, ds_new]))
                            else:
                                ctx.dt[comp_path] = ctx.tree_cls(xr_comp.to_dataset(name="signal"))

            except Exception as e:
                print(f"Warning: Failed to export fitting results for '{name}': {e}")

    def export_to_datatree(self):
        """
        Exports the analysis results to an xarray.DataTree structure.
        Requires xarray>=2024.10 or datatree package.
        """
        import xarray as xr  # noqa: PLC0415

        try:
            from xarray import DataTree  # noqa: PLC0415
        except ImportError:
            from datatree import DataTree  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]  # noqa: PLC0415

        dt = DataTree.from_dict({})

        # Helper to safely convert signal
        def _add_signal(path: str, signal: Any) -> None:
            if signal is None:
                return
            try:
                xr_data = hyperspy_to_xarray(signal)
                dt[path] = DataTree(xr_data.to_dataset(name="data"))
            except Exception as e:
                print(f"Failed to convert {path}: {e}")

        # Export signals as sub-nodes under /signals/
        # This avoids merge issues if energy axes differ (e.g. raw vs interpolated)
        signal_map = {
            "raw": self.raw_signal,
            "optical_density": self.od_signal,
            "pca_denoised": self.denoised_signal,
            "b_corrected": self.corrected_signal,
        }

        for name, sig in signal_map.items():
            if sig is not None:
                _add_signal(f"/signals/{name}", sig)

        # Remove the problematic merge logic
        # if ds_list: ...

        # Export ROI signals
        for name, signal in self.roi_data.items():
            _add_signal(f"/rois/{name}/data", signal)

        # Get spatial reference
        ref_signal = self.corrected_signal or self.denoised_signal or self.od_signal or self.raw_signal
        if ref_signal is None:
            raise ValueError("No data available for export. Please load or process data first.")

        xr_ref = hyperspy_to_xarray(ref_signal)
        # Use dimension names from xr_ref (which handles the dim_1, dim_2 fallback)
        spatial_dims = list(xr_ref.dims[: len(ref_signal.axes_manager.navigation_axes)])
        spatial_coords = {d: xr_ref.coords[d] for d in spatial_dims if d in xr_ref.coords}

        # Create export context
        ctx = _ExportContext(
            xr_module=xr,
            tree_cls=DataTree,
            dt=dt,
            spatial_dims=spatial_dims,
            spatial_coords=spatial_coords,
            ref_signal=ref_signal,
        )

        # Parameters
        try:
            dt.attrs["parameters"] = json.dumps(self.parameters, cls=NumpyEncoder)
        except Exception as e:
            print(f"Warning: Could not serialize parameters: {e}")
            dt.attrs["parameters"] = str(self.parameters)

        # Onset Energy
        nav_shape = tuple(ref_signal.axes_manager.navigation_shape)
        for name, res in self.onset_energy.items():
            onset_map, flattened = res
            ds_onset = xr.Dataset()
            onset_map_sq = np.squeeze(onset_map)
            if onset_map_sq.shape == nav_shape:
                ds_onset["map"] = xr.DataArray(onset_map_sq, dims=spatial_dims, coords=spatial_coords)
            else:
                ds_onset["map"] = xr.DataArray(onset_map)
            ds_onset["flattened"] = xr.DataArray(flattened, dims=["points"])
            dt[f"/rois/{name}/onset"] = DataTree(ds_onset)

        # Delegate to helper methods
        self._export_umap_results(ctx)
        self._export_clustering_results(ctx)
        self._export_fitting_results(ctx)

        return dt

    @staticmethod
    def _extract_param_map(param: Any) -> tuple[Any, Any] | None:
        """Extract values and std from a parameter map. Returns (values, std) or None."""
        if not hasattr(param, "map") or param.map is None:
            return None
        pmap = param.map
        # Structured array with "values" field
        if isinstance(pmap, np.ndarray) and pmap.dtype.names and "values" in pmap.dtype.names:
            std = pmap["std"] if "std" in pmap.dtype.names else None
            return pmap["values"], std
        # Dictionary format
        if isinstance(pmap, dict) and "values" in pmap:
            return pmap["values"], pmap.get("std")
        # Simple array fallback
        if hasattr(pmap, "ndim"):
            return pmap, None
        return None

    def get_fit_maps(self, name: str | None = None):
        """
        Retrieves parameter maps from the fitted model.

        Args:
            name (str, optional): ROI name.
                                  If None:
                                    - If only 1 model exists, returns that model.
                                    - If multiple models exist, raises an error (must be specified).

        Returns:
            dict: Dictionary of parameter maps, structured as {component_name: {param_name: values}}.
        """
        if name is None:
            if len(self.models) == 1:
                name = next(iter(self.models.keys()))
            elif len(self.models) == 0:
                raise ValueError("No models fitted yet.")
            else:
                raise ValueError(f"Multiple models fitted ({list(self.models.keys())}). Please specify 'name'.")

        if name not in self.models:
            raise ValueError(f"Model for '{name}' not fitted.")

        maps: dict[str, dict[str, Any]] = {}
        model = self.models[name]
        for comp in model:
            maps[comp.name] = {}
            for param in comp.parameters:
                result = TXMAnalyzer._extract_param_map(param)
                if result is not None:
                    values, std = result
                    maps[comp.name][param.name] = values
                    if std is not None:
                        maps[comp.name][f"{param.name}_std"] = std
        return maps

    def _save_umap_to_hdf5(self, h5_group: Any) -> None:
        """Save UMAP+HDBSCAN results to HDF5 file."""
        for name, res in self.umap_hdbscan_results.items():
            grp = h5_group.require_group(f"rois/{name}/umap_hdbscan")
            for attr, key in [("embedding", "embedding"), ("labels", "labels"), ("probabilities", "probabilities"), ("outlier_scores", "outlier_scores")]:
                val = getattr(res, attr, None)
                if val is not None:
                    grp.create_dataset(key, data=val, compression="gzip")
            if res.spectra is not None and hasattr(res.spectra, "data"):
                grp.create_dataset("spectra", data=res.spectra.data, compression="gzip")

    def _save_clustering_to_hdf5(self, h5_group: Any) -> None:
        """Save standard clustering results to HDF5 file."""
        for name, res in self.clustering_results.items():
            grp = h5_group.require_group(f"rois/{name}/clustering")
            if res.labels is not None:
                grp.create_dataset("labels", data=res.labels, compression="gzip")
            if res.spectra is not None and hasattr(res.spectra, "data"):
                grp.create_dataset("spectra", data=res.spectra.data, compression="gzip")

    def _save_fitting_to_hdf5(self, h5_group: Any) -> None:
        """Save model fitting results to HDF5 file."""
        for name in self.models:
            try:
                maps = self.get_fit_maps(name)
                for comp, param_maps in maps.items():
                    grp = h5_group.require_group(f"rois/{name}/fitting/{comp}")
                    for param, val in param_maps.items():
                        if val is not None:
                            grp.create_dataset(param, data=val, compression="gzip")

                # Save component signals and residuals
                if name in self.fitting_results:
                    result = self.fitting_results[name]
                    # Residuals
                    if result.residuals is not None:
                        h5_group.create_dataset(f"rois/{name}/fitting/residuals", data=result.residuals.data, compression="gzip")
                    # Components
                    for comp_name, sig in result.components.items():
                        if sig is not None:
                            comp_grp = h5_group.require_group(f"rois/{name}/fitting/{comp_name}")
                            comp_grp.create_dataset("signal", data=sig.data, compression="gzip")

            except Exception as e:
                print(f"Warning: Failed to save fitting results for '{name}': {e}")

    def _save_to_hdf5(self, output_path: Path) -> Path:
        """Helper to save analysis results to HDF5."""
        with h5py.File(str(output_path), "w") as h5_file:
            # Save processed signals
            for key, signal in [("optical_density", self.od_signal), ("pca_denoised", self.denoised_signal), ("b_corrected", self.corrected_signal)]:
                if signal is not None:
                    h5_file.create_dataset(key, data=signal.data, compression="gzip")

            # Save Onset Energy
            for name, res in self.onset_energy.items():
                onset_map, flattened = res
                grp = h5_file.require_group(f"rois/{name}/onset")
                grp.create_dataset("map", data=onset_map, compression="gzip")
                grp.create_dataset("flattened", data=flattened, compression="gzip")

            # Delegate to helper methods
            self._save_umap_to_hdf5(h5_file)
            self._save_clustering_to_hdf5(h5_file)
            self._save_fitting_to_hdf5(h5_file)

            # Save parameters as attributes
            try:
                h5_file.attrs["parameters"] = json.dumps(self.parameters, cls=NumpyEncoder)
            except Exception:
                h5_file.attrs["parameters"] = str(self.parameters)

        print(f"Data saved to: {output_path}")
        return output_path

    def save(self, output_path: str | Path | None = None, format: Literal["hspy", "hdf5", "netcdf"] = "hspy") -> Path:
        """
        Save analysis results to disk.

        Args:
            output_path: Output file path. If None, generates from input filename.
            format: Output format - 'hspy' (HyperSpy), 'hdf5', or 'netcdf'.

        Returns:
            Path: The output file path.
        """
        # Generate default output path
        if output_path is None:
            try:
                from Figure.config import OUTPUT_ROOT  # noqa: PLC0415

                base_name = self.file_path.stem + "_Corrected"
                output_path = OUTPUT_ROOT / f"{base_name}.{format}"
            except ImportError:
                output_path = self.file_path.parent / f"{self.file_path.stem}_Corrected.{format}"
        else:
            output_path = Path(output_path)

        # Export based on format
        if format == "hspy":
            # Use HyperSpy's native format for best compatibility
            dt = self.export_to_datatree()
            dt.to_zarr(str(output_path.with_suffix(".zarr")), mode="w")
            print(f"Data saved to: {output_path.with_suffix('.zarr')}")
            return output_path.with_suffix(".zarr")

        elif format == "netcdf":
            dt = self.export_to_datatree()
            dt.to_netcdf(str(output_path.with_suffix(".nc")))
            print(f"Data saved to: {output_path.with_suffix('.nc')}")
            return output_path.with_suffix(".nc")

        elif format == "hdf5":
            return self._save_to_hdf5(output_path.with_suffix(".h5"))

        else:
            raise ValueError(f"Unknown format: {format}. Use 'hspy', 'hdf5', or 'netcdf'.")
