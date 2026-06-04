"""XAS 多变量分析和拟合模块。

Handles PCA, NMF, and Linear Combination Fitting (LCF).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import xarray as xr

NMF: Any
PCA: Any
lmfit: Any

try:
    from sklearn.decomposition import NMF as _NMF
    from sklearn.decomposition import PCA as _PCA

    NMF = _NMF
    PCA = _PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    PCA = None
    NMF = None

try:
    import lmfit

    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False

logger = logging.getLogger(__name__)
ENERGY_ALIASES = ("energy_ev", "energyc", "energy")


def _energy_name(ds: xr.Dataset) -> str:
    """返回 Dataset 中使用的能量坐标名。"""
    for name in ENERGY_ALIASES:
        if name in ds.coords or name in ds.data_vars or name in ds.dims:
            return name
    raise ValueError(f"Dataset 缺少能量坐标，支持名称: {ENERGY_ALIASES}。")


def perform_pca(ds: xr.Dataset, n_components: Optional[int] = None, standardize: bool = True) -> xr.Dataset:
    """Perform Principal Component Analysis on XAS data.

    Args:
        ds: Input Dataset with 'absorption' and 'record' dimension.
        n_components: Number of components to keep. If None, all are kept.
        standardize: Whether to center the data before PCA (default True).
                     Note: sklearn PCA centers automatically.

    Returns:
        New Dataset containing:
        - pca_components: (component, energyc) - The eigenvectors (loadings).
        - pca_scores: (record, component) - The projections (weights).
        - pca_variance: (component) - Explained variance ratio.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for PCA.")

    if "record" not in ds.dims:
        raise ValueError("PCA requires a 'record' dimension (multiple spectra).")

    # 形状为 (n_samples, n_features)。
    x_data = ds.absorption.values.astype(float, copy=False)

    # PCA 不能直接处理 NaN。
    if np.isnan(x_data).any():
        logger.warning("数据包含 NaN，已用 0 填充，请谨慎解释结果。")
        x_data = np.nan_to_num(x_data)

    if standardize:
        x_data -= np.nanmean(x_data, axis=0)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(x_data)
    components = pca.components_
    variance = pca.explained_variance_ratio_

    n_comp_actual = components.shape[0]
    comp_coords = np.arange(1, n_comp_actual + 1)

    ds_out = xr.Dataset(coords=ds.coords)

    # Components (Loadings): depend on Energy
    energy_name = _energy_name(ds)

    ds_out["pca_components"] = xr.DataArray(
        components,
        dims=("component", energy_name),
        coords={"component": comp_coords, energy_name: ds[energy_name]},
    )

    # Scores (Weights): depend on Record
    ds_out["pca_scores"] = xr.DataArray(
        scores,
        dims=("record", "component"),
        coords={"record": ds.record, "component": comp_coords},
    )

    # Variance
    ds_out["pca_variance"] = xr.DataArray(variance, dims="component", coords={"component": comp_coords})

    # Copy metadata attributes
    ds_out.attrs = ds.attrs.copy()
    ds_out.attrs["pca_n_components"] = n_comp_actual

    return ds_out


def perform_nmf(ds: xr.Dataset, n_components: int = 2, init: str = "nndsvda") -> xr.Dataset:
    """Perform Non-negative Matrix Factorization.

    Args:
        ds: Input Dataset with 'absorption' (>0).
        n_components: Number of components.
        init: Initialization method for NMF.

    Returns:
        New Dataset with nmf_components and nmf_scores.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for NMF.")

    if "record" not in ds.dims:
        raise ValueError("NMF requires 'record' dimension.")

    x_data = ds.absorption.values
    if np.any(x_data < 0):
        logger.warning("数据包含负值，NMF 要求非负输入，已截断到 0。")
        x_data = np.maximum(x_data, 0)

    if np.isnan(x_data).any():
        x_data = np.nan_to_num(x_data)

    nmf = NMF(n_components=n_components, init=init, max_iter=1000)
    scores = nmf.fit_transform(x_data)
    components = nmf.components_

    comp_coords = np.arange(1, n_components + 1)

    ds_out = xr.Dataset(coords=ds.coords)

    energy_name = _energy_name(ds)

    ds_out["nmf_components"] = xr.DataArray(
        components,
        dims=("component", energy_name),
        coords={"component": comp_coords, energy_name: ds[energy_name]},
    )

    ds_out["nmf_scores"] = xr.DataArray(
        scores,
        dims=("record", "component"),
        coords={"record": ds.record, "component": comp_coords},
    )

    return ds_out


def perform_lcf(  # noqa: PLR0914, PLR0915
    ds: xr.Dataset,
    references: dict[str, np.ndarray | xr.DataArray],
    energy_range: Optional[tuple[float, float]] = None,
    sum_to_one: bool = True,
    non_negative: bool = True,
) -> xr.Dataset:
    """Perform Linear Combination Fitting on each record.

    Model: Data = sum(weight_i * ref_i)

    Args:
        ds: Input Dataset (target).
        references: Dictionary of {name: spectrum}. Spectra must be on same energy grid!
                    If they are not, use preprocessing.align_spectra first.
        energy_range: (min, max) energy to fit.
        sum_to_one: Constraint weights sum to 1.
        non_negative: Constraint weights >= 0.

    Returns:
        Dataset with:
        - lcf_weights: (record, ref_name)
        - lcf_fit: (record, energyc) - The reconstructed fit
        - lcf_residual: (record, energyc)
        - lcf_rfactor: (record)
    """
    if not HAS_LMFIT:
        raise ImportError("lmfit is required for LCF.")

    # Prepare data
    energy_name = _energy_name(ds)
    energy = ds[energy_name].values

    # Mask range
    mask = (energy >= energy_range[0]) & (energy <= energy_range[1]) if energy_range else np.ones_like(energy, dtype=bool)

    # Prepare References (matrix)
    ref_names = list(references.keys())
    ref_matrix = []

    for name in ref_names:
        ref_data = references[name]
        val = ref_data.values if isinstance(ref_data, xr.DataArray) else np.array(ref_data)

        # Ensure shape matches
        if val.shape != energy.shape:
            # Basic check. Ideally we should interpolate here if mismatch,
            # but we assume pre-alignment for performance.
            raise ValueError(f"Reference {name} shape {val.shape} does not match data {energy.shape}.")

        ref_matrix.append(val[mask])

    ref_matrix = np.array(ref_matrix).T  # (n_points, n_refs)

    # Function to minimize
    def residual(params: Any, y_data: Any, refs: Any) -> Any:
        model = np.zeros_like(y_data)
        for i, name in enumerate(ref_names):
            model += params[name].value * refs[:, i]
        return y_data - model

    # Setup Parameters
    params = lmfit.Parameters()
    for name in ref_names:
        params.add(name, value=1.0 / len(ref_names), min=0 if non_negative else -np.inf)

    if sum_to_one and len(ref_names) > 1:
        # 约束：最后一个参考谱权重 = 1 - 其他参考谱权重之和。
        expr = "1 - (" + " + ".join(ref_names[:-1]) + ")"
        params[ref_names[-1]].set(expr=expr)

    # Iterate over records
    if "record" in ds.dims:
        records = ds.record.values
        mu_all = ds.absorption.values
    else:
        records = [0]
        mu_all = ds.absorption.values[np.newaxis, :]

    weights_list = []
    fit_list = []
    resid_list = []
    rfactor_list = []

    for i in range(len(records)):
        mu_exp = mu_all[i][mask]

        result = lmfit.minimize(residual, params, args=(mu_exp, ref_matrix))

        # Extract weights
        # Explicitly ignore type check for result.params
        res_params = result.params
        w = [res_params[name].value for name in ref_names]
        weights_list.append(w)

        # Calculate full model
        full_ref_list = []
        for name in ref_names:
            ref_item = references[name]
            if isinstance(ref_item, xr.DataArray):
                full_ref_list.append(ref_item.values)
            else:
                full_ref_list.append(ref_item)

        full_ref_matrix = np.array(full_ref_list).T

        model_full = np.zeros_like(energy)
        for j, _name in enumerate(ref_names):
            model_full += w[j] * full_ref_matrix[:, j]

        fit_list.append(model_full)
        resid_list.append(mu_all[i] - model_full)
        rfactor_list.append(result.redchi)

    # Build Output
    ds_out = ds.copy()

    ds_out["lcf_weights"] = xr.DataArray(
        weights_list,
        dims=("record", "reference"),
        coords={
            "record": ds.record if "record" in ds.dims else [0],
            "reference": ref_names,
        },
    )

    ds_out["lcf_fit"] = xr.DataArray(
        np.asarray(fit_list),
        dims=("record", energy_name),
        coords={"record": ds.record if "record" in ds.dims else [0], energy_name: energy},
    )

    ds_out["lcf_residual"] = xr.DataArray(
        np.asarray(resid_list),
        dims=("record", energy_name),
        coords={"record": ds.record if "record" in ds.dims else [0], energy_name: energy},
    )

    ds_out["lcf_rfactor"] = xr.DataArray(
        rfactor_list,
        dims="record",
        coords={"record": ds.record if "record" in ds.dims else [0]},
    )

    return ds_out
