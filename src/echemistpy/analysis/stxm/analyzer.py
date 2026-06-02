# -*- coding: utf-8 -*-
"""STXM Data Analyzer for chemical mapping and spectroscopy analysis."""

from __future__ import annotations

import logging

import lmfit
import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.optimize
import umap
import xarray as xr
from echemistpy.processing.analyzers.registry import TechniqueAnalyzer
from skimage.registration import phase_cross_correlation
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from traitlets import Bool, Dict, Float, Int, List, Unicode

from echemistpy.io.structures import AnalysisData, AnalysisDataInfo, RawData

logger = logging.getLogger(__name__)


def b_value_model(x: np.ndarray, b: float, c: float) -> np.ndarray:
    """B-value model for thickness correction with improved numerical stability.

    Equation: I_thick = -ln((exp(-I_thin * C) + B) / (1 + B))
    Rewritten as: ln(1 + B) - ln(exp(-I_thin * C) + B)
    """
    # Enforce non-negative b for physical sense and stability during fitting exploration
    b_safe = max(b, 0.0)

    # Calculate terms
    # Term 1: ln(1 + B). Use log1p for precision when B is small
    term1 = np.log1p(b_safe)

    # Term 2: ln(exp(...) + B)
    # Prevent exp overflow/underflow (though x*c usually > 0, so exp -> 0)
    exp_val = np.exp(-x * c)

    # Add epsilon to prevent log(0) if exp_val -> 0 and b -> 0
    term2 = np.log(exp_val + b_safe + 1e-15)

    return term1 - term2


def build_lmfit_model(config: dict) -> tuple[lmfit.Model, lmfit.Parameters]:  # noqa: PLR0912
    """Build a composite model from configuration dictionary using lmfit."""
    components = config.get("components", [])
    if not components:
        raise ValueError("No components defined for model")

    composite_model = None
    params = lmfit.Parameters()

    for i, comp in enumerate(components):
        ctype = comp["type"].lower()
        prefix = f"c{i}_"

        # Select model component
        if ctype == "gaussian":
            model = lmfit.models.GaussianModel(prefix=prefix)
        elif ctype == "lorentzian":
            model = lmfit.models.LorentzianModel(prefix=prefix)
        elif ctype == "linear":
            model = lmfit.models.LinearModel(prefix=prefix)
        elif ctype in {"arctan", "step"}:
            model = lmfit.models.StepModel(prefix=prefix, form="arctan")
        else:
            logger.warning("Unknown component type: %s, skipping", ctype)
            continue

        # Initialize Parameters
        comp_params = comp.get("params", {})
        bounds = comp.get("bounds", {})

        # Standard lmfit param names
        model_params = model.make_params()

        # Update values from config
        for pname, pval in comp_params.items():
            full_name = f"{prefix}{pname}"
            if full_name in model_params:
                model_params[full_name].set(value=pval)

        # Apply bounds if provided
        lower_bounds = bounds.get("lower")
        upper_bounds = bounds.get("upper")

        # Define order map for supported types
        param_order = []
        if ctype in {"gaussian", "lorentzian"} or ctype in {"arctan", "step"}:
            param_order = ["amplitude", "center", "sigma"]
        elif ctype == "linear":
            param_order = ["slope", "intercept"]

        if lower_bounds and len(lower_bounds) == len(param_order):
            for pname, lb in zip(param_order, lower_bounds, strict=False):
                full_name = f"{prefix}{pname}"
                if full_name in model_params:
                    model_params[full_name].set(min=lb)

        if upper_bounds and len(upper_bounds) == len(param_order):
            for pname, ub in zip(param_order, upper_bounds, strict=False):
                full_name = f"{prefix}{pname}"
                if full_name in model_params:
                    model_params[full_name].set(max=ub)

        # Add to composite
        if composite_model is None:
            composite_model = model
        else:
            composite_model += model

        params.update(model_params)

    if composite_model is None:
        raise ValueError("No valid components found in configuration")

    return composite_model, params


class STXMAnalyzer(TechniqueAnalyzer):
    """Analyzer for Scanning Transmission X-ray Microscopy (STXM) data.

    Workflow:
    1. Preprocessing (Alignment, Energy interpolation)
    2. Denoising (PCA)
    3. Background Removal (Pre-edge linear subtraction)
    4. Thickness Correction (B-Value method)
    5. Chemical Analysis (ROI Mapping, Clustering)
    6. Spectrum Fitting
    """

    technique = Unicode("txm")
    name = Unicode("STXMAnalyzer")

    # Configuration traits
    energy_step = Float(0.1, help="Energy step for interpolation (eV)")
    align_images = Bool(True, help="Perform image alignment during preprocessing")
    alignment_method = Unicode("phase_correlation", help="Method used for image alignment")
    alignment_upsample_factor = Int(10, help="Upsample factor for subpixel alignment precision")

    pca_components = Int(5, help="Number of PCA components for denoising")

    # UMAP configuration
    use_umap = Bool(False, help="Enable UMAP dimensionality reduction")
    umap_n_components = Int(2, help="Dimension of the embedded space (typically 2 for visualization)")
    umap_n_neighbors = Int(15, help="Number of neighbors for UMAP")
    umap_min_dist = Float(0.1, help="Minimum distance between points in embedding")
    umap_metric = Unicode("euclidean", help="Metric to use for UMAP")

    pre_edge_range = List(Float(), default_value=[625.0, 635.0], help="Pre-edge energy range for background removal (eV)")
    roi_maps = Dict(key_trait=Unicode(), value_trait=List(Float()), help="Dictionary of ROI definitions: {name: [start, end]}", default_value={})
    spatial_rois = Dict(key_trait=Unicode(), value_trait=List(Float()), help="Dictionary of Spatial ROI definitions: {name: [x_start, x_end, y_start, y_end]}", default_value={})
    roi_ranges = List(help="Legacy: List of (start, end) tuples. Use roi_maps instead.")
    clustering_method = Unicode("kmeans", help="Clustering algorithm: kmeans, minibatch_kmeans, gmm, dbscan")
    clustering_params = Dict(help="Additional parameters for clustering algorithm", default_value={})
    n_clusters = Int(3, help="Number of clusters for K-means segmentation")

    # Model Fitting configuration
    fitting_models = Dict(
        key_trait=Unicode(),
        value_trait=Dict(),
        help="Dictionary of model configurations for fitting spectra. Key is name, Value is dict with 'components', 'ranges', 'targets'",
        default_value={},
    )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("optical_density",)

    @staticmethod
    def _get_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
        """Detect spatial dimension names from DataArray."""
        y_candidates = ["y", "y_um", "y_nm", "y_px", "Y"]
        x_candidates = ["x", "x_um", "x_nm", "x_px", "X"]

        y_dim = None
        x_dim = None

        for y_name in y_candidates:
            if y_name in da.dims:
                y_dim = y_name
                break

        for x_name in x_candidates:
            if x_name in da.dims:
                x_dim = x_name
                break

        if y_dim is None or x_dim is None:
            # Fallback: assume last two dims are spatial
            non_energy_dims = [str(d) for d in da.dims if "energy" not in str(d).lower()]
            if len(non_energy_dims) >= 2:
                y_dim, x_dim = non_energy_dims[-2:]
            else:
                raise ValueError(f"Could not detect spatial dimensions from {da.dims}")

        return str(y_dim), str(x_dim)

    # Step 1: Align
    def align_stack(self, ds: xr.Dataset | xr.DataTree) -> xr.Dataset:  # noqa: PLR0914
        """Align image stack to correct for drift using Scikit-image."""
        if isinstance(ds, xr.DataTree):
            if ds.dataset is None:
                raise TypeError("align_stack requires a Dataset or DataTree with a root dataset")
            ds = ds.dataset

        if "optical_density" not in ds:
            return ds

        # Work on a copy
        ds = ds.copy(deep=True)
        da = ds["optical_density"]

        # Ensure energy_eV is first dimension for iterating
        if "energy_eV" in da.dims:
            da_aligned = da.transpose("energy_eV", ...)
        else:
            return ds

        # Work on numpy array
        data = da_aligned.values.copy()
        n_images = data.shape[0]
        if n_images < 2:
            return ds

        # Reference: Middle image
        ref_idx = n_images // 2
        ref_img = np.nan_to_num(data[ref_idx])

        shifts = []
        logger.info("Aligning stack to frame %d using %s...", ref_idx, self.alignment_method)

        for i in range(n_images):
            if i == ref_idx:
                shifts.append((0.0, 0.0))
                continue

            curr_img = np.nan_to_num(data[i])
            dy, dx = 0.0, 0.0

            if self.alignment_method == "phase_correlation":
                try:
                    shift, _, _ = phase_cross_correlation(ref_img, curr_img, upsample_factor=self.alignment_upsample_factor)
                    # shift is [y, x] to move current to ref
                    dy, dx = float(shift[0]), float(shift[1])
                except Exception as e:
                    logger.debug("Alignment failed for frame %d: %s", i, e)

            elif self.alignment_method == "center_of_mass":
                try:
                    cy_ref, cx_ref = scipy.ndimage.center_of_mass(ref_img)
                    cy_curr, cx_curr = scipy.ndimage.center_of_mass(curr_img)
                    dy = cy_ref - cy_curr
                    dx = cx_ref - cx_curr
                except Exception as e:
                    logger.debug("Center of mass alignment failed for frame %d: %s", i, e)

            shifts.append((dy, dx))

            # Apply shift
            data[i] = scipy.ndimage.shift(data[i], (dy, dx), order=1, mode="nearest")

        # Update dataset
        ds["optical_density"] = (da_aligned.dims, data)
        ds.attrs["alignment_shifts"] = shifts

        return ds

    # Step 2: Interpolate
    def interpolate_energy(self, ds: xr.Dataset | xr.DataTree) -> xr.Dataset:
        """Interpolate energy axis to uniform grid."""
        if isinstance(ds, xr.DataTree):
            if ds.dataset is None:
                raise TypeError("interpolate_energy requires a Dataset or DataTree with a root dataset")
            ds = ds.dataset

        # Work on a copy
        ds = ds.copy(deep=True)

        if "energy_eV" not in ds.coords:
            logger.warning("No 'energy_eV' coordinate found. Skipping interpolation.")
            return ds

        energy = ds.coords["energy_eV"].values
        if len(energy) < 2:
            return ds

        # Handle duplicate energy values
        if not ds.indexes["energy_eV"].is_unique:
            logger.warning("Duplicate energy values found. Averaging duplicates.")
            ds = ds.drop_duplicates("energy_eV")
            energy = ds.coords["energy_eV"].values

        # Create uniform energy grid
        e_min, e_max = energy.min(), energy.max()
        new_energy = np.arange(e_min, e_max + self.energy_step, self.energy_step)

        # Interpolate
        # xarray's interp handles all variables with energy_eV dim
        cleaned_ds = ds.interp(energy_eV=new_energy, method="linear", kwargs={"fill_value": "extrapolate"})

        return cleaned_ds

    def preprocess(self, raw_data: RawData) -> RawData:
        """Run preprocessing steps: Alignment and Interpolation."""
        ds = raw_data.data.copy(deep=True)

        if self.align_images:
            try:
                ds = self.align_stack(ds)
            except Exception as e:
                logger.warning("Image alignment failed: %s", e)

        try:
            ds = self.interpolate_energy(ds)
        except Exception as e:
            logger.warning("Energy interpolation failed: %s", e)

        return RawData(data=ds)

    # Step 3: Denoise (PCA)
    def denoise_pca(self, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
        """Perform PCA denoising on the data."""
        ds = ds.copy(deep=True)
        results = {}

        if "optical_density" not in ds:
            logger.warning("Missing 'optical_density' for PCA.")
            return ds, results

        da = ds["optical_density"]
        if "energy_eV" in da.dims and da.dims[0] != "energy_eV":
            da = da.transpose("energy_eV", ...)

        original_shape = da.shape
        n_energy = original_shape[0]
        flat_data = da.values.reshape(n_energy, -1)

        mask_nan = np.isnan(flat_data)
        flat_data_clean = np.nan_to_num(flat_data) if mask_nan.any() else flat_data

        try:
            # PCA: X is (n_samples=pixels, n_features=energy)
            # sklearn PCA expects (n_samples, n_features).
            # Here we want to decompose the SPECTRA (features=energy channels) or PIXELS?
            # Standard PCA for image denoising: Treat each PIXEL as a sample, and ENERGY as features.
            # So input should be (n_pixels, n_energy).
            x_input = flat_data_clean.T  # Shape: (n_pixels, n_energy)

            # Standardize? Usually for PCA on spectra we center but might not scale if units are same.
            # Mantis normalizes each pixel vector to unit length?
            # Let's keep it simple: Centering is done by PCA automatically.

            n_comp = min(self.pca_components, *x_input.shape)
            pca = PCA(n_components=n_comp)

            x_transformed = pca.fit_transform(x_input)
            x_reconstructed = pca.inverse_transform(x_transformed)

            reconstructed_flat = x_reconstructed.T  # Back to (n_energy, n_pixels)

            if mask_nan.any():
                reconstructed_flat[mask_nan] = np.nan

            denoised_data = reconstructed_flat.reshape(original_shape)

            ds["denoised"] = (da.dims, denoised_data)
            results["pca_components"] = n_comp
            results["pca_explained_variance_ratio"] = pca.explained_variance_ratio_
            results["pca_singular_values"] = pca.singular_values_
            results["pca_components_matrix"] = pca.components_

            # Additional statistics for Scree Plot
            # To get full eigenvalues (for scree plot beyond n_components), we might need more components
            # But for performance, we stick to n_comp or a slightly larger number if requested?
            # Let's just return what we have.
            results["pca_explained_variance"] = pca.explained_variance_

        except Exception as e:
            logger.error("PCA failed: %s", e)
            ds["denoised"] = da

        return ds, results

    # Step 4: Background Removal
    def remove_background(self, ds: xr.Dataset) -> xr.Dataset:
        """Remove pre-edge linear background."""
        ds = ds.copy(deep=True)
        # Use denoised if available, else original
        work_da = ds["denoised"] if "denoised" in ds else ds["optical_density"]

        e_coords = work_da.coords["energy_eV"].values
        mask_pre = (e_coords >= self.pre_edge_range[0]) & (e_coords <= self.pre_edge_range[1])

        if mask_pre.any():
            try:
                x_pre = e_coords[mask_pre]
                y_pre = work_da.isel(energy_eV=mask_pre).values.reshape(len(x_pre), -1)

                # Linear regression: y = mx + c
                x_mat = np.vstack([x_pre, np.ones(len(x_pre))]).T
                # beta: [slope, intercept]
                beta, _, _, _ = scipy.linalg.lstsq(x_mat, y_pre)

                x_full = np.vstack([e_coords, np.ones(len(e_coords))]).T
                background_flat = x_full @ beta
                background = background_flat.reshape(work_da.shape)

                ds["background_removed"] = work_da - background
            except Exception as e:
                logger.warning("Background removal failed: %s", e)
                ds["background_removed"] = work_da
        else:
            ds["background_removed"] = work_da

        return ds

    # Step 5: Thickness Correction (B-Value)
    def correct_thickness(self, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:  # noqa: PLR0914
        """Apply B-Value correction for thickness effects."""
        ds = ds.copy(deep=True)
        results = {}

        work_da = ds["background_removed"] if "background_removed" in ds else (ds["denoised"] if "denoised" in ds else ds["optical_density"])

        # Calculate sum image to find thin/thick regions
        intensity_map = work_da.sum(dim="energy_eV")
        flat_int = intensity_map.values.flatten()
        flat_int = flat_int[~np.isnan(flat_int)]

        if len(flat_int) > 0:
            p_low, p_high = np.percentile(flat_int, [10, 90])
            mask_thin = intensity_map < p_low
            mask_thick = intensity_map > p_high

            if mask_thin.any() and mask_thick.any():
                y_dim, x_dim = self._get_spatial_dims(work_da)
                spec_thin = work_da.where(mask_thin).mean(dim=[y_dim, x_dim]).values
                spec_thick = work_da.where(mask_thick).mean(dim=[y_dim, x_dim]).values

                try:
                    popt, _ = scipy.optimize.curve_fit(b_value_model, spec_thin, spec_thick, p0=[0.1, 1.0], bounds=([0, 0], [np.inf, np.inf]))
                    b_val, c_val = popt
                    results["b_value"] = float(b_val)
                    results["c_value"] = float(c_val)

                    arg = (1 + b_val) * np.exp(-work_da) - b_val
                    arg = xr.where(arg > 1e-9, arg, 1e-9)
                    ds["thickness_corrected"] = -np.log(arg)
                except Exception as e:
                    logger.warning("B-Value correction failed: %s", e)
                    ds["thickness_corrected"] = work_da
            else:
                ds["thickness_corrected"] = work_da
        else:
            ds["thickness_corrected"] = work_da

        return ds, results

    # Step 6: ROI Selection
    def apply_rois(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply ROI selections (Spectral mapping and Spatial extraction)."""
        ds = ds.copy(deep=True)
        work_da = ds.get("thickness_corrected", ds.get("background_removed", ds.get("denoised", ds.get("optical_density"))))

        # 1. Spectral ROIs -> Image Maps
        rois_to_process = {}
        # Legacy
        if self.roi_ranges:
            for start, end in self.roi_ranges:
                rois_to_process[f"roi_{start}_{end}"] = (start, end)
        # New dict
        for name, rng in self.roi_maps.items():
            if len(rng) >= 2:
                rois_to_process[name] = (rng[0], rng[1])

        for name, (start, end) in rois_to_process.items():
            s, e = sorted([start, end])
            mask_roi = (work_da.energy_eV >= s) & (work_da.energy_eV <= e)
            if mask_roi.any():
                roi_map = work_da.sel(energy_eV=slice(s, e)).mean(dim="energy_eV")
                ds[name] = roi_map

        # 2. Spatial ROIs -> Spectra
        if self.spatial_rois:
            for name, coords in self.spatial_rois.items():
                if len(coords) >= 4:
                    x1, x2, y1, y2 = sorted(coords[0:2]) + sorted(coords[2:4])
                    try:
                        y_dim, x_dim = self._get_spatial_dims(work_da)
                        # Assume coords are indices/pixels for simplicity or consistent with previous usage
                        # If using sel() with slices, xarray includes bounds.
                        roi_spec = work_da.sel({x_dim: slice(x1, x2), y_dim: slice(y1, y2)}).mean(dim=[y_dim, x_dim])
                        ds[f"spectrum_{name}"] = roi_spec
                    except Exception as e:
                        logger.warning("Spatial ROI %s processing failed: %s", name, e)

        return ds

    # Step 7: Clustering (UMAP + Sklearn)
    def cluster_analysis(self, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:  # noqa: PLR0912, PLR0915, PLR0914
        """Perform clustering analysis (UMAP and/or KMeans/DBSCAN/GMM)."""
        ds = ds.copy(deep=True)
        results = {}
        work_da = ds.get("thickness_corrected", ds.get("background_removed", ds.get("denoised", ds.get("optical_density"))))

        if "energy_eV" in work_da.dims and work_da.dims[0] != "energy_eV":
            work_da = work_da.transpose("energy_eV", ...)

        original_shape = work_da.shape
        n_energy = original_shape[0]
        flat_data = work_da.values.reshape(n_energy, -1).T  # (pixels, energy)

        valid_pixels = ~np.isnan(flat_data).any(axis=1)
        data_for_clustering = flat_data[valid_pixels]

        if len(data_for_clustering) == 0:
            return ds, results

        # UMAP
        if self.use_umap:
            try:
                reducer = umap.UMAP(
                    n_components=self.umap_n_components,
                    n_neighbors=self.umap_n_neighbors,
                    min_dist=self.umap_min_dist,
                    metric=self.umap_metric,
                    random_state=42,
                )
                embedding = reducer.fit_transform(data_for_clustering)

                # Reconstruct full maps
                ny, nx = original_shape[1], original_shape[2]
                full_embedding = np.full((ny, nx, self.umap_n_components), np.nan)
                full_flat = full_embedding.reshape(-1, self.umap_n_components)
                full_flat[valid_pixels] = embedding
                full_embedding = full_flat.reshape(ny, nx, self.umap_n_components)

                y_dim, x_dim = self._get_spatial_dims(work_da)
                ds["umap_embeddings"] = ((y_dim, x_dim, "umap_component"), full_embedding)
            except Exception as e:
                logger.warning("UMAP embedding failed: %s", e)

        # Clustering
        if len(data_for_clustering) > self.n_clusters:
            try:
                labels_valid = None
                centroids = None
                method = self.clustering_method.lower()
                kwargs = self.clustering_params.copy()

                if method == "kmeans":
                    n_c = kwargs.pop("n_clusters", self.n_clusters)
                    model = KMeans(n_clusters=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.cluster_centers_

                elif method == "minibatch_kmeans":
                    n_c = kwargs.pop("n_clusters", self.n_clusters)
                    model = MiniBatchKMeans(n_clusters=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.cluster_centers_

                elif method == "gmm":
                    n_c = kwargs.pop("n_components", self.n_clusters)
                    model = GaussianMixture(n_components=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.means_

                if method == "dbscan":
                    model = DBSCAN(**kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    unique_labels = set(labels_valid)
                    centroids_list = []
                    # DBSCAN labels -1 is noise
                    valid_labels_sorted = sorted({lbl for lbl in unique_labels if lbl != -1})
                    for lbl in valid_labels_sorted:
                        centroids_list.append(data_for_clustering[labels_valid == lbl].mean(axis=0))
                    centroids = np.array(centroids_list) if centroids_list else None

                else:
                    logger.warning("Unknown clustering method %s, falling back to KMeans", method)
                    model = KMeans(n_clusters=self.n_clusters)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.cluster_centers_

                if labels_valid is not None:
                    full_labels = np.full(flat_data.shape[0], -1)
                    full_labels[valid_pixels] = labels_valid
                    label_map = full_labels.reshape(original_shape[1], original_shape[2])
                    y_dim, x_dim = self._get_spatial_dims(work_da)
                    ds["cluster_labels"] = ((y_dim, x_dim), label_map)

                    if centroids is not None:
                        results["cluster_centroids"] = centroids

            except Exception as e:
                logger.warning("Clustering failed: %s", e)

        return ds, results

    # Step 8: Pixel-wise Fitting (and Spectrum Fitting)
    def fit_pixels(self, ds: xr.Dataset, cluster_centroids: np.ndarray | None = None) -> tuple[xr.Dataset, dict]:  # noqa: PLR0912, PLR0914
        """Perform fitting on identified spectra (ROIs or Clusters)."""
        ds = ds.copy(deep=True)
        results = {}
        fit_results = {}

        if not self.fitting_models:
            return ds, results

        work_da = ds.get("thickness_corrected", ds.get("background_removed", ds.get("denoised", ds.get("optical_density"))))
        energy_coords = work_da.coords["energy_eV"].values

        for model_name, config in self.fitting_models.items():
            targets = config.get("targets", [])
            fit_range = config.get("range", None)

            try:
                composite, params = build_lmfit_model(config)
            except ValueError as e:
                logger.warning("Skipping model %s: %s", model_name, e)
                continue

            if composite is None:
                continue

            # Identify targets
            spectra_to_fit = {}

            if "cluster_centroids" in targets and cluster_centroids is not None:
                for i, centroid in enumerate(cluster_centroids):
                    spectra_to_fit[f"Cluster_{i}"] = (energy_coords, centroid)

            for target in targets:
                if target.startswith("spectrum_") and target in ds:
                    da_spec = ds[target]
                    spectra_to_fit[target] = (da_spec.coords["energy_eV"].values, da_spec.values)

            # Perform Fitting
            model_fits = {}
            for spec_name, (x, y) in spectra_to_fit.items():
                if fit_range:
                    mask = (x >= fit_range[0]) & (x <= fit_range[1])
                    x_fit = x[mask]
                    y_fit = y[mask]
                else:
                    x_fit, y_fit = x, y

                if len(x_fit) == 0:
                    continue

                try:
                    result = composite.fit(y_fit, params, x=x_fit)

                    # Extract params for backward compatibility
                    ordered_values = []
                    for pname in result.params:
                        ordered_values.append(result.params[pname].value)

                    model_fits[spec_name] = {
                        "params": ordered_values,
                        "param_dict": result.best_values,
                        "chisqr": result.chisqr,
                        "fitted_curve": (x_fit, result.best_fit),
                    }

                    # Evaluate on full range
                    y_full_calc = composite.eval(result.params, x=x)
                    da_fit = xr.DataArray(y_full_calc, coords={"energy_eV": x}, dims="energy_eV", name=f"fit_{model_name}_{spec_name}")
                    ds[f"fit_{model_name}_{spec_name}"] = da_fit

                except Exception as e:
                    logger.warning("Fitting %s to %s failed: %s", model_name, spec_name, e)

            fit_results[model_name] = model_fits

        results["fitting_results"] = fit_results
        return ds, results

    # Step 9: Chemical Mapping (NNLS)
    def map_chemical_components(self, ds: xr.Dataset, references: dict[str, np.ndarray]) -> tuple[xr.Dataset, dict]:  # noqa: PLR0914
        """Perform Linear Combination Fitting (NNLS) to map chemical components.

        Args:
            ds: Dataset containing the data (usually 'thickness_corrected' or 'denoised')
            references: Dictionary of {component_name: spectrum_array}

        Returns:
            Updated Dataset with component maps and residuals.
        """
        ds = ds.copy(deep=True)
        results = {}

        if not references:
            return ds, results

        work_da = ds.get("thickness_corrected", ds.get("background_removed", ds.get("denoised", ds.get("optical_density"))))

        # Align references to data energy
        energy = work_da.coords["energy_eV"].values

        # Build design matrix A (n_energy, n_components)
        component_names = sorted(references.keys())
        design_matrix_list = []
        valid_refs = []

        for name in component_names:
            ref_spec = references[name]
            if len(ref_spec) != len(energy):
                # Simple check, in production might need interpolation
                logger.warning("Reference %s length %d mismatch with energy %d. Skipping.", name, len(ref_spec), len(energy))
                continue
            design_matrix_list.append(ref_spec)
            valid_refs.append(name)

        if not design_matrix_list:
            return ds, results

        design_matrix = np.column_stack(design_matrix_list)  # (n_energy, n_components)

        # Prepare data B (n_energy, n_pixels)
        if "energy_eV" in work_da.dims and work_da.dims[0] != "energy_eV":
            work_da = work_da.transpose("energy_eV", ...)

        original_shape = work_da.shape
        n_energy = original_shape[0]
        flat_data = work_da.values.reshape(n_energy, -1)  # (n_energy, n_pixels)

        # NNLS for each pixel
        # Solves min|| Ax - b ||_2 for x >= 0
        # Scipy nnls is for single vector. For matrix, we loop or use optimization.
        # Since this can be slow, we iterate.

        n_pixels = flat_data.shape[1]
        n_comps = len(valid_refs)
        maps = np.zeros((n_comps, n_pixels))
        residuals = np.zeros(n_pixels)

        # mask nans
        mask_nan = np.isnan(flat_data).any(axis=0)
        valid_indices = np.where(~mask_nan)[0]

        # Optimization: Pre-compute if possible?
        # NNLS is iterative, hard to vectorize simply without external libs like cvxopt or special solvers.
        # We'll use a loop for now, maybe optimized later.

        for idx in valid_indices:
            b = flat_data[:, idx]
            x, rnorm = scipy.optimize.nnls(design_matrix, b)
            maps[:, idx] = x
            residuals[idx] = rnorm

        # Reshape back to images
        y_dim, x_dim = self._get_spatial_dims(work_da)
        ny, nx = original_shape[1], original_shape[2]

        reshaped_maps = maps.reshape(n_comps, ny, nx)
        reshaped_resid = residuals.reshape(ny, nx)

        # Store in Dataset
        for i, name in enumerate(valid_refs):
            ds[f"map_{name}"] = ((y_dim, x_dim), reshaped_maps[i])

        ds["nnls_residual"] = ((y_dim, x_dim), reshaped_resid)
        results["mapped_components"] = valid_refs

        return ds, results

    def _compute(self, raw_data: RawData) -> tuple[AnalysisData, AnalysisDataInfo]:
        """Execute the full STXM analysis workflow."""
        ds = raw_data.data.copy(deep=True)
        results = {}
        params_dict = {}

        # 1. Denoising
        ds, pca_info = self.denoise_pca(ds)
        results.update(pca_info)
        if "pca_components" in pca_info:
            params_dict["pca_components"] = pca_info["pca_components"]

        # 2. Background Removal
        ds = self.remove_background(ds)

        # 3. Thickness Correction
        ds, bval_info = self.correct_thickness(ds)
        results.update(bval_info)
        if "b_value" in bval_info:
            params_dict["b_value"] = bval_info["b_value"]
            params_dict["c_value"] = bval_info["c_value"]

        # 4. ROI
        ds = self.apply_rois(ds)

        # 5. Clustering
        ds, clust_info = self.cluster_analysis(ds)
        results.update(clust_info)

        # 6. Fitting (Peak Fitting)
        centroids = clust_info.get("cluster_centroids")
        ds, fit_info = self.fit_pixels(ds, cluster_centroids=centroids)
        results.update(fit_info)

        # 7. Chemical Mapping (NNLS using Cluster Centroids as references)
        # This allows "auto-mapping" based on identified clusters
        if centroids is not None:
            # Create reference dict from centroids
            # centroids shape: (n_clusters, n_energy)
            refs = {f"Cluster_{i}": centroid for i, centroid in enumerate(centroids)}
            ds, map_info = self.map_chemical_components(ds, refs)
            results.update(map_info)

        analysis_data = AnalysisData(data=ds)
        analysis_info = AnalysisDataInfo(parameters=params_dict, others=results)

        return analysis_data, analysis_info
