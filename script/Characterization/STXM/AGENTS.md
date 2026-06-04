# STXM Analysis Framework Knowledge Base

**Last Updated:** 2026-01-15
**Context**: Soft X-ray Transmission Microscopy (STXM) Data Analysis & Processing.

## OVERVIEW

This directory contains tools and notebooks for processing STXM data, specifically focused on spectromicroscopy of battery materials (e.g., Li-rich cathodes, Zn-MnO2). The core logic is encapsulated in `stxm_utils.py`.

## CORE MODULE: `stxm_utils.py`

### 1. Constants & Configuration

**Element Parameters Database** (`ELEMENT_PARAMS`):
| Element | Edge | Full Energy Range | Pre-edge Fit | Main Fit | Mapping Range | Peak Energies (L3, L2) |
|---------|------|-------------------|--------------|----------|---------------|------------------------|
| Mn | L3 | 600-700 eV | 625-635 eV | 630-670 eV | 638-644 eV | (640.3, 651.4) |
| Zn | L3 | 1000-1060 eV | 1015-1025 eV | 1010-1050 eV | 1020-1025 eV | (1021.8, 1044.9) |
| Fe | L3 | 690-740 eV | 695-705 eV | 700-730 eV | 705-715 eV | (708.65, 721.65) |
| O | K | 500-600 eV | 520-528 eV | 525-560 eV | 530-535 eV | - |

### 2. Data Structure & Preprocessing

| Function | Purpose | I/O |
|----------|---------|-----|
| `load_stxm_signal(file_path)` | Loads HDF5 data from MISTRAL beamline | `Path` → `Signal2D` (E, Y, X) |
| `align_and_transpose_stack(signal)` | Cross-correlation alignment + transpose | `Signal2D` → `Signal1D` (Y, X \| E) |
| `interpolate_energy_axis(signal, new_scale, degree)` | Uniform energy grid interpolation | `Signal1D` → `Signal1D` |
| `convert_to_optical_density(signal)` | Transmission to OD: `-log(T)` | `Signal1D` → `Signal1D` |

### 3. Denoising & Correction

**PCA Denoising** (`apply_pca_denoising`):
- Uses SVD decomposition with automatic elbow detection
- `components_number`: If `None`, auto-estimates using elbow method
- Returns reconstructed signal with noise suppression

**Saturation Correction** (`optimize_kappa`):
- Corrects for stray light (κ) using the formula:
  ```
  A_true = -ln((1+κ)·exp(-A_meas) - κ)
  ```
- **References**: Tonti et al. (2021, 2025)
- **Method**: Two-stage optimization:
  1. Coarse search via `_coarse_kappa_search`
  2. Fine optimization via `_refine_kappa`
- **Note**: Thickness ratio `c` is diagnostic only

### 4. Region of Interest (ROI) Management

**`ROI` Dataclass**:
```python
@dataclass
class ROI:
    energy: tuple[float, float] | None  # (min_eV, max_eV)
    element: str | None                  # "Mn", "Zn", etc.
    spatial: np.ndarray | None           # Boolean mask (True = Keep)
    hyperspy_roi: BaseROI | None         # HyperSpy interactive ROI
```

**Mask Convention**: `True = Keep` (inside ROI), `False = Exclude`

### 5. Advanced Analysis

**Onset Energy** (`calculate_onset_energy`):
- 10% max-intensity threshold method (Tan et al., 2012)
- Sub-pixel interpolation via `_interp_onset`
- Returns `(onset_map, flattened_values)`

**Clustering**:

| Method | Algorithm | Key Features |
|--------|-----------|--------------|
| `perform_clustering_analysis` | KMeans, sklearn | Auto-estimates n_clusters, excludes 1st PCA component |
| `perform_umap_hdbscan_clustering` | UMAP + HDBSCAN | Non-linear manifold, density-based, cosine metric |

**Model Fitting** (`fit_stepwise`):
- Stepwise peak fitting with configurable freezing strategies
- Supports: Offset, PowerLaw, Polynomial, Gaussian, DoubleStep components
- Freeze strategies: `"after_fit"`, `"partial"`, `"final_only"`, `"none"`
- `fit_global_first=True`: Fits mean spectrum first for better initial guesses

**Custom Component: `DoubleStep`**:
- Double Arctan Step function for L-edge continuum background
- Element-specific positions from `ELEMENT_PARAMS`
- Statistical branching ratio constraint: L2 height = 0.5 × L3 height

### 6. Main Analysis Class: `TXMAnalyzer`

**Key Methods**:
| Method | Description |
|--------|-------------|
| `preprocess(new_energy_scale=0.1)` | Aligns, transposes, interpolates, and converts to OD. |
| `denoise(components_number=None)` | Performs PCA denoising (auto-elbow if None). |
| `remove_background(element=...)` | Removes pre-edge background using PowerLaw/Offset/Polynomial. |
| `apply_b_value_correction(mask_ranges)` | Performs saturation correction (optimization of κ). |
| `set_roi(name, element, ...)` | Defines Region of Interest (spatial/energy). |
| `perform_clustering(rois, ...)` | K-Means clustering on PCA loadings. |
| `perform_umap_clustering(rois, ...)` | UMAP + HDBSCAN clustering. |
| `optimize_umap_params(rois, ...)` | **NEW**: Auto-tune UMAP+HDBSCAN params using Optuna. |
| `fit(peak_configs, rois, ...)` | Stepwise model fitting. |
| `get_fit_maps(name)` | Retrieves parameter maps from fitted model. |
| `export_to_datatree()` | Exports all results to `xarray.DataTree` structure. |
| `save(output_path, format)` | Saves to HDF5, NetCDF, or Zarr format. |

## INTERNAL HELPERS (Refactored 2026-01-15)

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `_b_value_model_func` | `_saturation_model` | Saturation correction model |
| `_compute_linear_correlation` | `_calc_r2` | R² calculation for kappa optimization |
| `_get_thin_thick_spectra` | `_split_spectra` | Split thin/thick regions |
| `_coarse_search_b_value` | `_coarse_kappa_search` | Coarse threshold search |
| `_refine_b_value` | `_refine_kappa` | Refine kappa parameters |
| `_fit_b_value_params` | `_fit_kappa_params` | Final kappa fitting |
| `optimize_saturation_params` | `optimize_kappa` | Main kappa optimization |
| `_get_energy_axis_and_datacube` | `_get_energy_data` | Get energy axis + data |
| `_find_onset_indices` | `_onset_indices` | Find onset threshold indices |
| `_linear_interpolate_onset` | `_interp_onset` | Sub-pixel interpolation |
| `_ensure_decomposition` | `_ensure_pca` | Ensure PCA is done |
| `_get_decomposition_signal` | `_get_pca_loadings` | Get PCA loadings |
| `_get_flattened_features` | `_flatten_features` | Flatten for sklearn |
| `_calculate_cluster_means` | `_cluster_means` | Cluster mean spectra |
| `_set_component_params` | `_init_component` | Initialize component |
| `_freeze_component_params` | `_freeze_params` | Freeze parameters |
| `_unfreeze_all_params` | `_unfreeze_all` | Unfreeze all |
| `fit_model_stepwise` | `fit_stepwise` | Stepwise fitting |
| `optimize_clustering_params` | `optimize_clustering_params` | **NEW**: Optuna objective function |

## USAGE WORKFLOW

```python
from stxm_utils import TXMAnalyzer, get_element_params

# 1. Initialize & Preprocess
analyzer = TXMAnalyzer("path/to/data.h5")
analyzer.preprocess(new_energy_scale=0.1)
analyzer.denoise(components_number=None)  # Auto-detect

# 2. Saturation Correction (Optional)
best_k, popt, r2 = analyzer.apply_b_value_correction(element="Mn")
print(f"Kappa (stray light): {popt[0]:.3f}, R²: {r2:.4f}")

# 3. Background Removal
analyzer.remove_background(element="Mn")

# 4. Define ROIs
analyzer.set_roi(element=["Mn", "Zn"])

# 5. Analysis & Optimization
# NEW: Automatically find best clustering parameters
study = analyzer.optimize_umap_params(rois="Mn", n_trials=50)
print("Best params:", study.best_params)

# Apply optimized parameters
analyzer.perform_umap_clustering(
    rois="Mn",
    exclude_first_component=True,
    **study.best_params
)

# 6. Model Fitting
peak_configs = [
    {"type": "Gaussian", "name": "L3", "param_guesses": {"centre": 640.0, "sigma": 1.0, "height": 1.0}},
    {"type": "Gaussian", "name": "L2", "param_guesses": {"centre": 651.0, "sigma": 1.0, "height": 0.5}},
    {"type": "DoubleStep", "name": "Step", "element": "Mn"},
]
analyzer.fit(peak_configs=peak_configs, rois="Mn", fit_global_first=True)

# 7. Export & Save
dt = analyzer.export_to_datatree()
analyzer.save("output.h5", format="hdf5")
```

## KEY DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| `hyperspy[all]` | `==2.3.0` | Core signal processing (pinned) |
| `scikit-learn` | `>=1.8.0` | Clustering (KMeans, HDBSCAN) |
| `umap-learn` | `>=0.5.9` | Manifold learning |
| `scipy` | `>=1.15.3` | Optimization |
| `h5py` | `>=3.14.0` | HDF5 I/O |
| `xarray` | `>=2025.6.1` | DataTree export |
| `optuna` | `>=4.6.0` | **NEW**: Parameter optimization |
| `optuna-dashboard` | `>=0.20.0` | **NEW**: Visualization of optimization |


## CONVENTIONS & ANTI-PATTERNS

### Conventions
- **Mask**: `True = Keep`, consistent across all functions
- **Signal axes**: After preprocessing, always `(Y, X | Energy)` layout
- **Energy units**: Always in eV
- **Lazy imports**: `umap` and `sklearn.cluster.HDBSCAN` are imported on-demand

### Anti-Patterns
- **DO NOT** use `c_val` (thickness ratio) for per-pixel correction
- **DO NOT** mix mask conventions
- **DO NOT** hardcode element parameters - use `ELEMENT_PARAMS`
- **DO NOT** access `param.map` directly without checking `hasattr(param, "map")`

## REFERENCES

1. **Vogt, S. (2004)**: "Cluster analysis of soft X-ray spectromicroscopy data." *Ultramicroscopy*, 99, 149-157.
2. **Tonti, D. et al. (2021)**: "Soft X-Ray Transmission Microscopy on Lithium-Rich Layered-Oxide Cathode Materials." *Applied Sciences*, 11(4), 1870.
3. **Tonti, D. et al. (2025)**: "Unveiling Capacity Limitations of MnO2 in Rechargeable Zn Chemistry." *Energy & Environmental Science*.
4. **Tan, H. et al. (2012)**: "Oxidation state and chemical shift investigation in transition metal oxides by EELS." *Ultramicroscopy*, 116, 24-33.
5. **Blanco-Portals, J. et al. (2022)**: "Strategies for EELS Data Analysis. Introducing UMAP and HDBSCAN." *Microscopy and Microanalysis*, 28(1), 109-122.
6. **Liebscher, B. et al. (2002)**: "Quantification of Ferrous/Ferric Ratios in Minerals." *Physics and Chemistry of Minerals*, 29, 579-588.
