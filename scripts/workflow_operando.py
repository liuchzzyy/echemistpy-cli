"""
Workflow Full: Comprehensive XAS Analysis Pipeline
Demonstrates: Load -> Deglitch -> Smooth -> Align -> Normalize -> PCA -> LCF.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from claess2025.io import load
from claess2025.analysis import (
    process_dataset,
    get_element_config,
    deglitch,
    smooth,
    align_spectra,
    calibrate_energy,
    merge_spectra,
    perform_pca,
    perform_lcf,
    plot_echem_xas,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workflow_full")


def run_full_pipeline(data_dir: str):
    logger.info("Phase 1: Ingest")
    raw_data, info = load(data_dir, instrument="alba_claess")
    tree = raw_data.data
    logger.info("Loaded tree: %s", tree)

    # Find a dataset with enough records (Operando)
    # We look for a node that has 'absorption' and 'record' > 10
    target_node = None
    target_element = None

    # Simple search
    for node in tree.subtree:
        if node.dataset is not None and "absorption" in node.dataset:
            ds = node.dataset
            if "record" in ds.dims and ds.sizes["record"] > 5:
                # Infer element
                if "Mn" in node.path:
                    target_element = "Mn"
                elif "Fe" in node.path:
                    target_element = "Fe"
                elif "Zn" in node.path:
                    target_element = "Zn"

                if target_element:
                    target_node = node
                    break

    if not target_node:
        logger.error("No suitable operando dataset found.")
        return

    if target_element is None:
        logger.error("Target element could not be determined.")
        return

    ds = target_node.dataset
    logger.info(f"Selected dataset: {target_node.path} (Element: {target_element})")

    # Phase 1.5: Echem Correlation (Optional Demo)
    # Ideally, we load echem data here and plot correlation before heavy processing.
    # echem_path = Path(data_dir) / "echem.mpt"
    # if echem_path.exists():
    #     logger.info("Phase 1.5: Echem-XAS Correlation")
    #     from claess2025.io import load as load_echem
    #     echem_data, _ = load_echem(echem_path, instrument="biologic")
    #     fig_lc = plot_echem_xas(echem_data.data, ds, group_by="file_name")
    #     fig_lc.savefig("workflow_echem_correlation.png")

    # Phase 2: Preprocessing
    logger.info("Phase 2: Preprocessing")

    # 2.0 Energy Calibration (Foil)
    energy_shift = 0.0
    foil_node = None

    # Search for a foil dataset matching the target element
    for node in tree.subtree:
        if node.dataset is not None and "foil" in node.path.lower():
            if target_element in node.path:
                foil_node = node
                break

    if foil_node:
        logger.info(f"  - Found reference foil: {foil_node.path}")
        ds_foil = foil_node.dataset

        # Merge foil scans to get high quality reference
        ds_foil_merged = merge_spectra(ds_foil, method="median")

        # Calculate Shift
        try:
            energy_shift = calibrate_energy(ds_foil_merged, element=target_element)
            logger.info(f"  - Calculated Energy Shift: {energy_shift:.2f} eV")
        except Exception as e:
            logger.warning(f"  - Calibration failed: {e}")
    else:
        logger.warning("  - No foil found. Skipping calibration.")

    # 2.1 Deglitching
    logger.info("  - Deglitching...")
    ds_clean = deglitch(ds, window=3, threshold=5.0)

    # 2.2 Fluorescence Correction (Optional check)
    # logger.info("  - Fluorescence Correction (Demo)...")
    # ds_clean = correct_fluorescence(ds_clean, formula="Mn2O3", edge="K")

    # 2.3 Smoothing
    logger.info("  - Smoothing...")
    ds_smooth = smooth(ds_clean, window_length=9, polyorder=3)

    # 2.4 Alignment
    # Align to the middle record energy grid
    # mid_idx = ds.sizes["record"] // 2  # Unused
    target_energy = ds.energyc.values  # Just use original grid

    logger.info(f"  - Aligning spectra (Shift: {energy_shift:.2f} eV)...")
    ds_aligned = align_spectra(
        ds_smooth, target_energy=target_energy, shift=energy_shift
    )

    # Phase 3: XAS Processing (Normalization)
    logger.info("Phase 3: Normalization")
    config = get_element_config(target_element)

    # Only normalize, skip AutoBK/FFT for now to save time/complexity if just checking pipeline
    # actually, why not all?
    config["normalize"] = config.get("normalize", {})

    ds_norm = process_dataset(ds_aligned, config)

    if "norm_absorption" not in ds_norm:
        logger.error("Normalization failed.")
        return

    if "e0" in ds_norm:
        e0_mean = ds_norm["e0"].mean().item()
        e0_std = ds_norm["e0"].std().item()
        logger.info(f"  - Sample E0 stats: Mean={e0_mean:.2f} eV, Std={e0_std:.3f} eV")

    # Phase 4: Multivariate Analysis
    logger.info("Phase 4: PCA")

    # PCA on normalized absorption
    # We need to construct a dataset with 'absorption' = 'norm_absorption' for PCA function
    # or update PCA function to take variable name.
    # Our perform_pca takes ds and uses 'absorption'.
    # Let's swap variable
    ds_for_pca = ds_norm.copy()
    ds_for_pca["absorption"] = ds_norm["norm_absorption"]

    ds_pca = perform_pca(ds_for_pca, n_components=3)

    logger.info("PCA Variance: %s", ds_pca.pca_variance.values)

    # Phase 5: LCF
    logger.info("Phase 5: LCF")

    # Create "Synthetic" references from first and last spectrum of the series
    # In reality, you'd load reference foils or compounds.
    ref_start = ds_norm.norm_absorption.isel(record=0)
    ref_end = ds_norm.norm_absorption.isel(record=-1)

    references = {"Start_State": ref_start, "End_State": ref_end}

    # LCF on normalized data
    # perform_lcf expects 'absorption' in ds.
    ds_lcf = perform_lcf(ds_for_pca, references)

    logger.info("LCF completed.")

    # Phase 6: Plotting
    logger.info("Phase 6: Visualization")

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # 1. Raw vs Smoothed (First record)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ds.energyc, ds.absorption.isel(record=0), label="Raw", alpha=0.5)
    ax1.plot(
        ds_smooth.energyc,
        ds_smooth.absorption.isel(record=0),
        label="Smoothed",
        linestyle="--",
    )
    ax1.set_title("Preprocessing")
    ax1.legend()

    # 2. PCA Scree Plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ds_pca.component, ds_pca.pca_variance, "o-")
    ax2.set_title("PCA Explained Variance")
    ax2.set_xlabel("Component")

    # 3. PCA Components
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(3):
        if i < ds_pca.sizes["component"]:
            ax3.plot(
                ds_pca.energyc,
                ds_pca.pca_components.isel(component=i),
                label=f"PC{i + 1}",
            )
    ax3.set_title("PCA Loadings")
    ax3.legend()

    # 4. LCF Weights
    ax4 = fig.add_subplot(gs[1, :])
    for ref_name in references:
        ax4.plot(
            ds_lcf.record, ds_lcf.lcf_weights.sel(reference=ref_name), label=ref_name
        )
    ax4.set_title("LCF Evolution")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("workflow_full_result.png")
    logger.info("Saved workflow_full_result.png")


if __name__ == "__main__":
    target_dir = Path("XAS/Data/Operando1")
    if target_dir.exists():
        print("Starting pipeline...")
        run_full_pipeline(str(target_dir))
        print("Pipeline finished.")
    else:
        logger.error("Data directory not found.")
