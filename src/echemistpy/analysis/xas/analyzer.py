"""XAS Analysis Module."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import xarray as xr
from traitlets import Dict, Unicode

from echemistpy.analysis.registry import TechniqueAnalyzer
from echemistpy.analysis.xas.processing import find_e0_by_derivative
from echemistpy.io.structures import AnalysisData, AnalysisDataInfo, RawData

try:
    from larch import Group  # type: ignore
    from larch.xafs import autobk, pre_edge, xftf  # type: ignore

    HAS_LARCH = True
except ImportError:
    HAS_LARCH = False
    Group = None
    autobk = None
    pre_edge = None
    xftf = None

logger = logging.getLogger(__name__)


class LarchXAS:
    """Wrapper for XAS analysis functions using xraylarch.

    This class provides a consistent API for XAS analysis (normalization,
    background removal, FFT) handling larch Group creation and error handling.
    """

    def __init__(self, energy: np.ndarray, mu: np.ndarray, label: str = "sample"):
        self.energy = energy
        self.mu = mu
        self.label = label
        self.group: Any = None

        if HAS_LARCH:
            self.group = Group(energy=energy, mu=mu, label=label)  # type: ignore
        else:
            logger.warning("Larch not available. Analysis capabilities limited.")

    def normalize(
        self,
        e0: Optional[float] = None,
        step: Optional[float] = None,
        nvict: float = 0,
        pre1: Optional[float] = None,
        pre2: Optional[float] = None,
        norm1: Optional[float] = None,
        norm2: Optional[float] = None,
    ) -> dict[str, Any]:
        """Normalize spectrum (pre-edge subtraction)."""
        if not HAS_LARCH or self.group is None:
            raise NotImplementedError("Normalization requires xraylarch.")

        pre_edge(  # type: ignore
            self.group,
            e0=e0,
            step=step,
            nvict=nvict,
            pre1=pre1,
            pre2=pre2,
            norm1=norm1,
            norm2=norm2,
        )
        return {
            "e0": getattr(self.group, "e0", None),
            "edge_step": getattr(self.group, "edge_step", None),
            "norm": getattr(self.group, "norm", None),
            "flat": getattr(self.group, "flat", None),
            "pre_edge": getattr(self.group, "pre_edge", None),
            "post_edge": getattr(self.group, "post_edge", None),
        }

    def remove_background(self, rbkg: float = 1.0, kmin: float = 0, kmax: float = 20, kweight: float = 2) -> dict[str, Any]:
        """Remove background (AutoBK)."""
        if not HAS_LARCH or self.group is None:
            raise NotImplementedError("Background removal requires xraylarch.")

        autobk(self.group, rbkg=rbkg, kmin=kmin, kmax=kmax, kweight=kweight)  # type: ignore
        return {
            "k": getattr(self.group, "k", None),
            "chi": getattr(self.group, "chi", None),
            "bkg": getattr(self.group, "bkg", None),
        }

    def fft(
        self,
        kmin: float = 2,
        kmax: float = 13,
        kweight: float = 2,
        window: str = "hanning",
    ) -> dict[str, Any]:
        """Perform FFT."""
        if not HAS_LARCH or self.group is None:
            raise NotImplementedError("FFT requires xraylarch.")

        xftf(self.group, kmin=kmin, kmax=kmax, kweight=kweight, window=window)  # type: ignore
        return {
            "r": getattr(self.group, "r", None),
            "chir": getattr(self.group, "chir", None),
            "chir_mag": getattr(self.group, "chir_mag", None),
            "chir_re": getattr(self.group, "chir_re", None),
            "chir_im": getattr(self.group, "chir_im", None),
        }


class XASAnalyzer(TechniqueAnalyzer):
    """Analyze X-ray Absorption Spectroscopy (XAS) data.

    Performs normalization, background removal (AutoBK), and FFT using xraylarch.
    """

    technique = Unicode("xas", help="Technique identifier")

    # Configuration parameters
    normalize_params = Dict(default_value={}, help="Parameters for normalization (e0, pre1, etc.)")
    autobk_params = Dict(default_value={"rbkg": 1.0, "kweight": 2}, help="Parameters for AutoBK")
    fft_params = Dict(default_value={"kmin": 2, "kmax": 12, "kweight": 2}, help="Parameters for FFT")

    # Theoretical E0 for finding edge
    theoretical_e0 = Unicode(None, allow_none=True, help="Theoretical edge energy (optional)")

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("energyc", "absorption")

    def _process_single_spectrum(self, energy: np.ndarray, mu: np.ndarray) -> dict[str, Any]:
        """Helper to process a single 1D spectrum."""
        # Ensure 1D and handle NaNs
        if energy.ndim > 1:
            energy = energy.flatten()
        if mu.ndim > 1:
            mu = mu.flatten()

        mask = ~np.isnan(energy) & ~np.isnan(mu)
        e_clean = energy[mask]
        mu_clean = mu[mask]

        if len(e_clean) < 10:  # Too few points
            return {}

        analyzer = LarchXAS(e_clean, mu_clean)
        results = {}

        # 1. Normalize
        try:
            current_e0 = self.normalize_params.get("e0")

            # Auto-find E0 if theoretical provided and explicit E0 missing
            if current_e0 is None and self.theoretical_e0:
                try:
                    theo_val = float(self.theoretical_e0)
                    current_e0 = find_e0_by_derivative(e_clean, mu_clean, theoretical_e0=theo_val, search_range=50.0)
                except Exception as e:
                    logger.warning("Constrained E0 search failed: %s", e)

            res = analyzer.normalize(e0=current_e0, **self.normalize_params)

            # Align result to original grid
            norm_aligned = np.full_like(energy, np.nan)
            norm_aligned[mask] = res["norm"]
            results["norm_absorption"] = norm_aligned
            results["e0"] = res["e0"]
            results["edge_step"] = res["edge_step"]
        except Exception as e:
            logger.warning("Normalization failed: %s", e)

        # 2. AutoBK
        try:
            res = analyzer.remove_background(**self.autobk_params)
            results["k"] = res["k"]
            results["chi_k"] = res["chi"]
        except Exception as e:
            logger.warning("AutoBK failed: %s", e)

        # 3. FFT
        try:
            res = analyzer.fft(**self.fft_params)
            results["r"] = res["r"]
            results["chir_mag"] = res["chir_mag"]
        except Exception as e:
            logger.warning("FFT failed: %s", e)

        return results

    def _compute(self, raw_data: RawData) -> tuple[AnalysisData, AnalysisDataInfo]:
        ds = raw_data.data
        if isinstance(ds, xr.DataTree):
            ds = ds.dataset if ds.dataset is not None else xr.Dataset()
            if not ds.data_vars:
                raise ValueError("DataTree root has no data variables.")

        if "energyc" not in ds.coords and "energyc" not in ds.data_vars:
            raise ValueError("Dataset missing 'energyc'.")
        if "absorption" not in ds.data_vars:
            raise ValueError("Dataset missing 'absorption'.")

        energy = ds.coords["energyc"].values if "energyc" in ds.coords else ds["energyc"].values

        # Determine if multiple records
        has_record = "record" in ds.dims

        results_ds = ds.copy()

        if not has_record:
            mu = ds.absorption.values
            res = self._process_single_spectrum(energy, mu)

            if "norm_absorption" in res:
                results_ds["norm_absorption"] = (ds.coords["energyc"].dims, res["norm_absorption"])
                results_ds["e0"] = res["e0"]
                results_ds["edge_step"] = res["edge_step"]
            if "chi_k" in res:
                results_ds["chi_k"] = xr.DataArray(res["chi_k"], dims="k", coords={"k": res["k"]})
            if "chir_mag" in res:
                results_ds["chir_mag"] = xr.DataArray(res["chir_mag"], dims="r", coords={"r": res["r"]})
        else:
            records = ds.record.values
            norm_list = []
            chi_list = []
            chir_list = []
            e0_list = []
            edge_step_list = []
            k_grid = None
            r_grid = None

            for i in range(len(records)):
                mu_i = ds.absorption.isel(record=i).values
                res = self._process_single_spectrum(energy, mu_i)

                # Collect Scalars
                e0_list.append(res.get("e0", np.nan))
                edge_step_list.append(res.get("edge_step", np.nan))

                # Collect Arrays
                if "norm_absorption" in res:
                    norm_list.append(res["norm_absorption"])
                else:
                    norm_list.append(np.full_like(energy, np.nan))

                if "chi_k" in res:
                    if k_grid is None:
                        k_grid = res["k"]
                    chi_list.append(res["chi_k"])
                else:
                    chi_list.append(None)

                if "chir_mag" in res:
                    if r_grid is None:
                        r_grid = res["r"]
                    chir_list.append(res["chir_mag"])
                else:
                    chir_list.append(None)

            # Stack results
            if norm_list:
                results_ds["norm_absorption"] = (("record", "energyc"), np.array(norm_list))

            results_ds["e0"] = (("record"), np.array(e0_list))
            results_ds["edge_step"] = (("record"), np.array(edge_step_list))

            # Helper to stack jagged arrays
            def stack_jagged(data_list: list[Any], grid: Any, dim_name: str) -> Optional[xr.DataArray]:
                if grid is None or not data_list:
                    return None
                valid = [x for x in data_list if x is not None]
                if not valid:
                    return None

                min_len = min(len(x) for x in valid)
                trunc_grid = grid[:min_len]

                stacked = np.array([x[:min_len] if x is not None else np.full(min_len, np.nan) for x in data_list])
                return xr.DataArray(stacked, dims=("record", dim_name), coords={"record": records, dim_name: trunc_grid})

            chi_da = stack_jagged(chi_list, k_grid, "k")
            if chi_da is not None:
                results_ds["chi_k"] = chi_da

            chir_da = stack_jagged(chir_list, r_grid, "r")
            if chir_da is not None:
                results_ds["chir_mag"] = chir_da

        # Build Info
        params = {"normalize": self.normalize_params, "autobk": self.autobk_params, "fft": self.fft_params}
        analysis_info = AnalysisDataInfo(parameters=params)

        return AnalysisData(data=results_ds), analysis_info
