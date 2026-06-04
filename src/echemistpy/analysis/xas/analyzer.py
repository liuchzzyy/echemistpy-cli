"""XAS 分析模块。"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

import numpy as np
import xarray as xr

from echemistpy.analysis.registry import TechniqueAnalyzer
from echemistpy.analysis.xas.processing import find_e0_by_derivative
from echemistpy.data.models import AnalysisBundle, DataBundle

Group: Any
autobk: Any
pre_edge: Any
xftf: Any

try:
    from larch import Group as _LarchGroup
    from larch.xafs import autobk as _autobk
    from larch.xafs import pre_edge as _pre_edge
    from larch.xafs import xftf as _xftf

    Group = _LarchGroup
    autobk = _autobk
    pre_edge = _pre_edge
    xftf = _xftf
    HAS_LARCH = True
except ImportError:
    HAS_LARCH = False
    Group = None
    autobk = None
    pre_edge = None
    xftf = None

logger = logging.getLogger(__name__)


class LarchXAS:
    """xraylarch 的 XAS 分析封装。

    该类统一封装归一化、背景扣除和傅里叶变换，并集中处理 larch Group 创建。
    """

    def __init__(self, energy: np.ndarray, mu: np.ndarray, label: str = "sample"):
        self.energy = energy
        self.mu = mu
        self.label = label
        self.group: Any = None

        if HAS_LARCH:
            self.group = Group(energy=energy, mu=mu, label=label)
        else:
            logger.warning("未安装 Larch，XAS 分析能力受限。")

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
        """归一化谱图（扣除 pre-edge）。"""
        if not HAS_LARCH or self.group is None:
            raise NotImplementedError("归一化需要安装 xraylarch。")

        pre_edge(
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
        """使用 AutoBK 扣除背景。"""
        if not HAS_LARCH or self.group is None:
            raise NotImplementedError("背景扣除需要安装 xraylarch。")

        autobk(self.group, rbkg=rbkg, kmin=kmin, kmax=kmax, kweight=kweight)
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
        """执行傅里叶变换。"""
        if not HAS_LARCH or self.group is None:
            raise NotImplementedError("FFT 需要安装 xraylarch。")

        xftf(self.group, kmin=kmin, kmax=kmax, kweight=kweight, window=window)
        return {
            "r": getattr(self.group, "r", None),
            "chir": getattr(self.group, "chir", None),
            "chir_mag": getattr(self.group, "chir_mag", None),
            "chir_re": getattr(self.group, "chir_re", None),
            "chir_im": getattr(self.group, "chir_im", None),
        }


class XASAnalyzer(TechniqueAnalyzer):
    """分析 X 射线吸收谱（XAS）数据。

    使用 xraylarch 执行归一化、AutoBK 背景扣除和傅里叶变换。
    """

    technique = "xas"

    # 分析参数。
    normalize_params: ClassVar[dict[str, Any]] = {}
    autobk_params: ClassVar[dict[str, Any]] = {"rbkg": 1.0, "kweight": 2}
    fft_params: ClassVar[dict[str, Any]] = {"kmin": 2, "kmax": 12, "kweight": 2}

    # 用于辅助寻找吸收边的理论 E0。
    theoretical_e0: str | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        """返回 XAS 分析所需标准列。"""
        return ("energy_ev", "absorption")

    def _process_single_spectrum(self, energy: np.ndarray, mu: np.ndarray) -> dict[str, Any]:
        """处理单条一维谱图。"""
        # 保证输入为一维，并移除 NaN。
        if energy.ndim > 1:
            energy = energy.flatten()
        if mu.ndim > 1:
            mu = mu.flatten()

        mask = ~np.isnan(energy) & ~np.isnan(mu)
        e_clean = energy[mask]
        mu_clean = mu[mask]

        if len(e_clean) < 10:
            return {}

        analyzer = LarchXAS(e_clean, mu_clean)
        results = {}

        # 1. 归一化。
        try:
            current_e0 = self.normalize_params.get("e0")

            # 若提供理论 E0 且未显式指定 E0，则自动搜索。
            if current_e0 is None and self.theoretical_e0:
                try:
                    theo_val = float(self.theoretical_e0)
                    current_e0 = find_e0_by_derivative(e_clean, mu_clean, theoretical_e0=theo_val, search_range=50.0)
                except Exception as e:
                    logger.warning("约束 E0 搜索失败: %s", e)

            res = analyzer.normalize(e0=current_e0, **self.normalize_params)

            # 将结果对齐回原始网格。
            norm_aligned = np.full_like(energy, np.nan)
            norm_aligned[mask] = res["norm"]
            results["norm_absorption"] = norm_aligned
            results["e0"] = res["e0"]
            results["edge_step"] = res["edge_step"]
        except Exception as e:
            logger.warning("归一化失败: %s", e)

        # 2. AutoBK
        try:
            res = analyzer.remove_background(**self.autobk_params)
            results["k"] = res["k"]
            results["chi_k"] = res["chi"]
        except Exception as e:
            logger.warning("AutoBK 失败: %s", e)

        # 3. FFT
        try:
            res = analyzer.fft(**self.fft_params)
            results["r"] = res["r"]
            results["chir_mag"] = res["chir_mag"]
        except Exception as e:
            logger.warning("FFT 失败: %s", e)

        return results

    def _compute(self, bundle: DataBundle) -> AnalysisBundle:
        ds = bundle.data
        if isinstance(ds, xr.DataTree):
            ds = ds.dataset if ds.dataset is not None else xr.Dataset()
            if not ds.data_vars:
                raise ValueError("DataTree 根节点没有数据变量。")

        if "energy_ev" not in ds.coords and "energy_ev" not in ds.data_vars:
            raise ValueError("Dataset 缺少 'energy_ev'。")
        if "absorption" not in ds.data_vars:
            raise ValueError("Dataset 缺少 'absorption'。")

        energy = ds.coords["energy_ev"].values if "energy_ev" in ds.coords else ds["energy_ev"].values

        # 判断是否包含多条记录。
        has_record = "record" in ds.dims

        results_ds = ds.copy()

        if not has_record:
            mu = ds.absorption.values
            res = self._process_single_spectrum(energy, mu)

            if "norm_absorption" in res:
                results_ds["norm_absorption"] = (ds.coords["energy_ev"].dims, res["norm_absorption"])
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

                # 收集标量结果。
                e0_list.append(res.get("e0", np.nan))
                edge_step_list.append(res.get("edge_step", np.nan))

                # 收集数组结果。
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

            # 堆叠结果。
            if norm_list:
                results_ds["norm_absorption"] = (("record", "energy_ev"), np.array(norm_list))

            results_ds["e0"] = (("record"), np.array(e0_list))
            results_ds["edge_step"] = (("record"), np.array(edge_step_list))

            # 将不等长数组截断到共同长度后堆叠。
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

        params = {"normalize": self.normalize_params, "autobk": self.autobk_params, "fft": self.fft_params}
        return AnalysisBundle(data=results_ds, meta=bundle.meta.copy(), parameters=params)
