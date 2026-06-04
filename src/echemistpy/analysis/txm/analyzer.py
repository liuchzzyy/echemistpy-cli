# -*- coding: utf-8 -*-
"""TXM 化学成像与谱图分析器。"""

from __future__ import annotations

import logging
from typing import ClassVar

import lmfit
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.ndimage
import scipy.optimize
import xarray as xr
from skimage.registration import phase_cross_correlation
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from echemistpy.analysis.registry import TechniqueAnalyzer
from echemistpy.data.models import AnalysisBundle, DataBundle

logger = logging.getLogger(__name__)

ENERGY = "energy_ev"
ENERGY_ALIASES = ("energy_ev", "energy", "energyc")


def b_value_model(x: np.ndarray, b: float, c: float) -> np.ndarray:
    """用于厚度校正的 B-value 模型。

    公式：I_thick = -ln((exp(-I_thin * C) + B) / (1 + B))
    改写为：ln(1 + B) - ln(exp(-I_thin * C) + B)，以改善数值稳定性。
    """
    # 约束 b 非负，保证拟合过程的物理意义和数值稳定性。
    b_safe = max(b, 0.0)

    # 第一项：ln(1 + B)。B 很小时 log1p 精度更好。
    term1 = np.log1p(b_safe)

    # 第二项：ln(exp(...) + B)。
    exp_val = np.exp(-x * c)

    # 增加极小值，避免 exp_val -> 0 且 b -> 0 时出现 log(0)。
    term2 = np.log(exp_val + b_safe + 1e-15)

    return term1 - term2


def build_lmfit_model(config: dict) -> tuple[lmfit.Model, lmfit.Parameters]:  # noqa: PLR0912
    """根据配置字典构建 lmfit 组合模型。"""
    components = config.get("components", [])
    if not components:
        raise ValueError("模型配置没有定义 components。")

    composite_model = None
    params = lmfit.Parameters()

    for i, comp in enumerate(components):
        ctype = comp["type"].lower()
        prefix = f"c{i}_"

        # 选择模型组件。
        if ctype == "gaussian":
            model = lmfit.models.GaussianModel(prefix=prefix)
        elif ctype == "lorentzian":
            model = lmfit.models.LorentzianModel(prefix=prefix)
        elif ctype == "linear":
            model = lmfit.models.LinearModel(prefix=prefix)
        elif ctype in {"arctan", "step"}:
            model = lmfit.models.StepModel(prefix=prefix, form="arctan")
        else:
            logger.warning("未知模型组件类型: %s，已跳过。", ctype)
            continue

        # 初始化参数。
        comp_params = comp.get("params", {})
        bounds = comp.get("bounds", {})

        # lmfit 标准参数名。
        model_params = model.make_params()

        # 从配置更新参数值。
        for pname, pval in comp_params.items():
            full_name = f"{prefix}{pname}"
            if full_name in model_params:
                model_params[full_name].set(value=pval)

        # 应用可选边界。
        lower_bounds = bounds.get("lower")
        upper_bounds = bounds.get("upper")

        # 定义支持类型的参数顺序。
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

        # 加入组合模型。
        if composite_model is None:
            composite_model = model
        else:
            composite_model += model

        params.update(model_params)

    if composite_model is None:
        raise ValueError("模型配置中没有有效组件。")

    return composite_model, params


class TXMAnalyzer(TechniqueAnalyzer):
    """透射 X 射线显微（TXM）分析器。

    流程：
    1. 预处理：图像配准和能量插值
    2. PCA 去噪
    3. pre-edge 线性背景扣除
    4. B-value 厚度校正
    5. ROI 映射和聚类
    6. 谱图拟合
    """

    technique = "txm"
    name = "TXMAnalyzer"

    # 分析配置。
    energy_step = 0.1
    align_images = True
    alignment_method = "phase_correlation"
    alignment_upsample_factor = 10
    pca_components = 5

    # UMAP 配置。
    use_umap = False
    umap_n_components = 2
    umap_n_neighbors = 15
    umap_min_dist = 0.1
    umap_metric = "euclidean"

    pre_edge_range: ClassVar[tuple[float, float]] = (625.0, 635.0)
    roi_maps: ClassVar[dict[str, list[float]]] = {}
    spatial_rois: ClassVar[dict[str, list[float]]] = {}
    roi_ranges: ClassVar[list[tuple[float, float]]] = []
    clustering_method = "kmeans"
    clustering_params: ClassVar[dict[str, object]] = {}
    n_clusters = 3

    # 谱图拟合配置。
    fitting_models: ClassVar[dict[str, dict]] = {}

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("optical_density",)

    @staticmethod
    def _as_dataset(data: xr.Dataset | xr.DataTree) -> xr.Dataset:
        """将 TXM 输入收窄为 Dataset。"""
        if isinstance(data, xr.Dataset):
            return data
        if data.dataset is not None and (data.dataset.data_vars or data.dataset.coords):
            return data.dataset
        for child in data.subtree:
            if child.dataset is not None and child.dataset.data_vars:
                return child.dataset
        raise TypeError("TXM 分析需要 Dataset 或至少一个包含数据的 DataTree 节点。")

    @classmethod
    def _normalize_energy_coord(cls, ds: xr.Dataset) -> xr.Dataset:
        """统一 TXM 能量坐标名到 energy_ev。"""
        if ENERGY in ds.coords or ENERGY in ds.dims:
            return ds
        for name in ENERGY_ALIASES:
            if name == ENERGY:
                continue
            if name in ds.coords or name in ds.dims:
                return ds.rename({name: ENERGY})
        raise ValueError(f"TXM 数据缺少能量坐标，支持名称: {ENERGY_ALIASES}。")

    @staticmethod
    def _working_signal(ds: xr.Dataset) -> xr.DataArray:
        """按处理优先级返回当前使用的 TXM 光密度数据。"""
        for name in ("thickness_corrected", "background_removed", "denoised", "optical_density"):
            if name in ds:
                return ds[name]
        raise ValueError("TXM 数据缺少 optical_density 或处理后的信号变量。")

    @staticmethod
    def _get_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
        """从 DataArray 中识别空间维度名。"""
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
            # 回退：默认最后两个非能量维度为空间维度。
            non_energy_dims = [str(d) for d in da.dims if "energy" not in str(d).lower()]
            if len(non_energy_dims) >= 2:
                y_dim, x_dim = non_energy_dims[-2:]
            else:
                raise ValueError(f"无法从 {da.dims} 识别空间维度。")

        return str(y_dim), str(x_dim)

    # 步骤 1：图像配准。
    def align_stack(self, ds: xr.Dataset | xr.DataTree) -> xr.Dataset:  # noqa: PLR0914
        """使用 Scikit-image 对图像堆栈做漂移校正。"""
        ds = self._normalize_energy_coord(self._as_dataset(ds))

        if "optical_density" not in ds:
            return ds

        # 在副本上执行配准，避免修改输入数据。
        ds = ds.copy(deep=True)
        da = ds["optical_density"]

        # 将 energy_ev 放到第一维，方便逐帧处理。
        if ENERGY in da.dims:
            da_aligned = da.transpose(ENERGY, ...)
        else:
            return ds

        # 转成 numpy 数组处理。
        data = da_aligned.values.copy()
        n_images = data.shape[0]
        if n_images < 2:
            return ds

        # 使用中间帧作为参考帧。
        ref_idx = n_images // 2
        ref_img = np.nan_to_num(data[ref_idx])

        shifts = []
        logger.info("使用 %s 将图像堆栈配准到第 %d 帧。", self.alignment_method, ref_idx)

        for i in range(n_images):
            if i == ref_idx:
                shifts.append((0.0, 0.0))
                continue

            curr_img = np.nan_to_num(data[i])
            dy, dx = 0.0, 0.0

            if self.alignment_method == "phase_correlation":
                try:
                    shift, _, _ = phase_cross_correlation(ref_img, curr_img, upsample_factor=self.alignment_upsample_factor)
                    # shift 为 [y, x]，表示把当前帧移动到参考帧。
                    dy, dx = float(shift[0]), float(shift[1])
                except Exception as e:
                    logger.debug("第 %d 帧配准失败: %s", i, e)

            elif self.alignment_method == "center_of_mass":
                try:
                    cy_ref, cx_ref = scipy.ndimage.center_of_mass(ref_img)
                    cy_curr, cx_curr = scipy.ndimage.center_of_mass(curr_img)
                    dy = cy_ref - cy_curr
                    dx = cx_ref - cx_curr
                except Exception as e:
                    logger.debug("第 %d 帧质心配准失败: %s", i, e)

            shifts.append((dy, dx))

            # 应用位移。
            data[i] = scipy.ndimage.shift(data[i], (dy, dx), order=1, mode="nearest")

        # 更新数据集。
        ds["optical_density"] = (da_aligned.dims, data)
        ds.attrs["alignment_shifts"] = shifts

        return ds

    # 步骤 2：能量轴插值。
    def interpolate_energy(self, ds: xr.Dataset | xr.DataTree) -> xr.Dataset:
        """将能量轴插值到均匀网格。"""
        ds = self._normalize_energy_coord(self._as_dataset(ds)).copy(deep=True)

        if ENERGY not in ds.coords:
            logger.warning("未找到 '%s' 坐标，跳过插值。", ENERGY)
            return ds

        energy = ds.coords[ENERGY].values
        if len(energy) < 2:
            return ds

        # 处理重复能量值。
        if not pd.Index(energy).is_unique:
            logger.warning("发现重复能量值，将对重复点取平均。")
            ds = ds.groupby(ENERGY).mean()
            energy = ds.coords[ENERGY].values

        # 构建均匀能量网格。
        e_min, e_max = float(np.nanmin(energy)), float(np.nanmax(energy))
        new_energy = np.arange(e_min, e_max + self.energy_step / 2, self.energy_step)

        # xarray.interp 会处理所有包含 energy_ev 维度的变量。
        cleaned_ds = ds.interp({ENERGY: new_energy}, method="linear", kwargs={"fill_value": "extrapolate"})

        return cleaned_ds

    def preprocess(self, bundle: DataBundle) -> DataBundle:
        """执行图像配准和能量插值预处理。"""
        ds = self._normalize_energy_coord(self._as_dataset(bundle.data).copy(deep=True))

        if self.align_images:
            try:
                ds = self.align_stack(ds)
            except Exception as e:
                logger.warning("图像配准失败: %s", e)

        try:
            ds = self.interpolate_energy(ds)
        except Exception as e:
            logger.warning("能量插值失败: %s", e)

        bundle.data = ds
        return bundle

    # 步骤 3：PCA 去噪。
    def denoise_pca(self, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
        """对数据执行 PCA 去噪。"""
        ds = self._normalize_energy_coord(ds).copy(deep=True)
        results = {}

        if "optical_density" not in ds:
            logger.warning("PCA 去噪缺少 'optical_density'。")
            return ds, results

        da = ds["optical_density"]
        if ENERGY in da.dims and da.dims[0] != ENERGY:
            da = da.transpose(ENERGY, ...)

        original_shape = da.shape
        n_energy = original_shape[0]
        flat_data = da.values.reshape(n_energy, -1)

        mask_nan = np.isnan(flat_data)
        flat_data_clean = np.nan_to_num(flat_data) if mask_nan.any() else flat_data

        try:  # noqa: PLW0717
            # PCA 输入为 (n_samples, n_features)，这里将像素作为样本、能量通道作为特征。
            x_input = flat_data_clean.T  # Shape: (n_pixels, n_energy)

            # PCA 会自动做中心化，这里不额外缩放谱图强度。

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

            # 返回当前组件的解释方差，供后续 scree plot 使用。
            results["pca_explained_variance"] = pca.explained_variance_

        except Exception as e:
            logger.error("PCA 去噪失败: %s", e)
            ds["denoised"] = da

        return ds, results

    # 步骤 4：背景扣除。
    def remove_background(self, ds: xr.Dataset) -> xr.Dataset:
        """扣除 pre-edge 线性背景。"""
        ds = self._normalize_energy_coord(ds).copy(deep=True)
        # 优先使用去噪结果，否则使用原始 optical_density。
        work_da = self._working_signal(ds)

        e_coords = work_da.coords[ENERGY].values
        mask_pre = (e_coords >= self.pre_edge_range[0]) & (e_coords <= self.pre_edge_range[1])

        if bool(mask_pre.any()):
            try:  # noqa: PLW0717
                x_pre = e_coords[mask_pre]
                y_pre = work_da.isel({ENERGY: mask_pre}).values.reshape(len(x_pre), -1)

                # 线性回归：y = mx + c。
                x_mat = np.vstack([x_pre, np.ones(len(x_pre))]).T
                # beta: [斜率, 截距]。
                beta, _, _, _ = scipy.linalg.lstsq(x_mat, y_pre)

                x_full = np.vstack([e_coords, np.ones(len(e_coords))]).T
                background_flat = x_full @ beta
                background = background_flat.reshape(work_da.shape)

                ds["background_removed"] = work_da - background
            except Exception as e:
                logger.warning("背景扣除失败: %s", e)
                ds["background_removed"] = work_da
        else:
            ds["background_removed"] = work_da

        return ds

    # 步骤 5：B-value 厚度校正。
    def correct_thickness(self, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:  # noqa: PLR0914
        """应用 B-value 方法校正厚度效应。"""
        ds = self._normalize_energy_coord(ds).copy(deep=True)
        results = {}

        work_da = self._working_signal(ds)

        # 通过积分强度图寻找薄区和厚区。
        intensity_map = work_da.sum(dim=ENERGY)
        flat_int = intensity_map.values.flatten()
        flat_int = flat_int[~np.isnan(flat_int)]

        if len(flat_int) > 0:
            p_low, p_high = np.percentile(flat_int, [10, 90])
            mask_thin = intensity_map < p_low
            mask_thick = intensity_map > p_high

            if bool(mask_thin.any()) and bool(mask_thick.any()):
                y_dim, x_dim = self._get_spatial_dims(work_da)
                spec_thin = work_da.where(mask_thin).mean(dim=[y_dim, x_dim]).values
                spec_thick = work_da.where(mask_thick).mean(dim=[y_dim, x_dim]).values

                try:  # noqa: PLW0717
                    popt, _ = scipy.optimize.curve_fit(b_value_model, spec_thin, spec_thick, p0=[0.1, 1.0], bounds=([0, 0], [np.inf, np.inf]))
                    b_val, c_val = popt
                    results["b_value"] = float(b_val)
                    results["c_value"] = float(c_val)

                    arg = (1 + b_val) * np.exp(-work_da) - b_val
                    arg = xr.where(arg > 1e-9, arg, 1e-9)
                    ds["thickness_corrected"] = -np.log(arg)
                except Exception as e:
                    logger.warning("B-value 厚度校正失败: %s", e)
                    ds["thickness_corrected"] = work_da
            else:
                ds["thickness_corrected"] = work_da
        else:
            ds["thickness_corrected"] = work_da

        return ds, results

    # 步骤 6：ROI 选择。
    def apply_rois(self, ds: xr.Dataset) -> xr.Dataset:
        """应用谱区 ROI 和空间 ROI。"""
        ds = self._normalize_energy_coord(ds).copy(deep=True)
        work_da = self._working_signal(ds)

        # 1. 谱区 ROI 生成图像映射。
        rois_to_process = {}
        # 旧接口保留为内部配置项，后续可删除。
        if self.roi_ranges:
            for start, end in self.roi_ranges:
                rois_to_process[f"roi_{start}_{end}"] = (start, end)
        # 新接口：按名称传入 ROI 范围。
        for name, rng in self.roi_maps.items():
            if len(rng) >= 2:
                rois_to_process[name] = (rng[0], rng[1])

        for name, (start, end) in rois_to_process.items():
            s, e = sorted([start, end])
            mask_roi = (work_da[ENERGY] >= s) & (work_da[ENERGY] <= e)
            if bool(mask_roi.any()):
                roi_map = work_da.sel({ENERGY: slice(s, e)}).mean(dim=ENERGY)
                ds[name] = roi_map

        # 2. 空间 ROI 生成平均谱。
        if self.spatial_rois:
            for name, coords in self.spatial_rois.items():
                if len(coords) >= 4:
                    x1, x2, y1, y2 = sorted(coords[0:2]) + sorted(coords[2:4])
                    try:
                        y_dim, x_dim = self._get_spatial_dims(work_da)
                        # 这里假设坐标为索引或像素坐标；xarray.sel 的 slice 会包含边界。
                        roi_spec = work_da.sel({x_dim: slice(x1, x2), y_dim: slice(y1, y2)}).mean(dim=[y_dim, x_dim])
                        ds[f"spectrum_{name}"] = roi_spec
                    except Exception as e:
                        logger.warning("空间 ROI %s 处理失败: %s", name, e)

        return ds

    # 步骤 7：UMAP 和 sklearn 聚类。
    def cluster_analysis(self, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:  # noqa: PLR0912, PLR0915, PLR0914
        """执行 UMAP 降维和 KMeans/DBSCAN/GMM 聚类。"""
        ds = self._normalize_energy_coord(ds).copy(deep=True)
        results = {}
        work_da = self._working_signal(ds)

        if ENERGY in work_da.dims and work_da.dims[0] != ENERGY:
            work_da = work_da.transpose(ENERGY, ...)

        original_shape = work_da.shape
        n_energy = original_shape[0]
        flat_data = work_da.values.reshape(n_energy, -1).T  # (pixels, energy)

        valid_pixels = ~np.isnan(flat_data).any(axis=1)
        data_for_clustering = flat_data[valid_pixels]

        if len(data_for_clustering) == 0:
            return ds, results

        # UMAP 降维。
        if self.use_umap:
            try:  # noqa: PLW0717
                import umap  # noqa: PLC0415

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
                logger.warning("UMAP 嵌入失败: %s", e)

        # 聚类。
        if len(data_for_clustering) > self.n_clusters:
            try:  # noqa: PLW0717
                labels_valid = None
                centroids = None
                method = self.clustering_method.lower()
                kwargs = self.clustering_params.copy()

                if method == "kmeans":
                    n_c = kwargs.pop("n_clusters", self.n_clusters)
                    kwargs.setdefault("random_state", 42)
                    model = KMeans(n_clusters=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.cluster_centers_

                elif method == "minibatch_kmeans":
                    n_c = kwargs.pop("n_clusters", self.n_clusters)
                    kwargs.setdefault("random_state", 42)
                    model = MiniBatchKMeans(n_clusters=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.cluster_centers_

                elif method == "gmm":
                    n_c = kwargs.pop("n_components", self.n_clusters)
                    kwargs.setdefault("random_state", 42)
                    model = GaussianMixture(n_components=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.means_

                elif method == "dbscan":
                    model = DBSCAN(**kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    unique_labels = set(labels_valid)
                    centroids_list = []
                    # DBSCAN 中标签 -1 表示噪声点。
                    valid_labels_sorted = sorted({lbl for lbl in unique_labels if lbl != -1})
                    for lbl in valid_labels_sorted:
                        centroids_list.append(data_for_clustering[labels_valid == lbl].mean(axis=0))
                    centroids = np.array(centroids_list) if centroids_list else None

                else:
                    logger.warning("未知聚类方法 %s，回退到 KMeans。", method)
                    model = KMeans(n_clusters=self.n_clusters, random_state=42)
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
                logger.warning("聚类失败: %s", e)

        return ds, results

    # 步骤 8：像素级拟合和谱图拟合。
    def fit_pixels(self, ds: xr.Dataset, cluster_centroids: np.ndarray | None = None) -> tuple[xr.Dataset, dict]:  # noqa: PLR0912, PLR0914
        """对 ROI 或聚类得到的谱图执行拟合。"""
        ds = self._normalize_energy_coord(ds).copy(deep=True)
        results = {}
        fit_results = {}

        if not self.fitting_models:
            return ds, results

        work_da = self._working_signal(ds)
        energy_coords = work_da.coords[ENERGY].values

        for model_name, config in self.fitting_models.items():
            targets = config.get("targets", [])
            fit_range = config.get("range", None)

            try:
                composite, params = build_lmfit_model(config)
            except ValueError as e:
                logger.warning("跳过模型 %s: %s", model_name, e)
                continue

            if composite is None:
                continue

            # 识别拟合目标。
            spectra_to_fit = {}

            if "cluster_centroids" in targets and cluster_centroids is not None:
                for i, centroid in enumerate(cluster_centroids):
                    spectra_to_fit[f"Cluster_{i}"] = (energy_coords, centroid)

            for target in targets:
                if target.startswith("spectrum_") and target in ds:
                    da_spec = ds[target]
                    spectra_to_fit[target] = (da_spec.coords[ENERGY].values, da_spec.values)

            # 执行拟合。
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

                try:  # noqa: PLW0717
                    result = composite.fit(y_fit, params, x=x_fit)

                    # 提取有序参数，便于后续结果消费。
                    ordered_values = []
                    for pname in result.params:
                        ordered_values.append(result.params[pname].value)

                    model_fits[spec_name] = {
                        "params": ordered_values,
                        "param_dict": result.best_values,
                        "chisqr": result.chisqr,
                        "fitted_curve": (x_fit, result.best_fit),
                    }

                    # 在完整范围上计算拟合曲线。
                    y_full_calc = composite.eval(result.params, x=x)
                    da_fit = xr.DataArray(y_full_calc, coords={ENERGY: x}, dims=ENERGY, name=f"fit_{model_name}_{spec_name}")
                    ds[f"fit_{model_name}_{spec_name}"] = da_fit

                except Exception as e:
                    logger.warning("模型 %s 拟合 %s 失败: %s", model_name, spec_name, e)

            fit_results[model_name] = model_fits

        results["fitting_results"] = fit_results
        return ds, results

    # 步骤 9：NNLS 化学组分映射。
    def map_chemical_components(self, ds: xr.Dataset, references: dict[str, np.ndarray]) -> tuple[xr.Dataset, dict]:  # noqa: PLR0914
        """通过非负最小二乘（NNLS）映射化学组分。

        Args:
            ds: 包含待处理数据的 Dataset，通常使用 thickness_corrected 或 denoised
            references: 组分名到参考谱数组的映射

        Returns:
            添加组分图和残差图后的 Dataset，以及结果字典。
        """
        ds = self._normalize_energy_coord(ds).copy(deep=True)
        results = {}

        if not references:
            return ds, results

        work_da = self._working_signal(ds)

        # 将参考谱对齐到数据能量轴。
        energy = work_da.coords[ENERGY].values

        # 构建设计矩阵 A，形状为 (n_energy, n_components)。
        component_names = sorted(references.keys())
        design_matrix_list = []
        valid_refs = []

        for name in component_names:
            ref_spec = references[name]
            if len(ref_spec) != len(energy):
                # 当前先做长度检查；后续可补充参考谱插值。
                logger.warning("参考谱 %s 长度 %d 与能量轴长度 %d 不一致，已跳过。", name, len(ref_spec), len(energy))
                continue
            design_matrix_list.append(ref_spec)
            valid_refs.append(name)

        if not design_matrix_list:
            return ds, results

        design_matrix = np.column_stack(design_matrix_list)  # (n_energy, n_components)

        # 准备数据矩阵 B，形状为 (n_energy, n_pixels)。
        if ENERGY in work_da.dims and work_da.dims[0] != ENERGY:
            work_da = work_da.transpose(ENERGY, ...)

        original_shape = work_da.shape
        n_energy = original_shape[0]
        flat_data = work_da.values.reshape(n_energy, -1)  # (n_energy, n_pixels)

        # 对每个像素求解 NNLS：min||Ax - b||_2 且 x >= 0。
        # scipy.optimize.nnls 处理单个向量，因此这里逐像素迭代。

        n_pixels = flat_data.shape[1]
        n_comps = len(valid_refs)
        maps = np.zeros((n_comps, n_pixels))
        residuals = np.zeros(n_pixels)

        # 排除包含 NaN 的像素。
        mask_nan = np.isnan(flat_data).any(axis=0)
        valid_indices = np.where(~mask_nan)[0]

        for idx in valid_indices:
            b = flat_data[:, idx]
            x, rnorm = scipy.optimize.nnls(design_matrix, b)
            maps[:, idx] = x
            residuals[idx] = rnorm

        # 还原为空间图像。
        y_dim, x_dim = self._get_spatial_dims(work_da)
        ny, nx = original_shape[1], original_shape[2]

        reshaped_maps = maps.reshape(n_comps, ny, nx)
        reshaped_resid = residuals.reshape(ny, nx)

        # 写回 Dataset。
        for i, name in enumerate(valid_refs):
            ds[f"map_{name}"] = ((y_dim, x_dim), reshaped_maps[i])

        ds["nnls_residual"] = ((y_dim, x_dim), reshaped_resid)
        results["mapped_components"] = valid_refs

        return ds, results

    def _compute(self, bundle: DataBundle) -> AnalysisBundle:
        """执行完整 TXM 分析流程。"""
        ds = self._normalize_energy_coord(self._as_dataset(bundle.data).copy(deep=True))
        results = {}
        params_dict = {}

        # 1. 去噪。
        ds, pca_info = self.denoise_pca(ds)
        results.update(pca_info)
        if "pca_components" in pca_info:
            params_dict["pca_components"] = pca_info["pca_components"]

        # 2. 背景扣除。
        ds = self.remove_background(ds)

        # 3. 厚度校正。
        ds, bval_info = self.correct_thickness(ds)
        results.update(bval_info)
        if "b_value" in bval_info:
            params_dict["b_value"] = bval_info["b_value"]
            params_dict["c_value"] = bval_info["c_value"]

        # 4. ROI 处理。
        ds = self.apply_rois(ds)

        # 5. 聚类。
        ds, clust_info = self.cluster_analysis(ds)
        results.update(clust_info)

        # 6. 峰拟合。
        centroids = clust_info.get("cluster_centroids")
        ds, fit_info = self.fit_pixels(ds, cluster_centroids=centroids)
        results.update(fit_info)

        # 7. 使用聚类中心作为参考谱执行 NNLS 化学映射。
        if centroids is not None:
            # centroids 形状为 (n_clusters, n_energy)。
            refs = {f"Cluster_{i}": centroid for i, centroid in enumerate(centroids)}
            ds, map_info = self.map_chemical_components(ds, refs)
            results.update(map_info)

        if results:
            params_dict["step_results"] = results

        return AnalysisBundle(data=ds, meta=bundle.meta.copy(), parameters=params_dict)


STXMAnalyzer = TXMAnalyzer
