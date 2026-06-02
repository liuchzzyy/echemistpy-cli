"""Electrochemical analysis helpers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from traitlets import Bool, Float, Unicode
from traitlets import List as TList

from echemistpy.io.structures import AnalysisData, AnalysisDataInfo, RawData

from .registry import TechniqueAnalyzer


class GalvanostaticAnalyzer(TechniqueAnalyzer):
    """Analyze galvanostatic (constant-current) experiments.

    Produces capacity (cumulative charge) vs time, start/end/average potentials,
    and a normalized potential for visualization.
    """

    technique = Unicode("galvanostatic", help="Technique identifier")
    supported_techniques = TList(
        Unicode(),
        default_value=["galvanostatic", "echem", "gpcl", "gcd"],
        help="List of supported technique identifiers",
    )

    # 数据列配置
    time_columns = TList(
        Unicode(),
        default_value=["time_s", "systime"],
        help="Candidate time column names in order of preference",
    )
    potential_columns = TList(
        Unicode(),
        default_value=["ewe_v", "voltage_v", "ece_v"],
        help="Candidate potential/voltage column names in order of preference",
    )
    current_columns = TList(
        Unicode(),
        default_value=["current_ma", "current_ua"],
        help="Candidate current column names in order of preference",
    )
    capacity_columns = TList(
        Unicode(),
        default_value=["capacity_mah", "capacity_uah", "specific_capacity_mah_g"],
        help="Candidate capacity column names in order of preference",
    )

    # 分析选项
    calculate_ce = Bool(
        True,
        help="Whether to calculate coulombic efficiency if cycle_number is present",
    )
    ce_order = Unicode(
        "discharge",
        help="Order of CE calculation: 'discharge' (CE=discharge/charge, first negative current as first cycle) or 'charge' (CE=charge/discharge, first positive current as first cycle)",
    )

    # 电荷容量计算参数
    # 默认值 3600 用于将秒转换为小时 (mAh = mA * s / 3600)
    time_unit_conversion = Float(
        3600.0,
        help="Conversion factor from time unit to hours (default: 3600 for seconds to hours)",
    )

    # 库伦效率计算参数
    ce_min_charge_threshold = Float(
        1e-6,
        help="Minimum charge capacity (mAh) to consider CE calculation valid (avoid division by zero)",
    )

    # 归一化范围参数
    normalization_range = TList(
        Float(),
        default_value=[0.0, 1.0],
        help="Normalization range for time/capacity sequences [min, max]",
    )

    # 归一化方式参数
    # None 或不指定: 全局归一化所有数据（默认）
    # list[int]: 仅将指定的循环圈数合并归一化（例如 [1, 2, 3]）
    normalize_per_cycle: list[int] | None = None

    def __init__(self, **kwargs):
        """Initialize GalvanostaticAnalyzer.

        Args:
            normalize_per_cycle: Normalization mode:
                - None: Global normalization across all data (default)
                - list[int]: Normalize only specified cycle numbers together (e.g., [1, 2, 3])
            **kwargs: Other parameters for TechniqueAnalyzer
        """
        super().__init__(**kwargs)

    @staticmethod
    def _pick(ds: xr.Dataset, candidates: list[str]) -> str | None:
        """从数据集中选择第一个存在的列名.

        数据已经在 io/standardizer.py 中标准化, 直接查找标准列名.

        Args:
            ds: xarray Dataset
            candidates: 候选列名列表

        Returns:
            第一个找到的列名, 如果都不存在则返回 None
        """
        available = set(ds.data_vars) | set(ds.coords)
        for col in candidates:
            if col in available:
                return col
        return None

    @staticmethod
    def _get_column_candidates(trait_list: Any) -> list[str]:
        """将 traitlets List 转换为普通 list[str] 供类型检查使用.

        Args:
            trait_list: TList[Unicode] trait

        Returns:
            list[str]
        """
        # 在运行时, TList 就是 list, 但类型检查器需要显式转换
        return list(trait_list) if trait_list else []

    @property
    def required_columns(self) -> tuple[str, ...]:
        """返回必需的数据列.

        恒流分析器需要时间、电位和电流列，但列名可能不同（如 time_s/systime）。
        返回每类列的首选名称，实际验证在 validate/preprocess 中动态进行。

        Returns:
            首选列名元组（用于文档目的，实际验证在 preprocess 中进行）
        """
        # 返回首选列名作为文档参考
        return ("time_s", "ewe_v", "current_ma")

    @staticmethod
    def _get_numeric_time(time_array: np.ndarray) -> np.ndarray:
        """将时间数组转换为数值型 (秒).

        支持两种输入:
        1. datetime64 类型: 转换为相对于第一个时间点的秒数
        2. 数值类型: 直接返回浮点数数组

        Args:
            time_array: 时间数组, 可能是 datetime64 或数值类型

        Returns:
            数值型时间数组 (单位: 秒)

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> # datetime64 输入
            >>> times = pd.date_range("2024-01-01", periods=3, freq="1s").to_numpy()
            >>> GalvanostaticAnalyzer._get_numeric_time(times)
            array([0., 1., 2.])

            >>> # 数值输入
            >>> GalvanostaticAnalyzer._get_numeric_time(np.array([1.5, 2.5, 3.5]))
            array([1.5, 2.5, 3.5])
        """
        if np.issubdtype(getattr(time_array, "dtype", object), np.datetime64):
            t_pd = pd.to_datetime(time_array)
            return (t_pd - t_pd[0]).total_seconds().to_numpy()
        else:
            return np.asarray(time_array, dtype=float)

    def validate(self, raw_data: RawData) -> None:
        """验证数据是否适合此分析器.

        基本验证由基类的 validate() 方法处理（检查 required_columns）。
        这里可以添加额外的电化学数据特定验证。
        """
        # technique 验证由基类在 analyze() 中使用 raw_info 处理
        # 这里只需要验证数据本身的完整性
        pass

    def preprocess(self, raw_data: RawData) -> RawData:
        """按时间排序并添加数值型时间坐标.

        Args:
            raw_data: 原始数据容器

        Returns:
            预处理后的数据容器, 包含数值型时间坐标

        Raises:
            ValueError: 如果缺少必需的数据列
        """
        ds = raw_data.data
        if raw_data.is_tree:
            ds = raw_data.data.dataset
            if ds is None:
                raise ValueError("DataTree has no root dataset for galvanostatic analysis.")

        # 使用 traitlets 配置的列名
        time_key = self._pick(ds, self._get_column_candidates(self.time_columns))
        pot_key = self._pick(ds, self._get_column_candidates(self.potential_columns))
        cur_key = self._pick(ds, self._get_column_candidates(self.current_columns))

        if time_key is None:
            raise ValueError(f"No time column found. Searched for: {self._get_column_candidates(self.time_columns)}")
        if pot_key is None:
            raise ValueError(f"No potential/voltage column found. Searched for: {self._get_column_candidates(self.potential_columns)}")
        if cur_key is None:
            raise ValueError(f"No current column found. Searched for: {self._get_column_candidates(self.current_columns)}")

        # 按时间排序 (如果时间列不是单调的)
        try:
            time_vals = ds[time_key].values
            if not (np.all(np.diff(time_vals) >= 0) or np.all(np.diff(time_vals) <= 0)):
                ds = ds.sortby(time_key)
        except (TypeError, ValueError, KeyError):
            # 如果排序失败 (例如非可排序类型), 继续使用原始顺序
            pass

        dim = ds[cur_key].dims[0]

        # 使用公共方法提取数值型时间数组
        t_numeric = self._get_numeric_time(ds[time_key].values)

        # 如果不存在 time_s 坐标, 则存储数值型时间
        if "time_s" not in ds.coords:
            ds = ds.assign_coords(time_s=(dim, t_numeric))

        raw_data.data = ds
        return raw_data

    @staticmethod
    def split_by_cycle(ds: xr.Dataset) -> dict[int, xr.Dataset]:
        """将数据按照循环次数分割.

        Args:
            ds: 包含 cycle_number 的数据集

        Returns:
            字典, 键为循环编号, 值为对应的数据集
        """
        if "cycle_number" not in ds.coords and "cycle_number" not in ds.data_vars:
            raise ValueError("数据集中未找到 cycle_number 列, 无法按循环分割")

        cycle_numbers = ds["cycle_number"].values
        unique_cycles = np.unique(cycle_numbers)

        cycles = {}
        # 获取主维度名称（第一个维度）
        main_dim = next(iter(ds.sizes.keys()))
        for cycle_id in unique_cycles:
            cycle_int = int(cycle_id)
            mask = cycle_numbers == cycle_id
            cycles[cycle_int] = ds.isel({main_dim: mask})

        return cycles

    def _compute_coulombic_efficiency(
        self,
        charge_capacity: float,
        discharge_capacity: float,
        cycle_num: int,
    ) -> float:
        """根据充放电容量计算库伦效率.

        根据 ce_order 参数决定计算方式：
        - ce_order='discharge': CE = 放电容量 / 充电容量 * 100%
        - ce_order='charge': CE = 充电容量 / 放电容量 * 100%

        Args:
            charge_capacity: 充电容量 (mAh)
            discharge_capacity: 放电容量 (mAh)
            cycle_num: 循环编号（用于警告信息）

        Returns:
            库伦效率百分比，如果分母低于阈值则返回 NaN
        """
        if self.ce_order == "discharge":
            numerator = discharge_capacity
            denominator = charge_capacity
            threshold_name = "充电容量"
        else:  # charge
            numerator = charge_capacity
            denominator = discharge_capacity
            threshold_name = "放电容量"

        if denominator > self.ce_min_charge_threshold:
            return (numerator / denominator) * 100
        else:
            warnings.warn(
                f"循环 {cycle_num} 的{threshold_name} ({denominator:.6f} mAh) 低于阈值 ({self.ce_min_charge_threshold} mAh), 库伦效率设为 NaN",
                stacklevel=3,
            )
            return float("nan")

    def _calculate_ce(self, ds: xr.Dataset) -> pd.DataFrame:  # noqa: PLR0912, PLR0914
        """计算每个循环的库伦效率.

        根据电流方向判断充电和放电过程，根据 ce_order 参数计算库伦效率：
        - ce_order='discharge': CE = 放电容量 / 充电容量 * 100% (首圈从负电流开始)
        - ce_order='charge': CE = 充电容量 / 放电容量 * 100% (首圈从正电流开始)

        Args:
            ds: 包含 cycle_number, current_ma 和 time_s 的数据集

        Returns:
            DataFrame, 包含每个循环的充电容量, 放电容量和库伦效率

        Raises:
            ValueError: 如果缺少必需的列或 ce_order 值无效
        """
        if self.ce_order not in {"discharge", "charge"}:
            raise ValueError(f"ce_order 必须是 'discharge' 或 'charge', 但得到: {self.ce_order}")
        if "cycle_number" not in ds.coords and "cycle_number" not in ds.data_vars:
            raise ValueError("数据集中未找到 cycle_number 列")

        # 尝试查找容量列
        cap_candidates = self._get_column_candidates(self.capacity_columns)
        cap_key = self._pick(ds, cap_candidates)

        # 按循环分割
        cycles = self.split_by_cycle(ds)

        results = []

        # 如果找到容量列，直接使用原始容量数据
        if cap_key is not None:
            # 还需要电流列来判断充放电阶段
            cur_candidates = self._get_column_candidates(self.current_columns)
            cur_key = self._pick(ds, cur_candidates)
            if cur_key is None:
                raise ValueError(f"找到容量列但未找到电流列，无法判断充放电阶段. 电流列搜索了: {cur_candidates}")

            for cycle_num, cycle_ds in cycles.items():
                capacity_vals = cycle_ds[cap_key].values
                current_vals = cycle_ds[cur_key].values

                # 根据电流正负判断充放电阶段
                # 充电阶段：电流 > 0，放电阶段：电流 < 0
                charge_mask = current_vals > 0
                discharge_mask = current_vals < 0

                # 计算充电容量：充电阶段的容量变化（末值 - 初值）
                if np.any(charge_mask):
                    charge_indices = np.where(charge_mask)[0]
                    charge_capacity_cal = float(capacity_vals[charge_indices[-1]] - capacity_vals[charge_indices[0]])
                else:
                    charge_capacity_cal = 0.0

                # 计算放电容量：放电阶段的容量变化的绝对值（|末值 - 初值|）
                if np.any(discharge_mask):
                    discharge_indices = np.where(discharge_mask)[0]
                    discharge_capacity_cal = float(np.abs(capacity_vals[discharge_indices[-1]] - capacity_vals[discharge_indices[0]]))
                else:
                    discharge_capacity_cal = 0.0

                # 使用提取的方法计算库伦效率
                coulombic_eff = self._compute_coulombic_efficiency(charge_capacity_cal, discharge_capacity_cal, cycle_num)

                results.append({
                    "cycle_number": cycle_num,
                    "charge_capacity_cal": charge_capacity_cal,
                    "discharge_capacity_cal": discharge_capacity_cal,
                    "coulombic_efficiency_%": coulombic_eff,
                })
        else:
            # 如果没有容量列，通过电流和时间积分计算
            cur_candidates = self._get_column_candidates(self.current_columns)
            cur_key = self._pick(ds, cur_candidates)
            if cur_key is None:
                raise ValueError(f"未找到容量列也未找到电流列. 容量列搜索了: {cap_candidates}, 电流列搜索了: {cur_candidates}")

            time_candidates = self._get_column_candidates(self.time_columns)
            time_key = self._pick(ds, time_candidates)
            if time_key is None:
                raise ValueError(f"未找到时间列. 搜索了: {time_candidates}")

            for cycle_num, cycle_ds in cycles.items():
                current = cycle_ds[cur_key].values

                # 使用公共方法获取数值型时间
                time_numeric = self._get_numeric_time(cycle_ds[time_key].values)
                dt = np.gradient(time_numeric)

                # 计算电荷量, 使用配置的时间单位转换
                # mAh = mA * h = mA * s / time_unit_conversion
                charge = current * dt / self.time_unit_conversion

                # 充电为正电流, 放电为负电流 (根据实际情况可能需要调整符号)
                charge_capacity_cal = float(np.sum(charge[charge > 0]))  # 充电容量
                discharge_capacity_cal = float(np.abs(np.sum(charge[charge < 0])))  # 放电容量

                # 使用提取的方法计算库伦效率
                coulombic_eff = self._compute_coulombic_efficiency(charge_capacity_cal, discharge_capacity_cal, cycle_num)

                results.append({
                    "cycle_number": cycle_num,
                    "charge_capacity_cal": charge_capacity_cal,
                    "discharge_capacity_cal": discharge_capacity_cal,
                    "coulombic_efficiency_%": coulombic_eff,
                })

        return pd.DataFrame(results)

    def _normalize_sequence(self, data: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:  # noqa: PLR6301
        """归一化序列到指定范围.

        Args:
            data: 输入数据数组
            norm_min: 归一化范围最小值
            norm_max: 归一化范围最大值

        Returns:
            归一化后的数组
        """
        if data.size > 0:
            d_min, d_max = data.min(), data.max()
            if d_max > d_min:
                return norm_min + (data - d_min) / (d_max - d_min) * (norm_max - norm_min)
            else:
                return np.full_like(data, norm_min)
        return data

    def _build_2d_result_dataset(  # noqa: PLR0913, PLR0917, PLR0914, PLR0912, PLR0915
        self,
        ds: xr.Dataset,
        cycles_dict: dict[int, xr.Dataset],
        time_key: str,
        pot_key: str | None,
        cur_key: str,
        cap_key: str | None,
        capacity: np.ndarray | None,
        normalized_capacity: np.ndarray | None,
        norm_min: float,
        norm_max: float,
    ) -> xr.Dataset:
        """构建二维结果数据集（按 cycle_number 组织）.

        Args:
            ds: 原始数据集
            cycles_dict: 按循环分割的数据字典
            time_key: 时间列名
            pot_key: 电位列名（可选）
            cur_key: 电流列名
            cap_key: 容量列名（可选）
            capacity: 容量数组（可选）
            normalized_capacity: 归一化容量数组（可选）
            norm_min: 归一化范围最小值
            norm_max: 归一化范围最大值

        Returns:
            二维结果数据集
        """
        cycle_numbers = sorted(cycles_dict.keys())
        dim = ds[cur_key].dims[0]
        max_records = max(cycles_dict[c].sizes[dim] for c in cycle_numbers)

        def create_2d_array(data_getter):
            """创建二维数组，从每个 cycle 获取数据."""
            arr = np.full((len(cycle_numbers), max_records), np.nan)
            for i, cycle_num in enumerate(cycle_numbers):
                cycle_ds = cycles_dict[cycle_num]
                data = data_getter(cycle_ds)
                arr[i, : len(data)] = data
            return arr

        # 构建二维数据变量
        # 时间归一化函数：根据 normalize_per_cycle 决定归一化方式
        if self.normalize_per_cycle is not None and isinstance(self.normalize_per_cycle, list):
            # 指定循环列表：仅将这些循环的数据合并归一化，其他循环设为 np.nan
            specified_cycles = set(self.normalize_per_cycle)
            # 获取指定循环的时间范围
            specified_time_vals = []
            for cycle_num in specified_cycles:
                if cycle_num in cycles_dict:
                    cycle_ds = cycles_dict[cycle_num]
                    specified_time_vals.extend(self._get_numeric_time(cycle_ds[time_key].values))

            if specified_time_vals:
                specified_time = np.array(specified_time_vals)
                t_min, t_max = specified_time.min(), specified_time.max()
            else:
                # 如果指定的循环不存在，使用全局范围
                global_time = self._get_numeric_time(ds[time_key].values)
                t_min, t_max = global_time.min(), global_time.max()

            def norm_time_func(c, _cycle_num):
                time_vals = self._get_numeric_time(c[time_key].values)
                # 只为指定的循环计算归一化值，其他返回 NaN
                if cycle_num in specified_cycles:
                    if t_max > t_min:
                        return norm_min + (time_vals - t_min) / (t_max - t_min) * (norm_max - norm_min)
                    else:
                        return np.full_like(time_vals, norm_min)
                else:
                    return np.full_like(time_vals, np.nan)

        else:
            # 默认：全局归一化所有数据
            global_time = self._get_numeric_time(ds[time_key].values)
            t_min, t_max = global_time.min(), global_time.max()

            def norm_time_func(c, _cycle_num):
                return (
                    norm_min + (self._get_numeric_time(c[time_key].values) - t_min) / (t_max - t_min) * (norm_max - norm_min)
                    if t_max > t_min
                    else np.full_like(self._get_numeric_time(c[time_key].values), norm_min)
                )

        def create_2d_array_with_cycle(data_getter):
            """创建二维数组，从每个 cycle 获取数据，支持 cycle_num 参数."""
            arr = np.full((len(cycle_numbers), max_records), np.nan)
            for i, cycle_num in enumerate(cycle_numbers):
                cycle_ds = cycles_dict[cycle_num]
                data = data_getter(cycle_ds, cycle_num)
                arr[i, : len(data)] = data
            return arr

        result_vars = {
            "time_s": (
                ["cycle_number", "record"],
                create_2d_array(lambda c: self._get_numeric_time(c[time_key].values)),
            ),
            "normalized_time": (
                ["cycle_number", "record"],
                create_2d_array_with_cycle(norm_time_func),
            ),
        }

        if pot_key:
            result_vars["voltage_v"] = (["cycle_number", "record"], create_2d_array(lambda c: c[pot_key].values))

        result_vars["current_ua"] = (["cycle_number", "record"], create_2d_array(lambda c: c[cur_key].values))

        if capacity is not None:  # noqa: PLR1702
            result_vars["capacity_uah"] = (
                ["cycle_number", "record"],
                create_2d_array(lambda c: c[cap_key].values if cap_key else np.array([])),
            )
            if normalized_capacity is not None:
                # 容量归一化函数：根据 normalize_per_cycle 决定归一化方式
                if self.normalize_per_cycle is not None and isinstance(self.normalize_per_cycle, list):
                    # 指定循环列表：仅将这些循环的数据合并归一化，其他循环设为 np.nan
                    specified_cycles = set(self.normalize_per_cycle)
                    # 获取指定循环的容量范围
                    specified_cap_vals = []
                    for cycle_num in specified_cycles:
                        if cycle_num in cycles_dict and cap_key:
                            cycle_ds = cycles_dict[cycle_num]
                            if cap_key in cycle_ds:
                                specified_cap_vals.extend(cycle_ds[cap_key].values)

                    if specified_cap_vals:
                        specified_cap = np.array(specified_cap_vals)
                        cap_min, cap_max = specified_cap.min(), specified_cap.max()
                    else:
                        # 如果指定的循环不存在，使用全局范围
                        cap_min, cap_max = capacity.min(), capacity.max()

                    def norm_cap_2d(c, _cycle_num):
                        if cap_key and cap_key in c:
                            cap_vals = c[cap_key].values
                            if len(cap_vals) > 0:
                                # 只为指定的循环计算归一化值，其他返回 NaN
                                if cycle_num in specified_cycles:
                                    if cap_max > cap_min:
                                        return norm_min + (cap_vals - cap_min) / (cap_max - cap_min) * (norm_max - norm_min)
                                    else:
                                        return np.full_like(cap_vals, norm_min)
                                else:
                                    return np.full_like(cap_vals, np.nan)
                        return np.array([])

                else:
                    # 默认：全局归一化所有容量数据
                    cap_min, cap_max = capacity.min(), capacity.max()

                    def norm_cap_2d(c, _cycle_num):
                        if cap_key and cap_key in c:
                            cap_vals = c[cap_key].values
                            if len(cap_vals) > 0:
                                if cap_max > cap_min:
                                    return norm_min + (cap_vals - cap_min) / (cap_max - cap_min) * (norm_max - norm_min)
                                else:
                                    return np.full_like(cap_vals, norm_min)
                        return np.array([])

                result_vars["normalized_capacity"] = (["cycle_number", "record"], create_2d_array_with_cycle(norm_cap_2d))

        # 创建坐标
        coords = {"cycle_number": cycle_numbers, "record": np.arange(max_records)}

        return xr.Dataset(result_vars, coords=coords)

    def _build_1d_result_dataset(  # noqa: PLR6301
        self,
        ds: xr.Dataset,
        normalized_time: np.ndarray,
        normalized_capacity: np.ndarray | None,
        dim: str,
    ) -> xr.Dataset:
        """构建一维结果数据集.

        Args:
            ds: 原始数据集
            normalized_time: 归一化时间数组
            normalized_capacity: 归一化容量数组（可选）
            dim: 主维度名称

        Returns:
            一维结果数据集
        """
        result_ds = ds.copy()
        result_ds["normalized_time"] = (dim, normalized_time)

        if normalized_capacity is not None:
            result_ds["normalized_capacity"] = (dim, normalized_capacity)

        return result_ds

    def _compute(self, raw_data: RawData) -> tuple[AnalysisData, AnalysisDataInfo]:  # noqa: PLR0914
        """计算累计电荷(容量)、电位统计和库伦效率 (内部方法).

        此函数：
        1. 提取和验证数据：查找时间、电位、电流、容量列
        2. 归一化处理：对时间/容量序列进行归一化，范围可配置
        3. 库伦效率计算：如果启用且存在 cycle_number

        Note:
            这是内部方法，用户应调用 analyze() 而不是直接调用此方法。

        Args:
            raw_data: 原始数据容器

        Returns:
            (AnalysisData, AnalysisDataInfo) 元组:
            - AnalysisData: 包含分析结果的数据集
            - AnalysisDataInfo: 分析过程的参数和元数据
        """
        ds = raw_data.data
        if isinstance(ds, xr.DataTree):
            ds = ds.dataset
            if ds is None:
                raise ValueError("DataTree has no root dataset for galvanostatic analysis.")

        # 1. 提取和验证数据 - 查找时间、电位、电流、容量列
        time_key = self._pick(ds, self._get_column_candidates(self.time_columns)) or "time_s"
        pot_key = self._pick(ds, self._get_column_candidates(self.potential_columns))
        cur_key = self._pick(ds, self._get_column_candidates(self.current_columns)) or next(iter(ds.data_vars))
        cap_key = self._pick(ds, self._get_column_candidates(self.capacity_columns))

        # 记录使用的列名
        used_columns = {
            "time_column": time_key,
            "potential_column": pot_key,
            "current_column": cur_key,
            "capacity_column": cap_key,
        }

        # 时间数组
        time = ds.coords[time_key].values if time_key in ds.coords else ds[time_key].values
        current = ds[cur_key].values
        capacity = ds[cap_key].values if cap_key else None

        dim = ds[cur_key].dims[0]

        # 使用公共方法计算时间步长
        t_numeric = self._get_numeric_time(time)
        dt = np.gradient(t_numeric)

        # 累计电荷 (电流对时间的积分)
        np.cumsum(current * dt)

        # 2. 归一化处理
        norm_min, norm_max = self.normalization_range[0], self.normalization_range[1]
        normalized_time = self._normalize_sequence(t_numeric, norm_min, norm_max)
        normalized_capacity = self._normalize_sequence(capacity, norm_min, norm_max) if capacity is not None else None

        # 3. 库伦效率计算（如果启用且存在 cycle_number）
        ce_results = None
        has_cycle = "cycle_number" in ds.coords or "cycle_number" in ds.data_vars

        if self.calculate_ce and has_cycle:
            try:
                ce_df = self._calculate_ce(ds)
                ce_results = ce_df.to_dict("records")
            except (ValueError, KeyError) as e:
                # CE 计算失败不应该影响其他分析
                warnings.warn(
                    f"库伦效率计算失败: {e}",
                    stacklevel=2,
                )
                ce_results = None

        # 4. 构建结果数据集
        if has_cycle:
            # 按 cycle 重构为二维数据 (cycle_number, record)
            cycles_dict = self.split_by_cycle(ds)
            result_ds = self._build_2d_result_dataset(
                ds=ds,
                cycles_dict=cycles_dict,
                time_key=time_key,
                pot_key=pot_key,
                cur_key=cur_key,
                cap_key=cap_key,
                capacity=capacity,
                normalized_capacity=normalized_capacity,
                norm_min=norm_min,
                norm_max=norm_max,
            )

            # 添加库伦效率数据（如果有）
            if ce_results is not None and len(ce_results) > 0:
                cycle_numbers = sorted(cycles_dict.keys())
                ce_df = pd.DataFrame(ce_results)
                ce_df = ce_df.set_index("cycle_number").reindex(cycle_numbers)

                result_ds["coulombic_efficiency_%"] = (["cycle_number"], ce_df["coulombic_efficiency_%"].values)
                result_ds["charge_capacity_cal"] = (["cycle_number"], ce_df["charge_capacity_cal"].values)
                result_ds["discharge_capacity_cal"] = (["cycle_number"], ce_df["discharge_capacity_cal"].values)
        else:
            # 没有 cycle_number，保持一维结构
            result_ds = self._build_1d_result_dataset(ds, normalized_time, normalized_capacity, dim)

        # 构建 parameters 字典（包含标量结果和参数）
        parameters = {
            "normalization_range": list(self.normalization_range),
            "normalize_per_cycle": self.normalize_per_cycle,
            "ce_order": self.ce_order,
            "used_columns": used_columns,
        }

        # 构建 AnalysisData 和 AnalysisDataInfo
        analysis_data = AnalysisData(data=result_ds)

        # 创建 AnalysisDataInfo，只包含 parameters
        # 以下字段会由基类的 analyze() 方法从 RawDataInfo 自动继承:
        # - technique: 技术标识（从原始数据继承，如 ["echem", "gpcl"]）
        # - sample_name: 样品名称
        # - start_time: 测量起始时间
        # - operator: 操作人员
        # - instrument: 仪器标识
        # - active_material_mass: 活性物质质量
        # - wave_number: 波数（光谱相关）
        analysis_info = AnalysisDataInfo(parameters=parameters)

        return analysis_data, analysis_info
