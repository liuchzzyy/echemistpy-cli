"""数据标准化工具模块。

本模块处理原始测量数据的标准化转换，包括：
- 列名标准化：将不同仪器的列名映射到统一的标准名称
- 单位标准化：转换数据单位到标准单位（如 mA、V、s 等）
- 列顺序标准化：按技术规范排列列的显示顺序

支持的标准化技术：
- 电化学 (echem)：CV、GCD、EIS、CA、CP 等
- XRD：X 射线衍射
- XPS：X 射线光电子能谱
- TGA：热重分析
- XAS：X 射线吸收谱
- TXM：透射 X 射线显微镜
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Optional

import numpy as np
import xarray as xr

from echemistpy.data.column_mappings import (
    ECHEM_PREFERRED_ORDER,
    get_echem_mappings,
    get_tga_mappings,
    get_txm_mappings,
    get_xas_mappings,
    get_xps_mappings,
    get_xrd_mappings,
)
from echemistpy.data.models import DataBundle
from echemistpy.data.utils import sanitize_variable_names


class DataStandardizer:
    """数据标准化器，将原始数据转换为 echemistpy 标准格式。

    功能：
    - 列名标准化：映射不同仪器的列名到统一标准
    - 单位标准化：转换到标准单位（mA、V、s 等）
    - 列顺序标准化：按技术规范排列显示顺序

    Attributes:
        dataset: 待标准化的 xarray.Dataset
        techniques: 技术类型列表（如 ['echem', 'peis']）
        instrument: 仪器标识符（可选）
    """

    # 电化学技术类别（用于映射分组）
    ECHEM_TECHNIQUES: ClassVar[set[str]] = {
        "cv",
        "gcd",
        "eis",
        "ca",
        "cp",
        "lsjv",
        "echem",
        "ec",
        "peis",
        "gpcl",
        "ocv",
    }

    def __init__(
        self,
        dataset: xr.Dataset,
        techniques: list[str] | str = "unknown",
        instrument: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化数据标准化器。

        Args:
            dataset: 待标准化的数据集
            techniques: 技术类型（字符串或列表）
            instrument: 可选的仪器标识符
            **kwargs: 保留参数；当前不支持额外配置
        """
        if kwargs:
            extra = ", ".join(sorted(kwargs))
            raise TypeError(f"DataStandardizer 不支持参数: {extra}")
        if isinstance(techniques, str):
            techniques = [techniques]
        self.dataset = dataset.copy(deep=True)
        self.techniques = [t.lower() for t in techniques]
        self.instrument = instrument

    @classmethod
    def _get_mappings_for_technique(cls, tech: str) -> dict[str, str]:
        """获取指定技术的列名映射。

        Args:
            tech: 技术类型标识符

        Returns:
            列名映射字典
        """
        tech_category = tech.lower()
        if tech_category in cls.ECHEM_TECHNIQUES:
            return get_echem_mappings()

        # 根据技术类别获取映射
        mapping_getters = {
            "echem": get_echem_mappings,
            "xrd": get_xrd_mappings,
            "xps": get_xps_mappings,
            "tga": get_tga_mappings,
            "xas": get_xas_mappings,
            "txm": get_txm_mappings,
        }

        getter = mapping_getters.get(tech_category)
        return getter() if getter else {}

    def _get_preferred_order_for_technique(self, tech: str) -> list[str]:
        """获取指定技术的首选列顺序。

        Args:
            tech: 技术类型标识符

        Returns:
            列名顺序列表
        """
        if tech.lower() in self.ECHEM_TECHNIQUES:
            return ECHEM_PREFERRED_ORDER
        return []

    def standardize(self, custom_mapping: Optional[dict[str, str]] = None) -> "DataStandardizer":
        """执行完整标准化（列名和单位）。

        Args:
            custom_mapping: 可选的自定义列名映射

        Returns:
            self，支持链式调用
        """
        return self.standardize_column_names(custom_mapping).standardize_units()

    def standardize_column_names(self, custom_mapping: Optional[dict[str, str]] = None) -> "DataStandardizer":
        """根据技术和仪器标准化列名。

        Args:
            custom_mapping: 可选的自定义列名映射（覆盖默认映射）

        Returns:
            self，支持链式调用
        """
        # 构建所有技术的聚合映射
        mapping = {}
        for tech in self.techniques:
            # 使用新的映射获取方法
            tech_mapping = self._get_mappings_for_technique(tech)
            mapping.update(tech_mapping)

            # 仪器特定映射（覆盖通用映射）
            if self.instrument:
                inst_key = f"{self.instrument.lower()}_{tech}"
                inst_mapping = self._get_mappings_for_technique(inst_key)
                mapping.update(inst_mapping)

        # 添加自定义映射（优先级最高）
        if custom_mapping:
            mapping.update(custom_mapping)

        # 应用重命名。若目标名已存在，只有源/目标数据等价时才去重；
        # 否则保留源数据并分配唯一后缀名，避免静默丢列。
        desired_renames = {}
        all_names = [str(name) for name in list(self.dataset.data_vars) + list(self.dataset.coords)]
        for name in all_names:
            if name in mapping and mapping[name] != name:
                desired_renames[name] = mapping[name]

        self._apply_renames(desired_renames)

        # 按首选顺序重排变量
        self._reorder_variables()

        return self

    def _reorder_variables(self) -> None:
        """按技术的首选顺序重排数据变量。"""
        for tech in self.techniques:
            preferred = self._get_preferred_order_for_technique(tech)
            if not preferred:
                continue

            # 获取按首选顺序排列的现有变量
            existing_vars = [v for v in preferred if v in self.dataset.data_vars]
            # 添加不在首选列表中的其他变量
            other_vars = [v for v in self.dataset.data_vars if v not in preferred]
            # 重排
            self.dataset = self.dataset[existing_vars + other_vars]
            break  # 只应用第一个技术的顺序

    def _apply_renames(self, desired_renames: dict[str, str]) -> None:
        """安全重命名，保留不等价的冲突列。"""
        if not desired_renames:
            return

        rename_dict: dict[str, str] = {}
        drop_names: list[str] = []
        planned_names = self._all_names()

        for old_name, requested_name in desired_renames.items():
            if old_name not in planned_names or requested_name == old_name:
                continue

            planned_names.discard(old_name)
            if requested_name in planned_names:
                if self._has_name(requested_name) and self._variables_equal(old_name, requested_name):
                    drop_names.append(old_name)
                    continue
                new_name = self._unique_name(requested_name, planned_names)
            else:
                new_name = requested_name

            rename_dict[old_name] = new_name
            planned_names.add(new_name)

        if drop_names:
            self.dataset = self.dataset.drop_vars(drop_names)
        if rename_dict:
            self.dataset = self.dataset.rename(rename_dict)

    def _all_names(self) -> set[str]:
        """返回所有变量名和坐标名。"""
        return {str(name) for name in list(self.dataset.data_vars) + list(self.dataset.coords)}

    def _has_name(self, name: str) -> bool:
        """判断变量或坐标是否存在。"""
        return name in self.dataset.data_vars or name in self.dataset.coords

    def _variables_equal(self, left: str, right: str) -> bool:
        """判断两个变量或坐标是否承载等价数据。"""
        if not (self._has_name(left) and self._has_name(right)):
            return False

        left_var = self.dataset[left]
        right_var = self.dataset[right]
        if left_var.dims != right_var.dims or left_var.shape != right_var.shape:
            return False

        if left_var.identical(right_var):
            return True

        try:
            return bool(np.array_equal(left_var.values, right_var.values, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left_var.values, right_var.values))
        except Exception:
            return False

    @staticmethod
    def _unique_name(base_name: str, used_names: set[str]) -> str:
        """追加数字后缀生成唯一名称。"""
        suffix = 1
        candidate = f"{base_name}_{suffix}"
        while candidate in used_names:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        return candidate

    def standardize_units(self) -> "DataStandardizer":
        """将单位转换为 echemistpy 标准约定。"""
        renames = {}
        conversions = {}

        # 辅助函数：替换多种单位后缀
        def replace_suffixes(name: str, replacements: dict[str, str]) -> str:
            new_name = name
            for old, new in replacements.items():
                new_name = new_name.replace(old, new)
            return new_name

        for name in list(self.dataset.data_vars.keys()) + list(self.dataset.coords.keys()):
            var_name = str(name)
            var_lower = var_name.lower()

            # 时间统一为秒，标准 time_s 保持数值秒。
            if var_name != "time_s" and ("time" in var_lower or var_name == "t"):
                if "min" in var_lower and "mah" not in var_lower:
                    conversions[var_name] = lambda x: x * 60
                    renames[var_name] = var_name.replace("min", "s")
                elif "h" in var_lower and "mah" not in var_lower:
                    conversions[var_name] = lambda x: x * 3600
                    renames[var_name] = var_name.replace("h", "s")

            # 电流单位统一。
            elif "current" in var_lower or var_name.startswith("I"):
                if "/a" in var_lower or "_a" in var_lower:
                    conversions[var_name] = lambda x: x * 1000
                    renames[var_name] = replace_suffixes(var_name, {"/A": "/ma", "_A": "_ma", "/a": "/ma", "_a": "_ma"})
                elif "/μa" in var_lower or "/ua" in var_lower:
                    conversions[var_name] = lambda x: x / 1000
                    renames[var_name] = replace_suffixes(var_name, {"/μA": "/ma", "/uA": "/ma", "/μa": "/ma", "/ua": "/ma"})

            # 电压单位统一。
            elif ("voltage" in var_lower or "potential" in var_lower or var_name.startswith("E")) and "/mv" in var_lower:
                conversions[var_name] = lambda x: x / 1000
                renames[var_name] = replace_suffixes(var_name, {"/mV": "/V", "/mv": "/V"})

            # 容量单位统一。
            elif ("capacity" in var_lower or var_name.startswith("Q")) and "/uah" in var_lower:
                conversions[var_name] = lambda x: x / 1000
                renames[var_name] = replace_suffixes(var_name, {"/uAh": "/mAh", "/uah": "/mAh"})

        # 批量执行转换（按依赖顺序，先转换再重命名）
        for var_name, converter in conversions.items():
            self.dataset[var_name] = converter(self.dataset[var_name])

        self._apply_renames(renames)

        return self

    def ensure_required_columns(self, required_columns: list[str]) -> "DataStandardizer":
        """确保必需列存在，缺失时创建 NaN 占位列。"""
        missing_cols = []
        for col in required_columns:
            if col not in self.dataset.data_vars:
                missing_cols.append(col)

        if missing_cols:
            warnings.warn(
                f"缺少必需列: {missing_cols}。正在创建占位列。",
                stacklevel=2,
            )
            # 使用 NaN 创建占位列。
            if "record" in self.dataset.coords:
                n_rows = len(self.dataset.coords["record"])
                for col in missing_cols:
                    self.dataset[col] = ("record", np.full(n_rows, np.nan))
            elif "row" in self.dataset.coords:
                n_rows = len(self.dataset.coords["row"])
                for col in missing_cols:
                    self.dataset[col] = ("row", np.full(n_rows, np.nan))

        return self

    def get_dataset(self) -> xr.Dataset:
        """返回标准化后的数据集。"""
        return self.dataset


def standardize_bundle(
    bundle: DataBundle,
    technique_hint: Optional[str | list[str]] = None,
    custom_mapping: Optional[dict[str, str]] = None,
    required_columns: Optional[list[str]] = None,
) -> DataBundle:
    """标准化数据包并保留元数据。"""
    techniques = _normalize_techniques(technique_hint or bundle.meta.get("technique") or ["unknown"])

    if isinstance(bundle.data, xr.Dataset):
        standardized = _standardize_dataset(
            bundle.data,
            techniques=techniques,
            instrument=bundle.meta.instrument,
            custom_mapping=custom_mapping,
            required_columns=required_columns,
        )
    elif isinstance(bundle.data, xr.DataTree):
        standardized = bundle.data.map_over_datasets(
            lambda ds: _standardize_tree_node(
                ds,
                global_techniques=techniques,
                instrument=bundle.meta.instrument,
                custom_mapping=custom_mapping,
                required_columns=required_columns,
            )
        )
    else:
        raise ValueError("DataBundle 必须包含 xarray.Dataset 或 DataTree 才能标准化。")

    meta = bundle.meta.copy()
    meta.technique = techniques
    provenance = dict(bundle.provenance)
    provenance["standardized"] = True

    return DataBundle(
        data=standardized,
        meta=meta,
        schema=bundle.schema,
        provenance=provenance,
        warnings=list(bundle.warnings),
    )


def _normalize_techniques(techniques: str | list[str]) -> list[str]:
    """将技术类型提示标准化为列表。"""
    if isinstance(techniques, str):
        return [techniques]
    return list(techniques)


def _standardize_dataset(
    ds: xr.Dataset,
    *,
    techniques: list[str],
    instrument: str | None,
    custom_mapping: Optional[dict[str, str]],
    required_columns: Optional[list[str]],
) -> xr.Dataset:
    """标准化单个 Dataset。"""
    standardizer = DataStandardizer(dataset=ds, techniques=techniques, instrument=instrument)
    standardizer.standardize(custom_mapping)
    if required_columns:
        standardizer.ensure_required_columns(required_columns)
    return standardizer.get_dataset()


def _standardize_tree_node(
    ds: xr.Dataset,
    *,
    global_techniques: list[str],
    instrument: str | None,
    custom_mapping: Optional[dict[str, str]],
    required_columns: Optional[list[str]],
) -> xr.Dataset:
    """标准化单个 DataTree 节点 Dataset。"""
    if not ds.data_vars:
        return ds

    node_techniques = _normalize_techniques(ds.attrs.get("technique", global_techniques))
    standardized_ds = _standardize_dataset(
        ds,
        techniques=node_techniques,
        instrument=instrument,
        custom_mapping=custom_mapping,
        required_columns=required_columns,
    )

    result = sanitize_variable_names(standardized_ds)
    return result if isinstance(result, xr.Dataset) else standardized_ds


__all__ = [
    "DataStandardizer",
    "standardize_bundle",
]
