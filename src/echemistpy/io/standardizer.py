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
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from traitlets import HasTraits, Instance, List, Unicode

from echemistpy.io.column_mappings import (
    ECHEM_PREFERRED_ORDER,
    get_echem_mappings,
    get_tga_mappings,
    get_txm_mappings,
    get_xas_mappings,
    get_xps_mappings,
    get_xrd_mappings,
)
from echemistpy.io.reader_utils import sanitize_variable_names
from echemistpy.io.structures import (
    RawData,
    RawDataInfo,
)


class DataStandardizer(HasTraits):
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

    dataset = Instance(xr.Dataset, help="待标准化的数据集")
    techniques = List(Unicode(), help="技术类型标识符列表（如 ['echem', 'peis']）")
    instrument = Unicode(None, allow_none=True, help="仪器标识符")

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
            **kwargs: 其他 traitlets 参数
        """
        if isinstance(techniques, str):
            techniques = [techniques]
        super().__init__(
            dataset=dataset.copy(deep=True),
            techniques=[t.lower() for t in techniques],
            instrument=instrument,
            **kwargs,
        )

    def _get_mappings_for_technique(self, tech: str) -> dict[str, str]:
        """获取指定技术的列名映射。

        Args:
            tech: 技术类型标识符

        Returns:
            列名映射字典
        """
        tech_category = tech.lower()

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

        # 应用重命名
        rename_dict = {}
        # 检查数据变量和坐标
        all_names = list(self.dataset.data_vars) + list(self.dataset.coords)
        for name in all_names:
            old_name = str(name)
            if old_name in mapping:
                new_name = mapping[old_name]
                if new_name != old_name:
                    # 避免冲突
                    if new_name in self.dataset:
                        # 如果目标名称已存在且数据相同，删除旧变量
                        if old_name in self.dataset:
                            self.dataset = self.dataset.drop_vars(old_name)
                        continue
                    rename_dict[old_name] = new_name

        if rename_dict:
            self.dataset = self.dataset.rename(rename_dict)

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
            self.dataset = self.dataset[existing_vars + other_vars]  # type: ignore
            break  # 只应用第一个技术的顺序

    def standardize_units(self) -> "DataStandardizer":  # noqa: PLR0912
        """Convert units to standard echemistpy conventions."""
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
            var_data = self.dataset[var_name]
            var_lower = var_name.lower()

            # Handle time conversions
            if var_name == "time_s":
                # Convert float seconds to timedelta64[ns]
                if var_data.dtype != "timedelta64[ns]":
                    conversions[var_name] = lambda x: (x * 1e9).astype("timedelta64[ns]")

            elif "time" in var_lower or var_name == "t":
                if "min" in var_lower and "mah" not in var_lower:
                    conversions[var_name] = lambda x: x * 60
                    renames[var_name] = var_name.replace("min", "s")
                elif "h" in var_lower and "mah" not in var_lower:
                    conversions[var_name] = lambda x: x * 3600
                    renames[var_name] = var_name.replace("h", "s")

            # Handle current conversions
            elif "current" in var_lower or var_name.startswith("I"):
                if "/a" in var_lower or "_a" in var_lower:
                    conversions[var_name] = lambda x: x * 1000
                    renames[var_name] = replace_suffixes(var_name, {"/A": "/mA", "_A": "_mA", "/a": "/mA", "_a": "_mA"})
                elif "/μa" in var_lower or "/ua" in var_lower:
                    conversions[var_name] = lambda x: x / 1000
                    renames[var_name] = replace_suffixes(var_name, {"/μA": "/mA", "/uA": "/mA", "/μa": "/mA", "/ua": "/mA"})

            # Handle voltage conversions
            elif ("voltage" in var_lower or "potential" in var_lower or var_name.startswith("E")) and "/mv" in var_lower:
                conversions[var_name] = lambda x: x / 1000
                renames[var_name] = replace_suffixes(var_name, {"/mV": "/V", "/mv": "/V"})

            # Handle capacity conversions
            elif ("capacity" in var_lower or var_name.startswith("Q")) and "/uah" in var_lower:
                conversions[var_name] = lambda x: x / 1000
                renames[var_name] = replace_suffixes(var_name, {"/uAh": "/mAh", "/uah": "/mAh"})

        # 批量执行转换（按依赖顺序，先转换再重命名）
        for var_name, converter in conversions.items():
            self.dataset[var_name] = converter(self.dataset[var_name])

        if renames:
            self.dataset = self.dataset.rename(renames)

        return self

    def ensure_required_columns(self, required_columns: list[str]) -> "DataStandardizer":
        """Ensure that required columns exist, creating placeholders if needed."""
        missing_cols = []
        for col in required_columns:
            if col not in self.dataset.data_vars:
                missing_cols.append(col)

        if missing_cols:
            warnings.warn(
                f"Missing required columns: {missing_cols}. Creating placeholders.",
                stacklevel=2,
            )
            # Create placeholder columns with NaN values
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
        """Return the standardized dataset."""
        return self.dataset


def standardize_names(
    raw_data: RawData,
    raw_data_info: RawDataInfo,
    technique_hint: Optional[str | list[str]] = None,
    custom_mapping: Optional[Dict[str, str]] = None,
    required_columns: Optional[list[str]] = None,
) -> Tuple[RawData, RawDataInfo]:
    """Standardize data to consistent format.

    Args:
        raw_data: Input data
        raw_data_info: Input metadata
        technique_hint: Override technique detection (string or list of strings)
        custom_mapping: Additional column name mappings
        required_columns: List of columns that must be present

    Returns:
        Tuple of (RawData, RawDataInfo) with standardized data
    """
    # Extract dataset from RawData
    if isinstance(raw_data.data, xr.Dataset):
        # Determine techniques
        techniques = technique_hint or raw_data_info.get("technique") or ["unknown"]
        if isinstance(techniques, str):
            techniques = [techniques]

        # Standardize data
        standardizer = DataStandardizer(dataset=raw_data.data, techniques=techniques, instrument=raw_data_info.instrument)
        standardizer.standardize(custom_mapping)

        if required_columns:
            standardizer.ensure_required_columns(required_columns)

        standardized_data = RawData(data=standardizer.get_dataset())
    elif isinstance(raw_data.data, xr.DataTree):
        # Determine global techniques
        global_techniques = technique_hint or raw_data_info.get("technique") or ["unknown"]
        if isinstance(global_techniques, str):
            global_techniques = [global_techniques]

        def _standardize_node(ds: xr.Dataset) -> xr.Dataset:
            if not ds.data_vars:
                return ds

            # Use node-specific techniques if available, otherwise use global
            node_techniques = ds.attrs.get("technique", global_techniques)
            if isinstance(node_techniques, str):
                node_techniques = [node_techniques]

            s = DataStandardizer(dataset=ds, techniques=node_techniques, instrument=raw_data_info.instrument)
            s.standardize(custom_mapping)
            if required_columns:
                s.ensure_required_columns(required_columns)

            standardized_ds = s.get_dataset()

            # Sanitize names for DataTree compatibility using utility function
            # 强制转换为 Dataset，因为 sanitize_variable_names 返回 Union
            result = sanitize_variable_names(standardized_ds)
            return result if isinstance(result, xr.Dataset) else standardized_ds

        standardized_tree = raw_data.data.map_over_datasets(_standardize_node)
        standardized_data = RawData(data=standardized_tree)
        techniques = global_techniques
    else:
        raise ValueError("RawData must contain an xarray.Dataset or DataTree for standardization")

    # Create standardized info
    info = raw_data_info.copy()
    info.technique = techniques

    return standardized_data, info


__all__ = [
    "DataStandardizer",
    "standardize_names",
]
