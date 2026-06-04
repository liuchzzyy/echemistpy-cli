"""分析器注册表。"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

from echemistpy.data.models import AnalysisBundle, DataBundle


class TechniqueAnalyzer(ABC):
    """分析器基类。"""

    technique = ""
    instrument: str | None = None
    name = ""

    def __init__(self, **kwargs: Any) -> None:
        """复制类级配置并应用实例级覆盖。"""
        for cls in reversed(self.__class__.mro()):
            if not issubclass(cls, TechniqueAnalyzer):
                continue
            for key, value in vars(cls).items():
                if key.startswith("_") or key in {"required_columns"}:
                    continue
                if isinstance(value, property | staticmethod | classmethod) or callable(value):
                    continue
                setattr(self, key, copy.deepcopy(value))

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise TypeError(f"{self.__class__.__name__} 不支持参数: {key}")
            setattr(self, key, value)

        if not self.name:
            self.name = self.__class__.__name__

    def analyze(
        self,
        bundle: DataBundle,
        **kwargs: Any,  # noqa: ARG002
    ) -> AnalysisBundle:
        """执行完整分析流程。"""
        self.validate(bundle)
        cleaned = self.preprocess(bundle.copy())
        result = self._compute(cleaned)
        result.meta = bundle.meta.copy()
        result.warnings = [*bundle.warnings, *result.warnings]
        provenance = dict(bundle.provenance)
        provenance.update(result.provenance)
        provenance["analyzer"] = self.name
        result.provenance = provenance
        return result

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """分析所需标准列。"""

    def validate(self, bundle: DataBundle) -> None:
        """检查数据包是否包含必需列。"""
        available = set(bundle.variables) | set(bundle.coords)
        missing = [col for col in self.required_columns if col not in available]
        if missing:
            raise ValueError(f"分析器 '{self.name}' 需要列 {self.required_columns}，但数据缺少 {missing}。")

    def preprocess(self, bundle: DataBundle) -> DataBundle:  # noqa: PLR6301
        """可选预处理步骤。"""
        return bundle

    @abstractmethod
    def _compute(self, bundle: DataBundle) -> AnalysisBundle:
        """执行核心计算并返回 AnalysisBundle。"""


class TechniqueRegistry:
    """将技术类型和仪器标识符映射到分析器实例。"""

    def __init__(self, analyzers: list[TechniqueAnalyzer] | None = None) -> None:
        """初始化分析器注册表。"""
        self._analyzers = list(analyzers or [])

    def register(self, analyzer: TechniqueAnalyzer) -> None:
        """注册分析器实例。

        Args:
            analyzer: TechniqueAnalyzer 实例
        """
        if analyzer not in self._analyzers:
            self._analyzers.append(analyzer)

    def unregister(self, analyzer: TechniqueAnalyzer) -> None:
        """注销分析器实例。

        Args:
            analyzer: TechniqueAnalyzer 实例
        """
        if analyzer in self._analyzers:
            self._analyzers.remove(analyzer)

    def get_analyzer(self, technique: str, instrument: str | None = None) -> TechniqueAnalyzer:
        """按技术类型和可选仪器获取分析器。

        Args:
            technique: 技术类型标识符（大小写不敏感）
            instrument: 可选仪器标识符（大小写不敏感）

        Returns:
            TechniqueAnalyzer 实例

        Raises:
            KeyError: 未找到匹配分析器
        """
        tech_lower = technique.lower()
        inst_lower = instrument.lower() if instrument else None

        # 1. 优先匹配指定仪器。
        if inst_lower:
            for a in self._analyzers:
                if _supports_technique(a, tech_lower) and a.instrument and a.instrument.lower() == inst_lower:
                    return a

        # 2. 匹配通用技术类型分析器（分析器未声明仪器）。
        for a in self._analyzers:
            if _supports_technique(a, tech_lower) and not a.instrument:
                return a

        # 3. 回退到第一个技术类型匹配项。
        for a in self._analyzers:
            if _supports_technique(a, tech_lower):
                return a

        raise KeyError(f"未注册技术类型 '{technique}' 的分析器" + (f"，instrument='{instrument}'" if instrument else ""))

    def available(self) -> list[str]:
        """返回已注册的技术类型列表。

        Returns:
            可用技术类型标识符列表
        """
        return sorted({tech for a in self._analyzers for tech in _technique_names(a)})

    def __contains__(self, technique: str) -> bool:
        """检查技术类型是否已注册。"""
        return any(_supports_technique(a, technique.lower()) for a in self._analyzers)

    def __len__(self) -> int:
        """返回已注册分析器数量。"""
        return len(self._analyzers)


def create_default_registry() -> TechniqueRegistry:
    """返回包含内置分析器的注册表。

    Returns:
        包含标准分析器的 TechniqueRegistry
    """
    from .echem import GCDAnalyzer  # noqa: PLC0415
    from .stxm import STXMAnalyzer  # noqa: PLC0415
    from .xas import XASAnalyzer  # noqa: PLC0415

    registry = TechniqueRegistry()
    registry.register(GCDAnalyzer())
    registry.register(STXMAnalyzer())
    registry.register(XASAnalyzer())
    return registry


def _technique_names(analyzer: TechniqueAnalyzer) -> tuple[str, ...]:
    """返回分析器支持的所有技术名。"""
    supported = getattr(analyzer, "supported_techniques", None)
    if supported:
        return tuple(str(tech) for tech in supported)
    return (analyzer.technique,)


def _supports_technique(analyzer: TechniqueAnalyzer, technique: str) -> bool:
    """判断分析器是否支持指定技术名。"""
    return technique in {name.lower() for name in _technique_names(analyzer)}
