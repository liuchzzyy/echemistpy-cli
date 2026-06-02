"""Registries that keep track of available analyzers."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar

from traitlets import HasTraits, Instance, MetaHasTraits, Unicode
from traitlets import List as TList

from echemistpy.io.structures import AnalysisData, AnalysisDataInfo, RawData, RawDataInfo


class ABCMetaHasTraits(ABCMeta, MetaHasTraits):
    """Metaclass combining ABC and HasTraits metaclasses."""

    pass


class TechniqueAnalyzer(HasTraits, metaclass=ABCMetaHasTraits):
    """Template used by all built-in analyzers."""

    # 可从 RawDataInfo 继承到 AnalysisDataInfo 的元数据字段
    INHERITABLE_METADATA_FIELDS: ClassVar[tuple[str, ...]] = (
        "technique",
        "sample_name",
        "start_time",
        "operator",
        "instrument",
        "active_material_mass",
        "wave_number",
    )

    technique = Unicode(help="Technique identifier")
    instrument = Unicode(None, allow_none=True, help="Instrument identifier")
    name = Unicode(help="Analyzer name")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.name:
            self.name = self.__class__.__name__

    def analyze(
        self,
        raw_data: RawData,
        raw_info: RawDataInfo,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[AnalysisData, AnalysisDataInfo]:
        """Perform the full analysis workflow.

        This includes validation, preprocessing, computation, and packaging.
        Metadata from raw_info is carried over to the results.

        Args:
            raw_data: Standardized raw data container
            raw_info: Metadata from the raw data (required for metadata inheritance)
            **kwargs: Additional parameters (currently unused, for future extension)

        Returns:
            Tuple of (AnalysisData, AnalysisDataInfo)
        """
        # 1. Validate
        self.validate(raw_data)

        # 2. Preprocess (on a copy to avoid side effects)
        cleaned = self.preprocess(raw_data.copy())

        # 3. Compute (returns AnalysisData + AnalysisDataInfo)
        analysis_data, analysis_info = self._compute(cleaned)

        # 4. Inherit metadata from raw_info
        # Copy standard metadata fields (explicitly exclude 'others' field)
        # technique is also inherited from raw_info to preserve original data source information
        for field in self.INHERITABLE_METADATA_FIELDS:
            value = getattr(raw_info, field, None)
            if value is not None:
                setattr(analysis_info, field, value)

        return analysis_data, analysis_info

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Columns that must be present in the data."""

    def validate(self, raw_data: RawData) -> None:
        """Check if raw_data contains all required columns.

        Args:
            raw_data: RawData instance to validate

        Raises:
            ValueError: If any required columns are missing
        """
        # For now, we check the root dataset variables and coordinates
        available = set(raw_data.variables) | set(raw_data.coords)
        missing = [col for col in self.required_columns if col not in available]
        if missing:
            raise ValueError(f"Analyzer '{self.name}' requires columns {self.required_columns}, but {missing} are missing from the data.")

    def preprocess(self, raw_data: RawData) -> RawData:  # noqa: PLR6301
        """Optional preprocessing step (e.g., filtering, normalization)."""
        return raw_data

    @abstractmethod
    def _compute(self, raw_data: RawData) -> tuple[AnalysisData, AnalysisDataInfo]:
        """Perform the main calculation and return results (internal method).

        Note:
            This is an internal method. Users should call analyze() instead,
            which handles validation, preprocessing, computation, and metadata inheritance.

        Args:
            raw_data: Preprocessed data container

        Returns:
            Tuple of (AnalysisData, AnalysisDataInfo) containing:
            - AnalysisData: Processed data as xarray.Dataset or xarray.DataTree
            - AnalysisDataInfo: Analysis parameters and metadata
        """


class TechniqueRegistry(HasTraits):
    """Map technique and instrument identifiers to analyzer instances."""

    _analyzers = TList(Instance(TechniqueAnalyzer), help="Internal list of registered analyzers")

    def register(self, analyzer: TechniqueAnalyzer) -> None:
        """Register an analyzer instance.

        Args:
            analyzer: TechniqueAnalyzer instance
        """
        if analyzer not in self._analyzers:
            self._analyzers.append(analyzer)

    def unregister(self, analyzer: TechniqueAnalyzer) -> None:
        """Unregister an analyzer instance.

        Args:
            analyzer: TechniqueAnalyzer instance
        """
        if analyzer in self._analyzers:
            self._analyzers.remove(analyzer)

    def get_analyzer(self, technique: str, instrument: str | None = None) -> TechniqueAnalyzer:
        """Get analyzer for a technique and optionally an instrument.

        Args:
            technique: Technique identifier (case-insensitive)
            instrument: Optional instrument identifier (case-insensitive)

        Returns:
            TechniqueAnalyzer instance

        Raises:
            KeyError: If no matching analyzer is found
        """
        tech_lower = technique.lower()
        inst_lower = instrument.lower() if instrument else None

        # 1. Try specific instrument match
        if inst_lower:
            for a in self._analyzers:
                if a.technique.lower() == tech_lower and a.instrument and a.instrument.lower() == inst_lower:
                    return a

        # 2. Try generic technique match (no instrument specified in analyzer)
        for a in self._analyzers:
            if a.technique.lower() == tech_lower and not a.instrument:
                return a

        # 3. Fallback to first technique match
        for a in self._analyzers:
            if a.technique.lower() == tech_lower:
                return a

        raise KeyError(f"No analyzer registered for technique '{technique}'" + (f" and instrument '{instrument}'" if instrument else ""))

    def available(self) -> list[str]:
        """Get list of registered techniques.

        Returns:
            List of available technique identifiers
        """
        return sorted({a.technique for a in self._analyzers})

    def __contains__(self, technique: str) -> bool:
        """Check if technique is registered."""
        return any(a.technique.lower() == technique.lower() for a in self._analyzers)

    def __len__(self) -> int:
        """Get number of registered analyzers."""
        return len(self._analyzers)


def create_default_registry() -> TechniqueRegistry:
    """Return a registry populated with the built-in analyzers.

    Returns:
        TechniqueRegistry with standard analyzers
    """
    from .echem import GalvanostaticAnalyzer  # noqa: PLC0415
    from .stxm import STXMAnalyzer  # noqa: PLC0415
    from .xas import XASAnalyzer  # noqa: PLC0415

    registry = TechniqueRegistry()
    registry.register(GalvanostaticAnalyzer())
    registry.register(STXMAnalyzer())
    registry.register(XASAnalyzer())
    return registry
