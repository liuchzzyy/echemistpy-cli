"""Public data containers for echemistpy services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xarray as xr

from echemistpy.data.schema import ANALYSIS_SCHEMA, RAW_SCHEMA
from echemistpy.io.structures import (
    AnalysisData,
    AnalysisDataInfo,
    RawData,
    RawDataInfo,
    ResultsData,
    ResultsDataInfo,
)


@dataclass
class DataBundle:
    """Standard raw data plus metadata and provenance."""

    data: xr.Dataset | xr.DataTree
    meta: RawDataInfo
    schema: str = RAW_SCHEMA
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisBundle:
    """Standard analysis result plus metadata and provenance."""

    data: xr.Dataset | xr.DataTree
    meta: AnalysisDataInfo
    schema: str = ANALYSIS_SCHEMA
    provenance: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


__all__ = [
    "AnalysisBundle",
    "AnalysisData",
    "AnalysisDataInfo",
    "DataBundle",
    "RawData",
    "RawDataInfo",
    "ResultsData",
    "ResultsDataInfo",
]
