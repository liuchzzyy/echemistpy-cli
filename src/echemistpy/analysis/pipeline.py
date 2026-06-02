"""High-level orchestration for data analysis workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from echemistpy.io import load
from echemistpy.io.structures import AnalysisData, AnalysisDataInfo

from .registry import TechniqueRegistry, create_default_registry

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates loading, analysis, and result management.

    This class provides a unified interface for processing experimental data
    files from raw format to analyzed results.
    """

    def __init__(self, registry: TechniqueRegistry | None = None) -> None:
        """Initialize the pipeline with an analyzer registry.

        Args:
            registry: Optional custom registry. If None, uses the default registry.
        """
        self.registry = registry or create_default_registry()

    def run(
        self,
        path: str | Path,
        technique: str | None = None,
        instrument: str | None = None,
        **kwargs: Any,
    ) -> tuple[AnalysisData, AnalysisDataInfo]:
        """Run the full pipeline for a single file.

        Args:
            path: Path to the data file
            technique: Technique identifier (if None, will try to detect from file)
            instrument: Instrument identifier (passed to loader)
            **kwargs: Additional arguments passed to both loader and analyzer

        Returns:
            Tuple of (AnalysisData, AnalysisDataInfo)

        Raises:
            ValueError: If technique cannot be determined or analyzer is missing
            FileNotFoundError: If the specified path does not exist
        """
        # 验证输入路径
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info("Starting analysis pipeline for: %s", path)

        # 1. Load data
        try:
            logger.debug("Loading data from %s", path)
            raw_data, raw_info = load(path, technique=technique, instrument=instrument, **kwargs)
            logger.debug("Data loaded successfully. Technique: %s, Instrument: %s", raw_info.technique, raw_info.instrument)
        except Exception as e:
            logger.error("Failed to load data from %s: %s", path, e)
            raise

        # 2. Determine technique
        if technique is None:
            # Try to get technique from raw_info
            techniques = raw_info.technique
            if techniques and techniques[0] != "Unknown":
                technique = techniques[0]
                logger.debug("Auto-detected technique: %s", technique)
            else:
                error_msg = f"Could not detect technique for {path}. Please specify 'technique' explicitly."
                logger.error(error_msg)
                raise ValueError(error_msg)

        # 3. Get analyzer
        try:
            analyzer = self.registry.get_analyzer(technique, raw_info.instrument)
            logger.debug("Using analyzer: %s", analyzer.name)
        except KeyError as exc:
            error_msg = f"No analyzer registered for technique '{technique}' and instrument '{raw_info.instrument}'."
            logger.error(error_msg)
            available_techniques = self.registry.available()
            logger.info("Available techniques: %s", available_techniques)
            raise ValueError(error_msg) from exc

        # 4. Analyze
        try:
            logger.debug("Running analysis with %s", analyzer.name)
            results_data, results_info = analyzer.analyze(raw_data, raw_info, **kwargs)
            logger.info("Analysis completed successfully for: %s", path)
            return results_data, results_info
        except Exception as e:
            logger.error("Analysis failed for %s: %s", path, e)
            raise


def run_analysis(
    path: str | Path,
    technique: str | None = None,
    instrument: str | None = None,
    **kwargs: Any,
) -> tuple[AnalysisData, AnalysisDataInfo]:
    """Convenience function to run analysis on a file using default settings.

    Args:
        path: Path to the data file
        technique: Technique identifier
        instrument: Instrument identifier
        **kwargs: Additional arguments

    Returns:
        Tuple of (AnalysisData, AnalysisDataInfo)
    """
    pipeline = AnalysisPipeline()
    return pipeline.run(path, technique=technique, instrument=instrument, **kwargs)
