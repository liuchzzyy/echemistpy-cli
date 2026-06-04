"""Data conversion helpers."""

from __future__ import annotations

from pathlib import Path

from echemistpy.data.storage import save_combined, save_data
from echemistpy.io.loaders import load


def convert_data(
    source: str | Path,
    output: str | Path,
    *,
    fmt: str | None = None,
    instrument: str | None = None,
    standardize: bool = True,
) -> Path:
    """Load source data and write it to a supported output format."""
    output_path = Path(output)
    raw_data, raw_info = load(
        source,
        fmt=fmt,
        instrument=instrument,
        standardize=standardize,
    )

    suffix = output_path.suffix.lower().lstrip(".")
    if suffix in {"nc", "netcdf"}:
        save_combined(raw_data, raw_info, output_path)
    else:
        save_data(raw_data, output_path)
    return output_path


__all__ = ["convert_data"]
