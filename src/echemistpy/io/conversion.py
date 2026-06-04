"""数据格式转换辅助函数。"""

from __future__ import annotations

from pathlib import Path

from echemistpy.io.loaders import load
from echemistpy.io.writer import write_bundle


def convert_data(
    source: str | Path,
    output: str | Path,
    *,
    fmt: str | None = None,
    instrument: str | None = None,
    standardize: bool = True,
) -> Path:
    """加载源数据并写出为支持的目标格式。"""
    output_path = Path(output)
    bundle = load(
        source,
        fmt=fmt,
        instrument=instrument,
        standardize=standardize,
    )

    return write_bundle(bundle, output_path)


__all__ = ["convert_data"]
