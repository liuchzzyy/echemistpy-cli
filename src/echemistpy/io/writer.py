"""DataBundle 和 AnalysisBundle 写出门面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from echemistpy.data.models import AnalysisBundle, DataBundle
from echemistpy.data.storage import save_bundle, save_combined, save_data, save_info

Bundle = DataBundle | AnalysisBundle


def write_bundle(
    bundle: Bundle,
    output: str | Path,
    *,
    fmt: str | None = None,
    **kwargs: Any,
) -> Path:
    """写出数据包并返回输出路径。"""
    output_path = Path(output)
    save_bundle(bundle, output_path, fmt=fmt, **kwargs)
    return output_path


def write_data(
    bundle: Bundle | Sequence[Bundle],
    output: str | Path,
    *,
    fmt: str | None = None,
    **kwargs: Any,
) -> Path:
    """只写出数据部分并返回输出路径。"""
    output_path = Path(output)
    save_data(bundle, output_path, fmt=fmt, **kwargs)
    return output_path


def write_info(
    bundle: Bundle | Sequence[Bundle],
    output: str | Path,
) -> Path:
    """只写出元数据并返回输出路径。"""
    output_path = Path(output)
    save_info(bundle, output_path)
    return output_path


def write_combined(
    bundle: Bundle | Sequence[Bundle],
    output: str | Path,
    **kwargs: Any,
) -> Path:
    """将数据和元数据写入同一个 NetCDF 文件。"""
    output_path = Path(output)
    save_combined(bundle, output_path, **kwargs)
    return output_path


__all__ = [
    "write_bundle",
    "write_combined",
    "write_data",
    "write_info",
]
