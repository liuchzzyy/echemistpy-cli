"""数据转换命令。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from echemistpy.io.conversion import convert_data as convert_path


def convert(
    source: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--out", "-o", help="输出文件路径。")],
    fmt: Annotated[str | None, typer.Option("--format", "-f", help="输入格式覆盖，例如 .ccs。")] = None,
    instrument: Annotated[str | None, typer.Option("--instrument", "-i", help="读取器仪器名称。")] = None,
    raw: Annotated[bool, typer.Option("--raw", help="保留 reader 原始列名，不转换为标准 schema 名称。")] = False,
) -> None:
    """将支持的数据文件转换为 CSV 或 NetCDF。"""
    written = convert_path(
        source,
        output,
        fmt=fmt,
        instrument=instrument,
        standardize=not raw,
    )
    typer.echo(str(written))
