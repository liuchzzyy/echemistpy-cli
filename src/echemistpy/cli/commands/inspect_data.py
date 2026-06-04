"""数据检查命令。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from echemistpy.io.summary import inspect_data as inspect_path


def inspect_data(
    source: Annotated[Path, typer.Argument(exists=True, readable=True)],
    fmt: Annotated[str | None, typer.Option("--format", "-f", help="输入格式覆盖，例如 .ccs。")] = None,
    instrument: Annotated[str | None, typer.Option("--instrument", "-i", help="读取器仪器名称。")] = None,
    raw: Annotated[bool, typer.Option("--raw", help="显示 reader 原始列名，不显示标准 schema 名称。")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="输出 JSON。")] = False,
) -> None:
    """检查一个支持的数据文件或目录。"""
    summary = inspect_path(
        source,
        fmt=fmt,
        instrument=instrument,
        standardize=not raw,
    )
    summary_dict = summary.to_dict()
    if as_json:
        typer.echo(json.dumps(summary_dict, ensure_ascii=False, indent=2))
        return

    typer.echo(f"Path: {summary.path}")
    typer.echo(f"Schema: {summary.schema}")
    typer.echo(f"Sample: {summary.sample_name}")
    typer.echo(f"Instrument: {summary.instrument}")
    typer.echo(f"Technique: {','.join(summary.technique)}")
    typer.echo(f"Dims: {summary.dims}")
    typer.echo(f"Variables: {', '.join(summary.variables)}")
    typer.echo(f"Coords: {', '.join(summary.coords)}")


__all__ = ["inspect_data"]
