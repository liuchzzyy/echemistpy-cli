"""Convert command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from echemistpy.io.conversion import convert_data as convert_path


def convert(
    source: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--out", "-o", help="Output file path.")],
    fmt: Annotated[str | None, typer.Option("--format", "-f", help="Input format override, such as .ccs.")] = None,
    instrument: Annotated[str | None, typer.Option("--instrument", "-i", help="Reader instrument name.")] = None,
    raw: Annotated[bool, typer.Option("--raw", help="Keep raw reader names instead of standard schema names.")] = False,
) -> None:
    """Convert a supported data file to CSV or NetCDF."""
    written = convert_path(
        source,
        output,
        fmt=fmt,
        instrument=instrument,
        standardize=not raw,
    )
    typer.echo(str(written))
