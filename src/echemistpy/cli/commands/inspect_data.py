"""Inspect command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from echemistpy.io.summary import inspect_data as inspect_path


def inspect_data(
    source: Annotated[Path, typer.Argument(exists=True, readable=True)],
    fmt: Annotated[str | None, typer.Option("--format", "-f", help="Input format override, such as .ccs.")] = None,
    instrument: Annotated[str | None, typer.Option("--instrument", "-i", help="Reader instrument name.")] = None,
    raw: Annotated[bool, typer.Option("--raw", help="Show raw reader names instead of standard schema names.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON output.")] = False,
) -> None:
    """Inspect one supported data file or directory."""
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
