"""Formats command."""

from __future__ import annotations

import typer


def formats() -> None:
    """Print supported reader formats."""
    from echemistpy.io import list_reader_specs  # noqa: PLC0415

    for spec in list_reader_specs():
        extensions = ",".join(spec.extensions)
        instruments = ",".join(spec.instruments)
        techniques = ",".join(spec.techniques)
        directory = "yes" if spec.supports_directory else "no"
        typer.echo(f"{extensions}\t{spec.name}\t{instruments}\t{techniques}\tdirectory={directory}")


__all__ = ["formats"]
