"""Typer application for the echem command."""

from __future__ import annotations

import typer

from echemistpy.cli.commands import convert, doctor, formats, inspect_data

app = typer.Typer(no_args_is_help=True)
app.command("formats")(formats)
app.command("doctor")(doctor)
app.command("inspect")(inspect_data)
app.command("convert")(convert)


def main() -> None:
    """Run the echem CLI."""
    app()


__all__ = ["app", "main"]
