"""Formats command."""

from __future__ import annotations

import typer


def formats(domain: str | None = None) -> None:
    """Print supported reader formats, optionally filtered by domain."""
    from echemistpy.io import list_reader_specs  # noqa: PLC0415

    domain_key = domain.lower() if domain else None
    for spec in list_reader_specs():
        if domain_key is not None and domain_key not in {tech.lower() for tech in spec.techniques}:
            continue
        extensions = ",".join(spec.extensions)
        instruments = ",".join(spec.instruments)
        techniques = ",".join(spec.techniques)
        directory = "yes" if spec.supports_directory else "no"
        typer.echo(f"{extensions}\t{spec.name}\t{instruments}\t{techniques}\tdirectory={directory}")


__all__ = ["formats"]
