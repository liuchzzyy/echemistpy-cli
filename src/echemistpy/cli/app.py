"""echemistpy 命令行应用。"""

from __future__ import annotations

import typer

from echemistpy.cli.commands import convert, doctor, formats, inspect_data

app = typer.Typer(no_args_is_help=True)
app.command("doctor")(doctor)


def _formats_command(domain: str):
    def command() -> None:
        formats(domain=domain)

    command.__name__ = f"{domain}_formats"
    command.__doc__ = f"打印支持的 {domain.upper()} reader 格式。"
    return command


def _domain_app(domain: str) -> typer.Typer:
    domain_app = typer.Typer(no_args_is_help=True)
    domain_app.command("formats")(_formats_command(domain))
    domain_app.command("inspect")(inspect_data)
    domain_app.command("convert")(convert)
    return domain_app


for _domain in ("echem", "xas", "xrd", "txm"):
    app.add_typer(_domain_app(_domain), name=_domain)


def main() -> None:
    """运行 echemistpy CLI。"""
    app()


__all__ = ["app", "main"]
