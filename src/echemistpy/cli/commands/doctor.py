"""运行环境检查命令。"""

from __future__ import annotations

import importlib.util
from typing import Any

import typer


def check_runtime() -> list[tuple[str, str, Any]]:
    """返回核心运行环境检查结果，不导入分析模块。"""
    checks: list[tuple[str, str, Any]] = []

    try:
        import echemistpy  # noqa: PLC0415

        checks.append(("ok", "package", getattr(echemistpy, "__version__", "unknown")))
    except Exception as exc:
        checks.append(("fail", "package", exc))

    try:
        from echemistpy.data import RAW_SCHEMA, names  # noqa: PLC0415

        checks.append(("ok", "data_schema", f"{RAW_SCHEMA}; echem={','.join(names('echem'))}"))
    except Exception as exc:
        checks.append(("fail", "data_schema", exc))

    try:
        from echemistpy.io import list_supported_formats  # noqa: PLC0415

        checks.append(("ok", "readers", f"{len(list_supported_formats())} formats"))
    except Exception as exc:
        checks.append(("fail", "readers", exc))

    for module_name, label in (
        ("typer", "typer"),
        ("pytest", "pytest"),
        ("larch", "xraylarch"),
        ("umap", "umap"),
    ):
        state = "ok" if importlib.util.find_spec(module_name) else "skip"
        detail = "installed" if state == "ok" else "not installed"
        checks.append((state, label, detail))

    return checks


def doctor() -> None:
    """打印运行环境检查结果。"""
    checks = check_runtime()
    for state, label, detail in checks:
        suffix = f" {detail}" if detail else ""
        typer.echo(f"{state}\t{label}{suffix}")
    if any(state == "fail" for state, _, _ in checks):
        raise typer.Exit(1)


__all__ = ["check_runtime", "doctor"]
