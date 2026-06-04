"""reader 和注册表使用的 I/O 契约。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReaderSpec:
    """Declared reader capability."""

    name: str
    extensions: tuple[str, ...]
    instruments: tuple[str, ...]
    techniques: tuple[str, ...]
    supports_directory: bool = False
    can_inspect: bool = True
    description: str = ""


__all__ = ["ReaderSpec"]
