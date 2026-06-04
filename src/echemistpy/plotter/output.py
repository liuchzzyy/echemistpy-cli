"""绘图输出路径工具。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from echemistpy.plotter.contracts import PlotResult


def timestamped_log_dir(
    *,
    domain: str = "echem",
    root: str | Path = "log",
    timestamp: str | None = None,
) -> Path:
    """创建 ``{domain}_{timestamp}`` 格式的绘图日志目录。"""
    safe_domain = _safe_path_token(domain)
    time_token = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(root) / f"{safe_domain}_{time_token}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plot_result(
    result: PlotResult,
    filename: str | Path,
    *,
    output_dir: str | Path,
    dpi: int = 300,
) -> Path:
    """保存 PlotResult 并返回输出路径。"""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    result.figure.savefig(path, dpi=dpi)
    return path


def _safe_path_token(value: str) -> str:
    """将任意文本转换为适合文件夹名的 token。"""
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "plot"


__all__ = ["save_plot_result", "timestamped_log_dir"]
