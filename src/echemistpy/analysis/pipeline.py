"""数据分析流程编排。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from echemistpy.data.models import AnalysisBundle
from echemistpy.io import load

from .registry import TechniqueRegistry, create_default_registry

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """编排加载、分析和结果返回。"""

    def __init__(self, registry: TechniqueRegistry | None = None) -> None:
        """初始化分析流程。

        Args:
            registry: 可选分析器注册表；为空时使用默认注册表。
        """
        self.registry = registry or create_default_registry()

    def run(
        self,
        path: str | Path,
        technique: str | None = None,
        instrument: str | None = None,
        **kwargs: Any,
    ) -> AnalysisBundle:
        """运行完整分析流程并返回 AnalysisBundle。

        Args:
            path: 数据文件路径
            technique: 技术类型；为空时从元数据推断
            instrument: 仪器标识符
            **kwargs: 传给加载器和分析器的额外参数

        Returns:
            分析结果数据包

        Raises:
            ValueError: 无法确定技术类型或找不到分析器
            FileNotFoundError: 数据文件不存在
        """
        # 验证输入路径
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        logger.info("开始分析流程: %s", path)

        # 1. 加载数据
        try:
            logger.debug("加载数据: %s", path)
            raw_bundle = load(path, technique=technique, instrument=instrument, **kwargs)
            raw_info = raw_bundle.meta
            logger.debug("数据加载成功。技术类型: %s, 仪器: %s", raw_info.technique, raw_info.instrument)
        except Exception as e:
            logger.error("加载数据失败 %s: %s", path, e)
            raise

        # 2. 确定技术类型
        if technique is None:
            techniques = raw_info.technique
            if techniques and techniques[0] != "unknown":
                technique = techniques[0]
                logger.debug("自动识别技术类型: %s", technique)
            else:
                error_msg = f"无法识别 {path} 的技术类型，请显式指定 technique。"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # 3. 获取分析器
        try:
            analyzer = self.registry.get_analyzer(technique, raw_info.instrument)
            logger.debug("使用分析器: %s", analyzer.name)
        except KeyError as exc:
            error_msg = f"未注册 technique='{technique}' 且 instrument='{raw_info.instrument}' 的分析器。"
            logger.error(error_msg)
            available_techniques = self.registry.available()
            logger.info("可用技术类型: %s", available_techniques)
            raise ValueError(error_msg) from exc

        # 4. 执行分析
        try:
            logger.debug("运行分析器: %s", analyzer.name)
            result_bundle = analyzer.analyze(raw_bundle, **kwargs)
            logger.info("分析完成: %s", path)
            return result_bundle
        except Exception as e:
            logger.error("分析失败 %s: %s", path, e)
            raise


def run_analysis(
    path: str | Path,
    technique: str | None = None,
    instrument: str | None = None,
    **kwargs: Any,
) -> AnalysisBundle:
    """使用默认注册表运行分析。

    Args:
        path: 数据文件路径
        technique: 技术类型
        instrument: 仪器标识符
        **kwargs: 额外参数

    Returns:
        分析结果数据包
    """
    pipeline = AnalysisPipeline()
    return pipeline.run(path, technique=technique, instrument=instrument, **kwargs)
