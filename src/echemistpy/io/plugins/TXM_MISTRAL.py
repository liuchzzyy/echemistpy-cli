# -*- coding: utf-8 -*-
"""MISTRAL 线站 TXM HDF5 文件读取器。"""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import xarray as xr

from echemistpy.data.models import DataBundle
from echemistpy.data.utils import apply_standard_attrs_txm
from echemistpy.io.base_reader import BaseReader
from echemistpy.io.contracts import ReaderSpec

logger = logging.getLogger(__name__)


class MISTRALReader(BaseReader):
    """MISTRAL TXM .hdf5 文件读取器。"""

    # --- 解析常量 ---
    INSTRUMENT_NAME: ClassVar[str] = "ALBA_MISTRAL"
    DEFAULT_TECHNIQUE: ClassVar[list[str]] = ["txm", "ex situ"]
    DATE_REGEX: ClassVar[re.Pattern[str]] = re.compile(r"(\d{8})")

    # --- reader 能力声明 ---
    supports_directories: ClassVar[bool] = False
    instrument: ClassVar[str] = "mistral"
    spec: ClassVar[ReaderSpec] = ReaderSpec(
        name="mistral_hdf5",
        extensions=(".hdf5",),
        instruments=("mistral",),
        techniques=("txm",),
        supports_directory=False,
        description="ALBA MISTRAL TXM HDF5 files",
    )

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """初始化 MISTRAL reader。

        Args:
            filepath: HDF5 文件或目录路径
            **kwargs: 额外元数据覆盖项
        """
        # 设置默认技术类型。
        if "technique" not in kwargs:
            kwargs["technique"] = self.DEFAULT_TECHNIQUE
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> DataBundle:
        """加载单个 MISTRAL HDF5 文件。

        Args:
            path: HDF5 文件路径
            **kwargs: 额外参数（未使用）

        Returns:
            DataBundle 数据包
        """
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("MISTRAL TXM 读取需要安装 echemistpy-cli[txm]。") from exc

        with h5py.File(path, "r") as f:
            if "SpecNormalized" not in f:
                raise ValueError(f"文件 {path} 不包含 'SpecNormalized' group。")

            group = f["SpecNormalized"]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"{path} 中的 'SpecNormalized' 不是有效 HDF5 Group。")

            ds = self._extract_dataset(group)

            # 从文件名提取日期。
            start_time = self._extract_date(path.name)

            # 创建元数据。
            metadata = {"file_path": str(path), "start_time": start_time}
            raw_info = self._create_metadata(metadata, default_sample_name=path.stem)

            return DataBundle(data=ds, meta=raw_info)

    def _extract_dataset(self, group: Any) -> xr.Dataset:
        """从 HDF5 group 提取数据并创建 xarray.Dataset。

        Args:
            group: 包含数据的 HDF5 group

        Returns:
            提取后的 xarray.Dataset
        """
        data_cube = group["spectroscopy_normalized_aligned"][:]
        energy = group["energy"][:]
        rotation_angle = group["rotation_angle"][:] if "rotation_angle" in group else None

        x_pixel_size = group["x_pixel_size"][0] if "x_pixel_size" in group else 1.0
        y_pixel_size = group["y_pixel_size"][0] if "y_pixel_size" in group else 1.0

        # 创建坐标。
        x_coords = np.arange(data_cube.shape[2]) * x_pixel_size
        y_coords = np.arange(data_cube.shape[1]) * y_pixel_size

        # 创建 Dataset。
        ds = xr.Dataset(
            data_vars={
                "transmission": (["energy", "y", "x"], data_cube),
                "optical_density": (["energy", "y", "x"], -np.log(data_cube.astype(np.float64))),
            },
            coords={
                "energy": energy,
                "y": y_coords,
                "x": x_coords,
            },
            attrs={
                "x_pixel_size": x_pixel_size,
                "y_pixel_size": y_pixel_size,
                "instrument": self.instrument,
            },
        )

        if rotation_angle is not None:
            ds["rotation_angle"] = (["energy"], rotation_angle)

        apply_standard_attrs_txm(ds)
        return ds

    @staticmethod
    def _extract_date(filename: str) -> str | None:
        """从文件名提取日期，例如 20230701 -> 2023-07-01。

        Args:
            filename: 文件名

        Returns:
            YYYY-MM-DD 格式日期字符串，无法提取时返回 None
        """
        match = MISTRALReader.DATE_REGEX.search(filename)
        if match:
            with contextlib.suppress(Exception):
                d = match.group(1)
                return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        return None

    @staticmethod
    def _get_file_extension() -> str:
        """返回该 reader 支持的文件扩展名。

        Returns:
            包含点号的文件扩展名
        """
        return ".hdf5"
