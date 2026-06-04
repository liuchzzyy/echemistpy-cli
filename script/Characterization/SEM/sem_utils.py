"""
SEM Analysis Utilities for Scanning Electron Microscopy Image Processing.

This module provides tools for processing SEM images, including:
- Data loading with automatic metadata extraction via HyperSpy
- Footer detection and scale bar validation via OCR
- Image preprocessing (cropping, rotation, contrast enhancement)
- Publication-ready figure generation with customizable scale bars
- Batch processing capabilities

Key Features:
- Encapsulated workflow via `SEMAnalyser` class
- Automatic pixel scale detection via HyperSpy (Zeiss, FEI, generic TIFF)
- Multiple contrast enhancement methods (percentile stretch, CLAHE)
- HyperSpy integration for metadata preservation
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import hyperspy.api as hs
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage import exposure, measure

# Configure module-level logger
logger = logging.getLogger(__name__)

# Adjust path to find Figure module if needed
try:
    from Figure.config import OUTPUT_ROOT, setup
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from Figure.config import OUTPUT_ROOT, setup

if TYPE_CHECKING:
    from hyperspy.signals import Signal2D

# Type aliases
FloatArray = np.ndarray


# ==================================================================================================
# 1. Constants & Configuration
# ==================================================================================================

# Default processing parameters
DEFAULT_CROP_BOTTOM = 60  # Pixels to crop from bottom (footer)
DEFAULT_CONTRAST_METHOD = "stretch"
DEFAULT_FIGURE_SIZE = (3.3, 2.5)  # inches
DEFAULT_DPI_LIST = (300, 600)
DEFAULT_FORMATS = ("tif",)

# Unit normalization mapping
UNIT_ALIASES = {
    "um": "µm",
    "micron": "µm",
    "microns": "µm",
    "nm": "nm",
    "mm": "mm",
    "m": "m",
}


@dataclass
class SEMMetadata:
    """
    Metadata extracted from SEM image file.

    Attributes:
        pixel_scale: Physical size per pixel (units/pixel).
        units: Physical units (e.g., 'µm', 'nm').
        original_filename: Name of the source file.
        source: Metadata source ('zeiss', 'fei', 'tiff_standard', 'ocr', 'manual').
    """

    pixel_scale: float | None = None
    units: str | None = None
    original_filename: str | None = None
    source: Literal["zeiss", "fei", "tiff_standard", "ocr", "manual", "unknown"] = "unknown"

    def is_valid(self) -> bool:
        """Check if metadata has valid scale information."""
        return self.pixel_scale is not None and self.pixel_scale > 0 and self.units is not None


@dataclass
class ProcessingParams:
    """
    Parameters for SEM image processing.

    Attributes:
        crop_bottom: Pixels to crop from bottom (0 = auto-detect).
        rotate_angle: Rotation angle in degrees.
        crop_edge: Pixels to crop from all edges after rotation.
        aspect_ratio: Target aspect ratio (width/height). None = no change.
                      Common values: 4/3, 16/9, 1.0 (square), 3/2.
        contrast_method: 'stretch', 'clahe', or 'none'.
        p_low: Lower percentile for contrast stretching (default 2%).
        p_high: Upper percentile for contrast stretching (default 98%).
        clip_limit: Clipping limit for CLAHE (default 0.01).
        check_ocr: Whether to verify scale with OCR (slow, default False).
    """

    crop_bottom: int = 0  # 0 = auto-detect
    rotate_angle: float = 0.0
    crop_edge: int = 0
    aspect_ratio: float | None = None  # Target aspect ratio (width/height)
    contrast_method: Literal["stretch", "clahe", "none"] = "stretch"
    p_low: float = 2.0
    p_high: float = 98.0
    clip_limit: float = 0.01
    check_ocr: bool = False


@dataclass
class ScaleBarConfig:
    """
    Configuration for scale bar rendering.

    Attributes:
        size: Physical size of scale bar (in units).
        units: Display units (e.g., 'µm', 'nm').
        color: Scale bar and text color.
        location: Anchor location ('lower left', 'lower right', etc.).
        sep: Separation between bar and label.
        bbox_offset: (x, y, w, h) for custom positioning in axes coordinates.
        font_size: Font size for label.
        font_weight: Font weight ('bold', 'normal').
        height_fraction: Height of bar relative to image width (default 0.015).
        frameon: Whether to draw a background box.
        frame_alpha: Opacity of background box.
        frame_color: Color of background box.
    """

    size: float | None = None
    units: str | None = None
    color: str = "white"
    location: str = "lower left"
    sep: int = 2
    bbox_offset: tuple[float, float, float, float] | None = None
    font_size: int | None = None
    font_weight: str = "bold"
    height_fraction: float = 0.015
    frameon: bool = False
    frame_alpha: float = 1.0
    frame_color: str = "black"


@dataclass
class SEMResult:
    """
    Results from SEM image processing.

    Attributes:
        image: Processed image array.
        metadata: Extracted/updated metadata.
        processing_params: Parameters used for processing.
        figure: Matplotlib figure (if generated).
        hyperspy_signal: HyperSpy Signal2D (if created).
    """

    image: FloatArray
    metadata: SEMMetadata
    processing_params: ProcessingParams
    figure: plt.Figure | None = None
    hyperspy_signal: Any = None


# ==================================================================================================
# 2. Metadata Extraction
# ==================================================================================================


def _normalize_units(units: str | None) -> str:
    """Normalize unit strings to standard format."""
    if units is None:
        return "pixels"
    units_lower = units.lower().strip()
    return UNIT_ALIASES.get(units_lower, units)


def _convert_scale_to_display_units(scale_m: float) -> tuple[float, str]:
    """
    Convert scale from meters to appropriate display units (nm or µm).

    Args:
        scale_m: Scale in meters per pixel.

    Returns:
        Tuple of (scale_value, units_string).
    """
    scale_nm = scale_m * 1e9  # meters -> nm
    if scale_nm >= 1000:
        return scale_nm / 1000, "µm"  # Use µm for large scales
    return scale_nm, "nm"


def _detect_metadata_source(original_metadata: dict) -> str:
    """Detect the instrument source from HyperSpy original_metadata."""
    if "CZ_SEM" in original_metadata:
        return "zeiss"
    if "fei_metadata" in original_metadata or "FEI_HELIOS" in original_metadata:
        return "fei"
    return "hyperspy"


def read_data(file_path: str | Path) -> tuple[FloatArray, SEMMetadata]:
    """
    Read SEM image data and metadata using HyperSpy.

    HyperSpy automatically extracts calibrated pixel scales from Zeiss (CZ_SEM)
    and FEI/Thermo Fisher metadata, returning scales in meters. This function
    converts to nm or µm for convenient display.

    Args:
        file_path: Path to the SEM image file.

    Returns:
        Tuple of (image_data, SEMMetadata).

    Raises:
        FileNotFoundError: If file does not exist.
        OSError: If file cannot be loaded.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = SEMMetadata(original_filename=file_path.name)

    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s: Any = hs.load(str(file_path))

    image_data = s.data

    # Extract scale from axes_manager (HyperSpy returns meters for SEM)
    scale = s.axes_manager[0].scale
    units = s.axes_manager[0].units

    # Check if HyperSpy found valid calibration
    if scale != 1.0 and units and units.lower() in {"m", "meter", "meters"}:
        # Convert meters to nm/µm
        display_scale, display_units = _convert_scale_to_display_units(scale)
        metadata.pixel_scale = display_scale
        metadata.units = display_units

        # Detect source from original_metadata
        try:
            om = s.original_metadata.as_dictionary()
            metadata.source = _detect_metadata_source(om)  # type: ignore
        except Exception:
            metadata.source = "hyperspy"  # type: ignore

    elif scale != 1.0 and units and units.lower() not in {"px", "pixels", ""}:
        # Non-meter units (already calibrated, e.g., nm or µm)
        metadata.pixel_scale = scale
        metadata.units = _normalize_units(units)
        try:
            om = s.original_metadata.as_dictionary()
            metadata.source = _detect_metadata_source(om)  # type: ignore
        except Exception:
            metadata.source = "hyperspy"  # type: ignore

    return image_data, metadata


# ==================================================================================================
# 3. Footer Detection & Scale Bar Validation
# ==================================================================================================


def _detect_footer(image: FloatArray) -> int:
    """
    Detect footer height via gradient analysis. Handles Zeiss and FEI styles.

    Args:
        image: Input image array.

    Returns:
        Footer height in pixels (0 if none found).
    """
    h, _w = image.shape[:2]

    # Search in bottom 15% of image (footers are usually smaller)
    search_h = int(h * 0.15)
    if search_h < 10:
        return 0

    bottom_region = image[h - search_h :, ...]
    row_means = np.mean(bottom_region, axis=1) if bottom_region.ndim == 2 else np.mean(bottom_region, axis=(1, 2))

    # Calculate gradient
    grad = np.diff(row_means)
    abs_grad = np.abs(grad)

    # Determine intensity normalization
    if image.dtype == np.uint8:
        max_val = 255
    elif image.dtype == np.uint16:
        max_val = 65535
    else:
        max_val = np.max(image) if np.max(image) > 1.0 else 1.0

    # Threshold: significant gradient is > 5% of max value
    grad_threshold = max_val * 0.05

    # Find strongest gradient that meets size constraints
    sorted_indices = np.argsort(abs_grad)[::-1]

    for idx in sorted_indices[:10]:  # Check top 10 gradient candidates
        # Skip weak gradients
        if abs_grad[idx] < grad_threshold:
            continue

        # Candidate footer height
        cand_h = search_h - (idx + 1)

        # Size constraints: 20px minimum, 150px maximum (reasonable data bar range)
        if cand_h < 20 or cand_h > 150:
            continue

        # Check that content area above footer has different characteristics
        content_mean = np.mean(row_means[: idx + 1]) if idx > 0 else row_means[0]
        footer_mean = np.mean(row_means[idx + 1 :])

        # Footer should be distinctly different from content
        # (either much darker or much brighter)
        mean_diff = abs(content_mean - footer_mean) / max_val
        if mean_diff > 0.1:  # At least 10% intensity difference
            return cand_h

    return 0


def _find_bar_width(footer_img: FloatArray) -> int | None:
    """
    Find scale bar width (white rectangle) in footer region.

    Args:
        footer_img: Footer region of the image.

    Returns:
        Width of scale bar in pixels, or None if not found.
    """
    max_val = np.max(footer_img)
    if max_val <= 0:
        return None

    # Threshold for white bar
    thresh_val = max_val * 0.8
    binary = footer_img > thresh_val

    if binary.ndim == 3:
        binary = np.any(binary, axis=2)

    labels = measure.label(binary)
    props = measure.regionprops(labels)

    # Filter for bar-like shapes (width >> height)
    bars = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        height = maxr - minr
        # Aspect ratio > 3, width > 20px
        if height > 0 and width > 3 * height and width > 20:
            bars.append(width)

    if bars:
        return max(bars)  # Longest bar is likely the scale bar
    return None


def _ocr_scale(footer_img: FloatArray) -> tuple[float | None, str | None]:
    """
    Use OCR to extract scale information from footer.

    Args:
        footer_img: Footer region of the image.

    Returns:
        Tuple of (scale_value, units) or (None, None) if OCR fails.
    """
    # Check if tesseract is available
    if not shutil.which("tesseract"):
        return None, None

    try:
        import pytesseract  # noqa: PLC0415

        # Prepare image for OCR
        ocr_img = footer_img
        if ocr_img.dtype != np.uint8:
            ocr_img = ((ocr_img - ocr_img.min()) / (ocr_img.max() - ocr_img.min() + 1e-10) * 255).astype(np.uint8)

        text: str = str(pytesseract.image_to_string(ocr_img))

        # Parse text for "number unit" pattern
        match = re.search(r"(\d+(?:\.\d+)?)\s*([µu]m|nm|mm|m)", text)
        if match:
            val_str, unit_str = match.groups()
            value = float(val_str)
            units = _normalize_units(unit_str)
            return value, units

    except Exception as e:
        logger.debug("OCR failed: %s", e)

    return None, None


def parse_footer(image: FloatArray, metadata: SEMMetadata, check_ocr: bool = False) -> tuple[int, SEMMetadata]:
    """
    Detect footer, find scale bar, and validate/update metadata.

    Args:
        image: Input image array.
        metadata: Current metadata (may be updated).
        check_ocr: Whether to force OCR validation (default False).

    Returns:
        Tuple of (footer_height, updated_metadata).
    """
    footer_height = _detect_footer(image)

    if footer_height <= 0:
        return 0, metadata

    # Optimization: Skip OCR if metadata is valid and validation not requested
    if metadata.is_valid() and not check_ocr:
        return footer_height, metadata

    h = image.shape[0]
    footer_img = image[h - footer_height :, ...]

    # Find scale bar width
    bar_width_px = _find_bar_width(footer_img)
    if bar_width_px is None:
        return footer_height, metadata

    # Try OCR to get scale value and units
    ocr_value, ocr_units = _ocr_scale(footer_img)
    if ocr_value is None:
        return footer_height, metadata

    # Calculate scale from bar
    calc_scale = ocr_value / bar_width_px
    calc_units = ocr_units

    # Update metadata if OCR found valid data
    updated_metadata = SEMMetadata(
        pixel_scale=metadata.pixel_scale,
        units=metadata.units,
        original_filename=metadata.original_filename,
        source=metadata.source,
    )

    if not metadata.is_valid():
        # No existing metadata, use OCR result
        updated_metadata.pixel_scale = calc_scale
        updated_metadata.units = calc_units
        updated_metadata.source = "ocr"
        logger.info("Scale detected from OCR: %.4f %s/px", calc_scale, calc_units)
    else:
        # Compare with existing metadata
        units_match = (metadata.units == calc_units) or (metadata.units in {"µm", "um"} and calc_units in {"µm", "um"})
        if units_match and metadata.pixel_scale:
            diff = abs(calc_scale - metadata.pixel_scale) / metadata.pixel_scale
            if diff > 0.05:  # >5% difference
                logger.warning("Scale mismatch! Meta: %.4e, OCR: %.4e. Using OCR value.", metadata.pixel_scale, calc_scale)
                updated_metadata.pixel_scale = calc_scale
                updated_metadata.units = calc_units
                updated_metadata.source = "ocr"

    return footer_height, updated_metadata


# ==================================================================================================
# 4. Image Processing
# ==================================================================================================


def _crop_to_aspect_ratio(image: FloatArray, target_ratio: float) -> FloatArray:
    """
    Crop image to target aspect ratio (width/height), centering the crop.

    Args:
        image: Input image array (H, W) or (H, W, C).
        target_ratio: Target width/height ratio.

    Returns:
        Cropped image with the desired aspect ratio.
    """
    h, w = image.shape[:2]
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 0.01:
        # Already at target ratio
        return image

    if current_ratio > target_ratio:
        # Image is too wide, crop width
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        if image.ndim == 2:
            return image[:, offset : offset + new_w]
        return image[:, offset : offset + new_w, ...]
    else:
        # Image is too tall, crop height
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        if image.ndim == 2:
            return image[offset : offset + new_h, :]
        return image[offset : offset + new_h, :, ...]


def crop_image(
    image: FloatArray,
    params: ProcessingParams,
    metadata: SEMMetadata | None = None,
) -> tuple[FloatArray, int, SEMMetadata | None]:
    """
    Process the SEM image: Crop bottom metadata -> Rotate -> Crop edges -> Adjust aspect ratio.

    Args:
        image: Input image array.
        params: Processing parameters including:
            - crop_bottom: Pixels to crop from bottom (0 = auto-detect).
            - rotate_angle: Rotation angle in degrees.
            - crop_edge: Pixels to crop from all edges after rotation.
            - aspect_ratio: Target width/height ratio (None = no change).
                           Common values: 4/3 (1.33), 16/9 (1.78), 1.0 (square), 3/2 (1.5).
        metadata: Optional metadata (for scale validation).

    Returns:
        Tuple of (cropped_image, actual_crop_height, updated_metadata).
    """
    data = image.copy()
    updated_metadata = metadata

    # 1. Auto-detect footer if crop_bottom is 0
    if params.crop_bottom == 0 and metadata is not None:
        detected_height, updated_metadata = parse_footer(image, metadata)
        actual_crop = detected_height if detected_height > 0 else DEFAULT_CROP_BOTTOM
    else:
        actual_crop = params.crop_bottom if params.crop_bottom > 0 else DEFAULT_CROP_BOTTOM

    # 2. Crop Bottom
    if actual_crop > 0 and actual_crop < data.shape[0]:
        data = data[:-actual_crop, ...]

    # 3. Rotate
    if params.rotate_angle != 0:
        data = ndi.rotate(data, params.rotate_angle, axes=(0, 1), reshape=True, mode="constant", cval=0)

    # 4. Crop Edges (Post-rotation cleanup)
    if params.crop_edge > 0:
        h: int = data.shape[0]
        w: int = data.shape[1] if len(data.shape) > 1 else 1
        if 2 * params.crop_edge < h and 2 * params.crop_edge < w:
            data = data[params.crop_edge : -params.crop_edge, params.crop_edge : -params.crop_edge, ...]

    # 5. Adjust Aspect Ratio (NEW)
    if params.aspect_ratio is not None and params.aspect_ratio > 0:
        data = _crop_to_aspect_ratio(data, params.aspect_ratio)

    return data, actual_crop, updated_metadata


def enhance_contrast(image: FloatArray, params: ProcessingParams) -> FloatArray:
    """
    Automatically adjust image contrast to improve visibility.

    Args:
        image: Input image (grayscale).
        params: Processing parameters with contrast settings.

    Returns:
        Enhanced image.
    """
    if params.contrast_method == "none":
        return image

    if params.contrast_method == "stretch":
        # Optimization: Percentile stretching on subsampled image
        # Avoids full float copy and speeds up sort
        h, w = image.shape[:2]
        stride = max(1, min(h, w) // 500)
        v_min, v_max = np.percentile(image[::stride, ::stride], (params.p_low, params.p_high))
        return exposure.rescale_intensity(image, in_range=(float(v_min), float(v_max)))

    if params.contrast_method == "clahe":
        # CLAHE requires float normalization
        img_float = image.astype(float)
        img_range = img_float.max() - img_float.min()
        img_norm = (img_float - img_float.min()) / img_range if img_range > 0 else img_float
        return exposure.equalize_adapthist(img_norm, clip_limit=params.clip_limit)

    return image


# ==================================================================================================
# 5. Visualization
# ==================================================================================================


def add_scale_bar(
    ax: plt.Axes,
    config: ScaleBarConfig,
    pixel_scale: float,
) -> AnchoredSizeBar:
    """
    Add a scale bar to the axes.

    Args:
        ax: Matplotlib axes.
        config: Scale bar configuration.
        pixel_scale: Physical size per pixel.

    Returns:
        AnchoredSizeBar artist.
    """
    import matplotlib.font_manager as fm  # noqa: PLC0415

    size = config.size if config.size is not None else 1.0
    units = config.units if config.units is not None else "units"

    # Configure font
    fontprops = fm.FontProperties(size=config.font_size, weight=config.font_weight)

    # Calculate bar height (in pixels) based on image width
    # Note: ax.get_xlim() returns (0, width) for image axes
    xlim = ax.get_xlim()
    img_width = abs(xlim[1] - xlim[0])
    size_vertical = int(img_width * config.height_fraction)

    asb = AnchoredSizeBar(
        ax.transData,
        size / pixel_scale,
        f"{size} {units}",
        loc=config.location,
        pad=0.3,
        borderpad=0.5,
        sep=config.sep,
        frameon=config.frameon,
        color=config.color,
        size_vertical=size_vertical,
        label_top=True,
        fontproperties=fontprops,
    )

    if config.frameon:
        asb.patch.set_facecolor(config.frame_color)
        asb.patch.set_alpha(config.frame_alpha)
        asb.patch.set_edgecolor("none")

    if config.bbox_offset is not None:
        bbox = mtransforms.Bbox.from_bounds(*config.bbox_offset).transformed(ax.transAxes)
        asb.set_bbox_to_anchor(bbox)

    ax.add_artist(asb)
    return asb


def _auto_scale_bar(img_width_px: int, px_scale: float, units: str | None = None) -> tuple[float, str, float]:
    """
    Calculate appropriate scale bar size (~20% of image width).

    Auto-selects round numbers and converts units (nm ↔ µm) if needed.

    Args:
        img_width_px: Image width in pixels.
        px_scale: Physical size per pixel.
        units: Current units ('nm', 'µm', etc.).

    Returns:
        Tuple of (bar_size, display_units, converted_px_scale).
    """
    units = units or "µm"
    img_width_real = img_width_px * px_scale
    target = img_width_real * 0.2

    if target <= 0:
        return 10.0, units, px_scale

    # Unit conversion: prefer µm for large scales, nm for small
    if units == "nm" and target >= 1000:
        target /= 1000
        px_scale /= 1000
        units = "µm"
    elif units == "µm" and target < 0.1:
        target *= 1000
        px_scale *= 1000
        units = "nm"

    # Round to nice numbers: 1, 2, 5, 10, 20, 50, ...
    order = 10 ** math.floor(math.log10(target))
    best = order
    for n in [1, 2, 5]:
        candidate = n * order
        if candidate <= target * 1.5:
            best = candidate

    return max(best, order), units, px_scale


# ==================================================================================================
# 6. Main Analyzer Class
# ==================================================================================================


class SEMAnalyser:
    """
    Main class for SEM image analysis workflow.

    Encapsulates the full pipeline:
    1. Load image and extract metadata
    2. Preprocess (crop, rotate, enhance contrast)
    3. Create publication-ready figure with scale bar
    4. Save to multiple formats/resolutions

    Example:
        >>> analyser = SEMAnalyser("path/to/image.tif")
        >>> analyser.preprocess(rotate_angle=0, contrast_method="stretch")
        >>> analyser.plot(scale_bar_size=2, scale_bar_units="µm")
        >>> analyser.save(output_dir="output/", dpi_list=(300, 600))
    """

    def __init__(self, file_path: str | Path | None = None):
        """
        Initialize the SEM analyser.

        Args:
            file_path: Path to SEM image file. If None, must call load() later.
        """
        self._file_path: Path | None = None
        self._raw_image: FloatArray | None = None
        self._processed_image: FloatArray | None = None
        self._metadata: SEMMetadata = SEMMetadata()
        self._params: ProcessingParams = ProcessingParams()
        self._scalebar: ScaleBarConfig = ScaleBarConfig()
        self._figure: plt.Figure | None = None
        self._hyperspy_signal: Signal2D | None = None
        self._crop_h: int = 0

        if file_path is not None:
            self.load(file_path)

    # ------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------

    @property
    def raw_image(self) -> FloatArray | None:
        """Raw image data before processing."""
        return self._raw_image

    @property
    def processed_image(self) -> FloatArray | None:
        """Processed image data after cropping and enhancement."""
        return self._processed_image

    @property
    def metadata(self) -> SEMMetadata:
        """Image metadata including scale information."""
        return self._metadata

    @property
    def figure(self) -> plt.Figure | None:
        """Current matplotlib figure."""
        return self._figure

    @property
    def hyperspy_signal(self) -> Signal2D | None:
        """HyperSpy Signal2D for advanced processing."""
        return self._hyperspy_signal

    # ------------------------------------------------------------------------------------------
    # Core Methods
    # ------------------------------------------------------------------------------------------

    def load(self, file_path: str | Path) -> SEMAnalyser:
        """
        Load an SEM image from file.

        Args:
            file_path: Path to the SEM image file.

        Returns:
            Self for method chaining.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        self._file_path = Path(file_path)
        self._raw_image, self._metadata = read_data(self._file_path)

        logger.info(
            "Loaded: %s (shape=%s, scale=%.4g %s)",
            self._file_path.name,
            self._raw_image.shape,
            self._metadata.pixel_scale or 0,
            self._metadata.units or "unknown",
        )

        return self

    def set_scale(self, pixel_scale: float, units: str = "µm") -> SEMAnalyser:
        """
        Manually set the pixel scale.

        Args:
            pixel_scale: Physical size per pixel.
            units: Physical units.

        Returns:
            Self for method chaining.
        """
        self._metadata.pixel_scale = pixel_scale
        self._metadata.units = _normalize_units(units)
        self._metadata.source = "manual"
        return self

    def preprocess(  # noqa: PLR0913, PLR0917
        self,
        crop_bottom: int = 0,
        rotate_angle: float = 0.0,
        crop_edge: int = 0,
        aspect_ratio: float | None = None,
        contrast_method: Literal["stretch", "clahe", "none"] = "stretch",
        p_low: float = 2.0,
        p_high: float = 98.0,
        clip_limit: float = 0.01,
        check_ocr: bool = False,
    ) -> SEMAnalyser:
        """
        Preprocess the SEM image.

        Args:
            crop_bottom: Pixels to crop from bottom (0 = auto-detect).
            rotate_angle: Rotation angle in degrees.
            crop_edge: Pixels to crop from all edges after rotation.
            aspect_ratio: Target width/height ratio (None = no change).
                         Common values: 4/3 (1.33), 16/9 (1.78), 1.0 (square), 3/2 (1.5).
            contrast_method: Contrast enhancement method.
            p_low: Lower percentile for stretching.
            p_high: Upper percentile for stretching.
            clip_limit: CLAHE clipping limit.
            check_ocr: Whether to verify scale with OCR (slow).

        Returns:
            Self for method chaining.
        """
        if self._raw_image is None:
            raise ValueError("No image loaded. Call load() first.")

        self._params = ProcessingParams(
            crop_bottom=crop_bottom,
            rotate_angle=rotate_angle,
            crop_edge=crop_edge,
            aspect_ratio=aspect_ratio,
            contrast_method=contrast_method,
            p_low=p_low,
            p_high=p_high,
            clip_limit=clip_limit,
            check_ocr=check_ocr,
        )

        # Crop and rotate
        cropped, self._crop_h, updated_metadata = crop_image(
            self._raw_image,
            self._params,
            self._metadata,
        )
        if updated_metadata is not None:
            self._metadata = updated_metadata

        # Enhance contrast
        self._processed_image = enhance_contrast(cropped, self._params)

        logger.info(
            "Preprocessed: crop_bottom=%d, rotate=%.1f°, contrast=%s",
            self._crop_h,
            rotate_angle,
            contrast_method,
        )

        return self

    def plot(  # noqa: PLR0913, PLR0917
        self,
        scale_bar_size: float | None = None,
        scale_bar_units: str | None = None,
        scale_bar_color: str = "white",
        scale_bar_location: str = "lower left",
        scale_bar_sep: float = 2.0,
        scale_bar_font_size: int | None = None,
        scale_bar_font_weight: str = "bold",
        scale_bar_height_fraction: float = 0.015,
        scale_bar_frameon: bool = False,
        scale_bar_frame_alpha: float = 0.5,
        scale_bar_frame_color: str = "black",
        bbox_offset: tuple[float, float, float, float] | None = None,
        figsize: tuple[float, float] = DEFAULT_FIGURE_SIZE,
        cmap: str = "gray",
        show: bool = True,
    ) -> SEMAnalyser:
        """
        Create a publication-ready figure with scale bar.

        Args:
            scale_bar_size: Physical size of scale bar (auto-calculate if None).
            scale_bar_units: Scale bar units (use metadata if None).
            scale_bar_color: Scale bar text/bar color.
            scale_bar_location: Anchor location.
            scale_bar_sep: Separation between bar and label (points).
            scale_bar_font_size: Font size (None = default).
            scale_bar_font_weight: Font weight ('bold', 'normal').
            scale_bar_height_fraction: Bar height as fraction of image width.
            scale_bar_frameon: Whether to draw background box.
            scale_bar_frame_alpha: Background box opacity.
            scale_bar_frame_color: Background box color.
            bbox_offset: Custom positioning (x, y, w, h) in axes coordinates.
            figsize: Figure size in inches.
            cmap: Colormap for grayscale display.
            show: Whether to display the figure.

        Returns:
            Self for method chaining.
        """
        if self._processed_image is None:
            # Auto-preprocess if not done
            self.preprocess()

        if self._processed_image is None:
            raise ValueError("No processed image available.")

        # Determine scale
        pixel_scale = self._metadata.pixel_scale or 1.0
        original_units = self._metadata.units or "pixels"
        units = scale_bar_units or original_units

        # Calculate scale bar size if not provided (with auto unit conversion)
        if scale_bar_size is None:
            auto_size, auto_units, pixel_scale = _auto_scale_bar(self._processed_image.shape[1], pixel_scale, units)
            scale_bar_size = auto_size
            units = auto_units
        # Manual size specified: convert pixel_scale if units differ from original
        elif units != original_units:
            if original_units == "nm" and units == "µm":
                pixel_scale /= 1000.0  # nm -> µm
            elif original_units == "µm" and units == "nm":
                pixel_scale *= 1000.0  # µm -> nm

        # Format units for LaTeX display
        if units == "µm":
            units_display = r"$\mathrm{\mu m}$"
        elif units == "nm":
            units_display = r"$\mathrm{nm}$"
        else:
            units_display = units

        # Configure scale bar
        self._scalebar = ScaleBarConfig(
            size=scale_bar_size,
            units=units_display,
            color=scale_bar_color,
            location=scale_bar_location,
            sep=int(scale_bar_sep),
            bbox_offset=bbox_offset,
            font_size=scale_bar_font_size,
            font_weight=scale_bar_font_weight,
            height_fraction=scale_bar_height_fraction,
            frameon=scale_bar_frameon,
            frame_alpha=scale_bar_frame_alpha,
            frame_color=scale_bar_frame_color,
        )

        # Create figure
        # Note: Do not close all figures; create a new one for this instance.
        fig = plt.figure(figsize=figsize)

        # Use add_axes to fill the entire figure (0,0 to 1,1) for the image
        # This avoids whitespace issues and complex gridspec/subfigure usage
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))

        # Display image
        ax.imshow(self._processed_image, cmap=cmap, interpolation="nearest")

        # Add scale bar
        add_scale_bar(ax, self._scalebar, pixel_scale)

        # Style
        ax.set_axis_off()
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        self._figure = fig

        if show:
            plt.gcf().set_facecolor("white")
            plt.show()

        return self

    def save(  # noqa: PLR0913, PLR0917
        self,
        output_dir: str | Path | None = None,
        filename_stem: str | None = None,
        dpi_list: tuple[int, ...] = DEFAULT_DPI_LIST,
        formats: tuple[str, ...] = DEFAULT_FORMATS,
        save_hyperspy: bool = True,
        save_xarray: bool = True,
    ) -> SEMAnalyser:
        """
        Save the figure and optionally the HyperSpy signal and Xarray data.

        Args:
            output_dir: Output directory (default: OUTPUT_ROOT/SEM_Processed).
            filename_stem: Base filename (default: original filename stem).
            dpi_list: DPI values for saving.
            formats: File formats ('tif', 'png', 'pdf', etc.).
            save_hyperspy: Whether to save HyperSpy signal (.hspy).
            save_xarray: Whether to save Xarray DataArray (.nc).

        Returns:
            Self for method chaining.
        """
        if self._figure is None:
            self.plot(show=False)

        if self._figure is None:
            raise ValueError("No figure to save. Call plot() first.")

        # Determine output path
        if output_dir is None:
            output_dir = OUTPUT_ROOT / "SEM_Processed"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename
        if filename_stem is None:
            filename_stem = self._file_path.stem if self._file_path else "sem_image"

        # Save figures
        for dpi in dpi_list:
            for fmt in formats:
                save_path = output_dir / f"{dpi}_{filename_stem}.{fmt}"

                pil_kwargs = {}
                if fmt in {"tif", "tiff"}:
                    pil_kwargs = {"compression": "tiff_lzw"}

                self._figure.savefig(
                    save_path,
                    pad_inches=0.01,
                    bbox_inches="tight",
                    dpi=dpi,
                    transparent=False,
                    pil_kwargs=pil_kwargs,
                )
                logger.info("Saved: %s", save_path)

        # Create and save HyperSpy signal
        if save_hyperspy and self._processed_image is not None:
            self._create_hyperspy_signal()
            if self._hyperspy_signal is not None:
                hspy_path = output_dir / f"{filename_stem}.hspy"
                self._hyperspy_signal.save(str(hspy_path), overwrite=True)
                logger.info("Saved: %s", hspy_path)

        # Create and save Xarray DataArray
        if save_xarray and self._processed_image is not None:
            try:
                da = self._create_xarray_object()
                xr_path = output_dir / f"{filename_stem}.nc"
                da.to_netcdf(xr_path)
                logger.info("Saved: %s", xr_path)
            except Exception as e:
                logger.warning("Failed to save xarray: %s", e)

        return self

    def _create_xarray_object(self) -> xr.DataArray:
        """Create xarray DataArray with metadata."""
        if self._processed_image is None:
            raise ValueError("No processed image available.")

        img = self._processed_image
        h, w = img.shape
        scale = self._metadata.pixel_scale or 1.0
        units = self._metadata.units or "px"

        # Create coordinates (physical units)
        y = np.arange(h) * scale
        x = np.arange(w) * scale

        # Create DataArray
        da = xr.DataArray(
            img,
            coords={"y": y, "x": x},
            dims=("y", "x"),
            name="sem_image",
            attrs={
                "units": units,
                "pixel_scale": scale,
                "original_filename": self._metadata.original_filename or "",
                "source": self._metadata.source,
                "crop_bottom": self._params.crop_bottom,
                "rotate_angle": self._params.rotate_angle,
                "contrast_method": self._params.contrast_method,
            },
        )
        da.y.attrs = {"units": units}
        da.x.attrs = {"units": units}

        return da

    def _create_hyperspy_signal(self) -> None:
        """Create HyperSpy Signal2D with metadata."""
        if self._processed_image is None:
            return

        s = hs.signals.Signal2D(self._processed_image)

        # Add metadata
        pixel_scale = self._metadata.pixel_scale or 1.0
        units = self._metadata.units or "pixels"

        metadata_dict = {
            "General": {
                "original_filename": self._metadata.original_filename,
                "title": self._file_path.stem if self._file_path else "SEM Image",
            },
            "SemProcessing": {
                "crop_bottom": self._params.crop_bottom,
                "footer_height_detected": self._crop_h,
                "rotate_angle": self._params.rotate_angle,
                "crop_edge": self._params.crop_edge,
                "scale_factor": pixel_scale,
                "scale_units": units,
                "scale_bar_size": self._scalebar.size,
                "contrast_method": self._params.contrast_method,
            },
        }
        s.metadata.add_dictionary(metadata_dict)

        # Set axis scale
        for i, name in enumerate(["y", "x"]):
            s.axes_manager[i].name = name
            s.axes_manager[i].scale = pixel_scale
            s.axes_manager[i].units = units

        self._hyperspy_signal = s

    def get_result(self) -> SEMResult:
        """
        Get the complete analysis result.

        Returns:
            SEMResult containing all processed data.
        """
        if self._processed_image is None:
            self.preprocess()

        # Ensure we have an image to return
        img = self._processed_image if self._processed_image is not None else self._raw_image
        if img is None:
            raise ValueError("No image available. Call load() first.")

        return SEMResult(
            image=img,
            metadata=self._metadata,
            processing_params=self._params,
            figure=self._figure,
            hyperspy_signal=self._hyperspy_signal,
        )


# ==================================================================================================
# 7. CLI
# ==================================================================================================


def main() -> None:
    """Command-line interface for batch SEM processing."""
    # Apply matplotlib style
    setup()

    parser = argparse.ArgumentParser(description="Batch process SEM images (Crop, Scale Bar, Save).")
    parser.add_argument("input_dir", nargs="?", type=str, help="Input directory containing .tif files")
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory (default: OUTPUT_ROOT/SEM_Processed)")
    parser.add_argument("--scale_bar_size", "-s", type=float, help="Scale bar size (e.g., 2, 10, 500)")
    parser.add_argument("--scale_bar_units", "-u", type=str, help="Scale bar units (e.g., 'µm', 'nm')")
    parser.add_argument("--crop_bottom", "-c", type=int, default=0, help="Pixels to crop from bottom (0 = auto-detect)")
    parser.add_argument("--contrast", type=str, default="stretch", choices=["stretch", "clahe", "none"], help="Contrast enhancement method")

    args = parser.parse_args()

    # Determine Input Directory
    if args.input_dir:
        input_path = Path(args.input_dir)
    else:
        input_path = Path.cwd()
        print(f"No input directory provided. Using current directory: {input_path}")

    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        return

    # Determine Output Directory
    output_path = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / "SEM_Processed" / input_path.name

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Find TIF files
    tif_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))

    if not tif_files:
        print("No .tif or .tiff files found.")
        return

    print(f"Found {len(tif_files)} images.")

    # Process each file
    for file_path in tif_files:
        print(f"Processing: {file_path.name}...")
        try:
            analyser = SEMAnalyser(file_path)
            analyser.preprocess(
                crop_bottom=args.crop_bottom,
                contrast_method=args.contrast,
            )
            analyser.plot(
                scale_bar_size=args.scale_bar_size,
                scale_bar_units=args.scale_bar_units,
                show=False,
            )
            analyser.save(output_dir=output_path)
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    print("Batch processing complete.")


if __name__ == "__main__":
    main()
