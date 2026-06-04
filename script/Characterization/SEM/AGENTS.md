# SEM Analysis Framework Knowledge Base

**Last Updated:** 2026-01-18
**Context**: Scanning Electron Microscopy (SEM) Image Processing & Publication Figure Generation.

## OVERVIEW

This directory contains tools for processing SEM images, specifically focused on battery materials characterization. The core logic is encapsulated in `sem_utils.py`.

**Supported Instruments:**
- **Zeiss** (CZ_SEM metadata, Tag 34118)
- **FEI/Thermo Fisher** (FEI metadata, Tag 34682)
- Generic TIFF with resolution tags

## CORE MODULE: `sem_utils.py`

### 1. Constants & Configuration

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_CROP_BOTTOM` | 60 | Default pixels to crop from footer |
| `DEFAULT_CONTRAST_METHOD` | "stretch" | Default contrast enhancement |
| `DEFAULT_FIGURE_SIZE` | (3.3, 2.5) | Figure size in inches |
| `DEFAULT_DPI_LIST` | (300, 600) | Output resolutions |

**Unit Aliases** (`UNIT_ALIASES`):
| Input | Normalized |
|-------|------------|
| "um", "micron", "microns" | "µm" |
| "nm" | "nm" |
| "mm" | "mm" |

### 2. Dataclasses

#### `SEMMetadata`
```python
@dataclass
class SEMMetadata:
    pixel_scale: float | None  # Physical size per pixel
    units: str | None          # 'µm', 'nm', etc.
    original_filename: str | None
    source: Literal["zeiss", "fei", "tiff_standard", "ocr", "manual", "unknown"]
```

#### `ProcessingParams`
```python
@dataclass
class ProcessingParams:
    crop_bottom: int = 0       # 0 = auto-detect
    rotate_angle: float = 0.0
    crop_edge: int = 0
    aspect_ratio: float | None = None  # Target width/height ratio (None = no change)
    contrast_method: Literal["stretch", "clahe", "none"] = "stretch"
    p_low: float = 2.0         # Lower percentile for stretching
    p_high: float = 98.0       # Upper percentile for stretching
    clip_limit: float = 0.01   # CLAHE clip limit
    check_ocr: bool = False    # Whether to verify scale with OCR
```

**Common Aspect Ratios:**
| Ratio | Value | Use Case |
|-------|-------|----------|
| 4:3 | 1.333 | Standard presentation |
| 16:9 | 1.778 | Widescreen |
| 3:2 | 1.5 | Classic photo |
| 1:1 | 1.0 | Square |

#### `ScaleBarConfig`
```python
@dataclass
class ScaleBarConfig:
    size: float | None = None  # Auto-calculate if None
    units: str | None = None
    color: str = "white"
    location: str = "lower left"
    sep: int = 2
    bbox_offset: tuple[float, float, float, float] | None = None
```

#### `SEMResult`
```python
@dataclass
class SEMResult:
    image: FloatArray
    metadata: SEMMetadata
    processing_params: ProcessingParams
    figure: plt.Figure | None = None
    hyperspy_signal: Any = None
```

### 3. Metadata Extraction

| Function | Purpose |
|----------|---------|
| `read_data(file_path)` | Load image + metadata via HyperSpy (Zeiss, FEI auto-detected) |

**HyperSpy-Based Loading**:
- HyperSpy automatically extracts calibrated pixel scales from Zeiss (CZ_SEM) and FEI metadata
- Returns scales in **meters** → `read_data()` converts to nm or µm for display

**Metadata Sources (Priority Order)**:
1. **Zeiss CZ_SEM** (Tag 34118): `ap_image_pixel_size` → nm or µm
2. **FEI/Thermo Fisher** (Tag 34682): `EScan.PixelWidth` (meters) → auto-convert
3. **OCR Fallback**: Reads scale bar text from footer (optional, slow)

### 4. Footer Detection & Scale Validation

| Function | Purpose |
|----------|---------|
| `_detect_footer(image)` | Auto-detect metadata footer height (Zeiss/FEI styles) |
| `_find_bar_width(footer_img)` | Find scale bar width (white rectangle) |
| `_ocr_scale(footer_img)` | OCR to extract "10 µm" text |
| `parse_footer(image, metadata)` | Validate metadata vs OCR |

**Algorithm**:
1. Analyze bottom 15% of image for intensity transition
2. Detect significant gradient (>5% of max value)
3. Validate footer height (20-150 px range)
4. Optionally OCR to read scale bar text
5. Cross-validate with metadata (warn if >5% mismatch)

### 5. Image Processing

| Function | Purpose |
|----------|---------|
| `crop_image(image, params, metadata)` | Crop footer → Rotate → Crop edges → Aspect ratio |
| `enhance_contrast(image, params)` | Enhance contrast (stretch/CLAHE) |
| `_crop_to_aspect_ratio(image, target_ratio)` | Center-crop to target width/height ratio |

**Contrast Methods**:
- `"stretch"`: Percentile-based intensity rescaling (p2-p98 by default)
- `"clahe"`: Contrast Limited Adaptive Histogram Equalization
- `"none"`: No enhancement

### 6. Visualization

| Function | Purpose |
|----------|---------|
| `add_scale_bar(ax, config, pixel_scale)` | Add AnchoredSizeBar to axes |
| `_auto_scale_bar(img_width_px, px_scale, units)` | Auto-calculate ~20% width with unit conversion |

**Auto Unit Conversion**:
- If scale bar would be >= 1000 nm → convert to µm
- If scale bar would be < 0.1 µm → convert to nm
- Rounds to nice numbers: 1, 2, 5, 10, 20, 50, 100, etc.

**Manual Unit Override**:
- When `scale_bar_units` differs from metadata units, `pixel_scale` is auto-converted

### 7. Main Class: `SEMAnalyser`

**Workflow Methods**:
| Method | Description |
|--------|-------------|
| `load(file_path)` | Load SEM image and extract metadata |
| `set_scale(pixel_scale, units)` | Manually override scale |
| `preprocess(...)` | Crop, rotate, enhance contrast |
| `plot(...)` | Create figure with scale bar |
| `save(output_dir, ...)` | Save to multiple DPI/formats |
| `get_result()` | Return `SEMResult` dataclass |

**Example**:
```python
from Characterization.SEM.sem_utils import SEMAnalyser

# Method chaining workflow (auto-detect everything)
analyser = SEMAnalyser("image.tif")
analyser.preprocess(crop_bottom=0, contrast_method="stretch")
analyser.plot(scale_bar_size=None)  # Auto-calculate size & units
analyser.save(output_dir="output/", dpi_list=(300, 600))

# Step-by-step with manual scale
analyser = SEMAnalyser()
analyser.load("image.tif")
analyser.set_scale(0.0145, "µm")  # Manual scale if needed
analyser.preprocess()
analyser.plot(scale_bar_size=2, scale_bar_units="µm", show=True)
```

### 8. CLI

```bash
python sem_utils.py input_dir/ --output_dir output/ --scale_bar_size 10 --contrast stretch
```

## USAGE WORKFLOW

```python
import sys
from pathlib import Path

# Setup path (adjust depth as needed)
sys.path.append(str(Path.cwd().parent.parent))
from Characterization.SEM.sem_utils import SEMAnalyser

# 1. Load (auto-detect Zeiss/FEI metadata)
analyser = SEMAnalyser("path/to/SEM.tif")

# 2. Check metadata
print(f"Scale: {analyser.metadata.pixel_scale} {analyser.metadata.units}")
print(f"Source: {analyser.metadata.source}")  # 'zeiss', 'fei', etc.

# 3. Preprocess (auto-detect footer, enhance contrast, optional aspect ratio)
analyser.preprocess(
    crop_bottom=0,
    contrast_method="stretch",
    aspect_ratio=16/9,  # Optional: crop to 16:9 widescreen
)

# 4. Plot with auto scale bar (size & units auto-calculated)
analyser.plot(
    scale_bar_size=None,        # Auto-calculate
    scale_bar_color="white",
    show=True,
)

# 5. Save
analyser.save(
    output_dir="output/",
    dpi_list=(300, 600),
    formats=("tif",),
    save_hyperspy=True,
    save_xarray=True,
)
```

## KEY DEPENDENCIES

| Package | Purpose |
|---------|---------|
| `hyperspy` | Signal2D, metadata extraction & preservation |
| `skimage` | Contrast enhancement, region detection |
| `scipy.ndimage` | Image rotation |
| `matplotlib` | Figure generation |
| `xarray` | NetCDF export with coordinates |
| `pytesseract` | OCR (optional, requires Tesseract) |

## CONVENTIONS & ANTI-PATTERNS

### Conventions
- **Scale Units**: Always normalize to `µm` or `nm` via `UNIT_ALIASES`
- **Footer Detection**: `crop_bottom=0` triggers auto-detection
- **Figure Config**: Use `Figure/config.py` for consistent styling
- **Auto Scale Bar**: Pass `scale_bar_size=None` for automatic calculation

### Anti-Patterns
- **DO NOT** hardcode absolute paths (use `DATA_ROOT`, `OUTPUT_ROOT`)
- **DO NOT** manually set matplotlib params (use `setup()`)
- **DO NOT** assume metadata is always available (check `metadata.is_valid()`)

## FILE STRUCTURE

```
Characterization/SEM/
├── sem_utils.py      # Core module with SEMAnalyser class (~1280 lines)
├── SEM.ipynb         # Demo notebook with Zeiss/FEI examples
├── SEM.tif           # Example Zeiss image (241.4 nm/px)
├── SEM2.tif          # Example FEI image (58.3 nm/px)
└── AGENTS.md         # This knowledge base
```

## TEST RESULTS

| File | Instrument | Detected Scale | Source |
|------|------------|----------------|--------|
| SEM.tif | Zeiss | 241.4 nm/px | zeiss (CZ_SEM) |
| SEM2.tif | FEI Quanta | 58.28 nm/px | fei (EScan) |
