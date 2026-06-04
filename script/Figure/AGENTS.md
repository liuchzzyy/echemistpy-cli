# PROJECT KNOWLEDGE BASE: Figure/

**Generated:** 2026-01-12
**Context**: Shared tooling and publication outputs for Materials Science research.

## OVERVIEW
The `Figure/` directory serves as the centralized visualization synthesis and configuration hub for the entire project. it contains shared style definitions, path configurations, and paper-specific manuscript figures.

## STRUCTURE
```
Figure/
├── config.py           # Configuration Hub: Paths, styles, and setup logic
├── colors.py           # Paul Tol color schemes (color-blind safe)
├── liuchzzyy.mplstyle  # Global Matplotlib style sheet (Arial fonts, etc.)
├── PaperUno/           # Manuscript-specific figure notebooks (e.g., PaperUno_Figure_MS.ipynb)
├── PaperDos/           # Manuscript-specific figure notebooks
└── PaperTres/          # Manuscript-specific figure notebooks
```

## WHERE TO LOOK
| Component | Purpose | Notes |
|-----------|---------|-------|
| **config.py** | Global paths & styles | Entry point for all notebooks; defines `DATA_ROOT`, `OUTPUT_ROOT`. |
| **colors.py** | Colorblind-safe palettes | Implementation of Paul Tol's vibrant, muted, and rainbow schemes. |
| **Paper*/ **  | Final Manuscripts | Notebooks that combine analysis results into publication-ready panels. |

## CONVENTIONS
- **Initialization**: Every figure notebook MUST import and run `setup()` from `Figure.config`.
  ```python
  import sys
  from pathlib import Path
  sys.path.append(str(Path.cwd().parent.parent)) # Add Repo Root
  from Figure.config import setup, DATA_ROOT, OUTPUT_ROOT
  colors = setup()
  ```
- **Pathing**: Use `DATA_ROOT` for inputs (OneDrive) and `OUTPUT_ROOT` for saving figures. NEVER hardcode absolute local paths.
- **Styling**: Prefer `liuchzzyy.mplstyle` settings. Use the `colors` list returned by `setup()` for consistent plot colors.
- **Revision Control**: Keep track of revisions within the paper folders (e.g., `.R1.ipynb` for first revision).

## ANTI-PATTERNS
- **Hardcoded Styles**: Do not manually set `plt.rcParams` in notebooks; update `config.py` or the `.mplstyle` if global changes are needed.
- **One-off Config**: Avoid modifying `config.py` for a single plot. Use local overrides in the notebook only if absolutely necessary.
- **Large Assets**: Do not commit high-resolution PDF/TIFF outputs to the repo. Save them to `OUTPUT_ROOT`.
