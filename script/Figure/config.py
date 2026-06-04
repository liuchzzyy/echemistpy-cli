from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from .colors import tol_cmap, tol_cset

# ==========================================
# Path Configuration
# ==========================================

# Repository Root (assuming this file is in Figure/)
REPO_ROOT = Path(__file__).parent.parent

# Data Root
# User can set ICMAB_DATA_PATH env var, or modify this file
DATA_ROOT = Path(r"D:\CHENG\OneDrive - UAB\ICMAB-Data")

# Output Root
OUTPUT_ROOT = Path(r"E:\Desktop\Figure")

# ==========================================
# Style Configuration
# ==========================================


def setup():
    """
    Configures matplotlib style, xarray options, and registers custom colormaps.
    Returns the default color list (Paul Tol 'vibrant').
    """
    # Style
    style_file = Path(__file__).parent / "liuchzzyy.mplstyle"
    if style_file.exists():
        plt.style.use(str(style_file))

    # Xarray options
    xr.set_options(
        cmap_sequential="viridis",
        cmap_divergent="viridis",
        display_width=150,
    )

    # Register Custom Colormaps
    if "sunset" not in plt.colormaps():
        cmap = tol_cmap("sunset")
        if isinstance(cmap, LinearSegmentedColormap):
            plt.colormaps.register(cmap)

    if "rainbow_PuRd" not in plt.colormaps():
        cmap = tol_cmap("rainbow_PuRd")
        if isinstance(cmap, LinearSegmentedColormap):
            plt.colormaps.register(cmap)

    # Font Settings (ensure consistency)
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Arial"
    mpl.rcParams["mathtext.it"] = "Arial:italic"
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.sf"] = "Arial"
    mpl.rcParams["mathtext.tt"] = "Arial"
    mpl.rcParams["mathtext.cal"] = "Arial"
    mpl.rcParams["mathtext.default"] = "regular"

    # Return default color set
    colors = tol_cset("vibrant")
    if colors is not None:
        return list(colors)
    else:
        # Fallback to Paul Tol vibrant scheme
        return ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"]
