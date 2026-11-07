"""One colour scheme for every figure and table of the paper and SI.
Teal = our greenery measures, blue = official greenery measures (NDVI, WHO, ESA WHO, NE), magenta = prescribing.
Signs (tables, correlation matrices, MGWR): blue = negative, red = positive, light grey midpoint.
Checked with the dataviz palette validator: the three hues pass lightness, chroma and normal-vision separation.
"""
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

TEAL, BLUE, MAGENTA = "#1a9479", "#4166b5", "#a84886"
NEG, POS, MID = "#4166b5", "#c4453f", "#f2f1ee"
INK, NODATA = "#2b2b2b", "#e4e3dd"
OFFICIAL = {"SAT_NDVI", "WHO", "ESA_WHO_exposure", "Natural_England1"}

# light -> dark, used for maps (continuous) and septile classes (7 steps)
_RAMPS = {"teal": ["#eef7f3", TEAL, "#0b4a3d"], "blue": ["#eef2f9", BLUE, "#1b2f5e"],
          "magenta": ["#f8edf3", MAGENTA, "#561e45"]}
SEQ = {k: LinearSegmentedColormap.from_list(k, v) for k, v in _RAMPS.items()}
DIV = LinearSegmentedColormap.from_list("bluered", ["#3a5590", NEG, "#c3cfe9", MID, "#efc9c4", POS, "#9b3a35"])
CELL_NEG, CELL_POS = "DCE4F3", "F5DCD9"  # LaTeX table shading (HTML hex), defined as negcell / poscell


def family(measure: str) -> str:
    """Ramp name for a greenery measure (model name or analysis column)."""
    return "blue" if measure.split("__")[0] in OFFICIAL else "teal"


def steps(name: str, n: int = 7) -> ListedColormap:
    """n discrete classes of a ramp, skipping the near-white and the darkest ends."""
    cm = SEQ[name]
    return ListedColormap([cm(0.12 + 0.72 * i / (n - 1)) for i in range(n)])
