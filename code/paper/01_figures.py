"""Paper figures from the ward analysis table: Figure 1 (prescribing), Figure 2 (greenery) and the ward maps in Methods.

Inputs : data/data_analysis_wards7s.csv, London 2018 ward shapefile (data/raw_data/geo_wards)
Outputs: results/paper/figures/: box plots of prescribing (Fig. 1) and of all greenery measures (Fig. 2),
         one ward map per outcome and per greenery measure (standardised scores)
Run    : python 01_figures.py
Notes  : colours from palette.py (teal ours, blue widely used, magenta prescribing).
"""
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from palette import BLUE, INK, MAGENTA, NODATA, SEQ, family, steps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "paper" / "figures"
SHP = REPO / "data/raw_data/geo_wards/London_Ward.shp"

OUTCOMES = ["diabetes", "hypertension", "asthma", "opioids", "total"]
GREENERY = {  # file stem: (column, map title)
    "on_road_aerial": ("SAT_on_road_share_u", "on-road, aerial"),
    "on_road_aerial_weighted": ("SAT_on_road_share_w", "on-road, aerial, weighted"),
    "on_road_street_view": ("GSV_on_road_u", "on-road, street view"),
    "on_road_street_view_weighted": ("GSV_on_road_w", "on-road, street view, weighted"),
    "off_road": ("SAT_off_road", "off-road"),
    "NDVI": ("SAT_NDVI", "NDVI"),
    "WHO": ("WHO", "WHO"),
    "ESA_WHO": ("ESA_WHO_exposure", "ESA WHO"),
    "NE": ("Natural_England1", "NE"),
}
BOX = dict(patch_artist=True, showfliers=False, widths=0.6,
           boxprops=dict(linewidth=1.5, color=INK), whiskerprops=dict(linewidth=1.2, color=INK),
           capprops=dict(linewidth=1.5, color=INK), medianprops=dict(color="white", linewidth=2.5))


def style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=15)
    ax.spines[["top"]].set_visible(False)
    ax.tick_params(labelsize=13)


def prescriptions_figure(wd: pd.DataFrame, path: Path) -> None:
    """Fig. 1: the four conditions on the left axis; total (about 20x larger) in its own panel with its own axis."""
    four = OUTCOMES[:4]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 5.3), gridspec_kw=dict(width_ratios=[4, 1.25], wspace=0.08))
    for ax, cols in ((a, four), (b, ["total"])):
        bx = ax.boxplot([wd[f"{o} prescriptions"].dropna() for o in cols], tick_labels=cols, **BOX)
        for patch in bx["boxes"]:
            patch.set_facecolor(MAGENTA)
    style(a, "yearly prescription quantity per patient")
    for ax, letter in ((a, "a)"), (b, "b)")):
        ax.set_title(letter, loc="left", fontsize=15, fontweight="bold")
    a.spines[["right"]].set_visible(False)
    b.yaxis.tick_right()
    b.spines[["left", "top"]].set_visible(False)
    b.tick_params(labelsize=13)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def greenery_figure(wd: pd.DataFrame, path: Path) -> None:
    """Fig. 2: all greenery measures on one 0-100% scale; left axis = share green, right axis = share of people with access."""
    ours = ["on_road_aerial", "on_road_street_view", "off_road", "NDVI"]
    access = ["ESA_WHO", "WHO", "NE"]
    labels = ["on-road\naerial", "on-road\nstreet view", "off-road", "NDVI", "ESA WHO", "WHO", "NE"]
    colours = [steps("teal", 5)(i) for i in (3, 2, 1)] + [BLUE] + [steps("blue", 5)(i) for i in (3, 2, 1)]
    fig, ax = plt.subplots(figsize=(10, 5.3))
    bx = ax.boxplot([wd[GREENERY[k][0]].dropna() * 100 for k in ours + access], tick_labels=labels, **BOX)
    for patch, c in zip(bx["boxes"], colours):
        patch.set_facecolor(c)
    ax.axvline(4.5, color="#9a9994", linewidth=1, linestyle=(0, (4, 3)))
    style(ax, "greenery percentage")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    right = ax.twinx()   # same 0-100% scale, second label only
    right.set_ylim(ax.get_ylim())
    right.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    right.set_ylabel("percentage of people with access", fontsize=15)
    right.tick_params(labelsize=13)
    right.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def ward_map(gdf: gpd.GeoDataFrame, col: str, title: str, cmap, path: Path) -> None:
    z = (gdf[col] - gdf[col].mean()) / gdf[col].std()
    fig, ax = plt.subplots(figsize=(7, 5))
    gdf.assign(z=z).plot("z", ax=ax, cmap=cmap, legend=True, linewidth=0,
                         missing_kwds=dict(color=NODATA), legend_kwds=dict(shrink=0.45))
    ax.set_title(title, fontsize=22)
    ax.set_axis_off()
    fig.axes[-1].tick_params(labelsize=18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    wd = pd.read_csv(REPO / "data" / "data_analysis_wards7s.csv")

    prescriptions_figure(wd, OUT / "plot_distribution_prescriptions.pdf")
    greenery_figure(wd, OUT / "plot_distribution_greenery.png")

    gdf = gpd.read_file(SHP).to_crs(27700)
    gdf = gdf.merge(wd.rename(columns={"ward_code": "GSS_CODE"}), on="GSS_CODE", how="left")
    for o in OUTCOMES:
        ward_map(gdf, f"{o} prescriptions", o, SEQ["magenta"], OUT / f"map_{o}.pdf")
    for stem, (col, title) in GREENERY.items():
        ward_map(gdf, col, title, SEQ[family(col)], OUT / f"map_{stem}.pdf")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
