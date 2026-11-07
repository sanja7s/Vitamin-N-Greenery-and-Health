"""Multiscale GWR (MGWR) of ward diabetes prescribing on weighted street view greenery plus the four controls (SI Figure S5).
Inputs : data/data_analysis_wards7s.csv, London 2018 ward boundaries
Outputs: results/models/mgwr_diabetes_wards.csv (local coefficient, t, significance per ward) and _info.json (bandwidths),
         results/paper/figures/mgwr_diabetes_GSV_on_road_w.pdf (map of the local greenery effect)
Run    : python 03_mgwr_diabetes.py [--draw]   (about 5 minutes; --draw only redraws the map from the saved results)
Notes  : y and X standardised; local effects are also given as % of mean diabetes prescribing per 1 SD of greenery,
         to compare with double ML. Significance uses mgwr's critical t, adjusted for multiple testing (95 %).
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from mgwr.gwr import MGWR  # noqa: E402
from mgwr.sel_bw import Sel_BW  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from causal import CONFOUNDERS, rename_columns  # noqa: E402

sys.path.insert(0, str(HERE.parent / "paper"))
from palette import DIV, NODATA  # noqa: E402

REPO = HERE.parents[1]
SHP = REPO / "data/raw_data/geo_wards/London_Ward.shp"
OUT_CSV = REPO / "results" / "models"
OUT_FIG = REPO / "results" / "paper" / "figures"
OUTCOME = "diabetes_prescriptions"
COORDS = ["centroid_lon", "centroid_lat"]   # British National Grid metres
MEASURE = "GSV_on_road_w"
TITLE = "on-road, street view, weighted"


def fit(data: pd.DataFrame, measure: str) -> tuple[pd.DataFrame, dict]:
    cols = list(CONFOUNDERS) + [measure]
    work = data.dropna(subset=cols + [OUTCOME] + COORDS).reset_index(drop=True)
    X = ((work[cols] - work[cols].mean()) / work[cols].std()).to_numpy()
    y_raw = work[OUTCOME].to_numpy(float)
    y = ((y_raw - y_raw.mean()) / y_raw.std()).reshape(-1, 1)
    coords = list(zip(work[COORDS[0]], work[COORDS[1]]))
    sel = Sel_BW(coords, y, X, multi=True)
    sel.search(multi_bw_min=[2])
    res = MGWR(coords, y, X, sel).fit()
    crit = res.critical_tval()                      # one adjusted critical t per covariate (intercept first)
    j = 1 + cols.index(measure)
    b, t = res.params[:, j], res.tvalues[:, j]
    to_pct = 100 * y_raw.std() / y_raw.mean()       # SD units of y -> % of mean prescribing, per 1 SD of greenery
    out = pd.DataFrame({"ward_code": work["ward_code"], "coef_sd": b, "coef_pct_per_sd": b * to_pct, "t": t,
                        "significant": np.abs(t) > crit[j]})
    info = {"measure": measure, "n": len(work), "bandwidths": dict(zip(["intercept"] + cols, np.asarray(sel.bw[0], float))),
            "crit_t": float(crit[j]), "adj_r2": float(res.adj_R2), "aicc": float(res.aicc)}
    return out, info


def draw(local: pd.DataFrame, info: dict, path: Path) -> None:
    gdf = gpd.read_file(SHP).to_crs(27700).merge(local, left_on="GSS_CODE", right_on="ward_code", how="left")
    lim = float(np.ceil(np.nanmax(np.abs(gdf["coef_pct_per_sd"])))) + 1   # symmetric, with room so a uniform effect is not drawn at the darkest end
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot("coef_pct_per_sd", ax=ax, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), linewidth=0.1, edgecolor="white",
             legend=True, legend_kwds=dict(shrink=0.6, label="% change in diabetes prescribing per 1 SD greenery"),
             missing_kwds=dict(color=NODATA))
    ns = gdf[~gdf["significant"].fillna(False).astype(bool) & gdf["coef_pct_per_sd"].notna()]
    ns.plot(ax=ax, facecolor="none", edgecolor="#777777", hatch="////", linewidth=0)
    gdf.dissolve("DISTRICT").boundary.plot(ax=ax, color="#555555", linewidth=0.5)
    ax.set_title(f"MGWR: diabetes and {TITLE}", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    csv, js = OUT_CSV / "mgwr_diabetes_wards.csv", OUT_CSV / "mgwr_diabetes_wards_info.json"
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    if "--draw" in sys.argv:
        local, info = pd.read_csv(csv), json.loads(js.read_text())
    else:
        local, info = fit(rename_columns(pd.read_csv(REPO / "data" / "data_analysis_wards7s.csv")), MEASURE)
        OUT_CSV.mkdir(parents=True, exist_ok=True)
        local.to_csv(csv, index=False)
        js.write_text(json.dumps(info, indent=1, default=float))
    draw(local, info, OUT_FIG / f"mgwr_diabetes_{MEASURE}.pdf")
    s = local[local.significant]
    print(info, f"significant wards: {len(s)} of {len(local)} (negative {int((s.coef_pct_per_sd < 0).sum())}),",
          f"% per SD: median {local.coef_pct_per_sd.median():.2f}, range {local.coef_pct_per_sd.min():.2f} to {local.coef_pct_per_sd.max():.2f}")
