"""Interactive results page: measures, Tables 1-2 and OLS (wards and LSOAs), cost estimate, QOF check, correlations, maps.
Inputs : results/models/{psm,dml,ols}_{wards,lsoas}.csv, results/models/cost_estimates_diabetes.csv, results/qof/*.csv,
         data/data_analysis_{wards,lsoas}7s.csv, ward boundaries, data/raw_data/prescriptions_epd/costs_2016_london.csv
Outputs: results/results.html (self-contained: data and simplified ward shapes embedded as JSON)
Run    : python 04_results_page.py   (template: results_page_template.html, same folder)
Notes  : colours from palette.py; maps use seven classes of about a seventh of the wards each.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from matplotlib.colors import to_hex

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "models"))
from causal import OUTCOMES, TREATMENTS, rename_columns  # noqa: E402
from palette import BLUE, MAGENTA, TEAL, steps  # noqa: E402

MODELS = REPO / "results" / "models"
SHP = REPO / "data/raw_data/geo_wards/London_Ward.shp"
LABEL = dict(TREATMENTS)
WIDTH = 1000   # SVG units across London


def to_svg(geom, x0, y1, k) -> str:
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return "".join("M" + "L".join(f"{(x - x0) * k:.1f},{(y1 - y) * k:.1f}" for x, y in ring.coords[:-1]) + "Z"
                   for p in polys for ring in [p.exterior, *p.interiors])


def ward_shapes() -> dict:
    """Simplified ward and borough outlines in SVG units."""
    shapes = gpd.read_file(SHP).to_crs(27700)
    shapes["geometry"] = shapes.geometry.simplify(20, preserve_topology=True)
    boroughs = shapes.dissolve("DISTRICT").geometry.simplify(20)
    x0, y0, x1, y1 = shapes.total_bounds
    k = WIDTH / (x1 - x0)
    return {"w": WIDTH, "h": round((y1 - y0) * k), "boroughs": [to_svg(g, x0, y1, k) for g in boroughs],
            "areas": [{"c": r.GSS_CODE, "n": r.NAME, "d": to_svg(r.geometry, x0, y1, k)} for r in shapes.itertuples()]}


def box(v: pd.Series, label: str) -> dict:
    """Box-plot numbers; whiskers at the most extreme values within 1.5 IQR."""
    q1, med, q3 = v.quantile([.25, .5, .75]); iqr = q3 - q1
    stats = dict(q1=q1, med=med, q3=q3, lo=v[v >= q1 - 1.5 * iqr].min(), hi=v[v <= q3 + 1.5 * iqr].max(), min=v.min(), max=v.max())
    return {"g": label, "n": int(v.notna().sum()), **{k: round(float(x), 2) for k, x in stats.items()}}


def model_rows(model: str, level: str) -> list:
    if model == "ols":
        d = pd.read_csv(MODELS / f"ols_{level}.csv")
        d = d[d.Control == d.Greenery]
        return [{"g": LABEL[r.Greenery], "o": r.Outcome.replace("_prescriptions", ""), "e": round(r.Coeff, 3),
                 "s": bool(r.P < 0.05), "n": int(r.N)} for r in d.itertuples()]
    d = pd.read_csv(MODELS / f"{model}_{level}.csv")
    e, se = ("ATE_pct", "SE_pct") if model == "psm" else ("Effect_pct", "SE_pct")
    return [{"g": LABEL[r["Treatment"]], "o": r["Outcome"].replace("_prescriptions", ""), "e": round(r[e], 2),
             "se": round(r[se], 2), "s": bool(r["Significant"]), "n": int(r["N"])} for _, r in d.iterrows()]


def main() -> None:
    wd = rename_columns(pd.read_csv(REPO / "data/data_analysis_wards7s.csv")).set_index("ward_code")
    nl = len(pd.read_csv(REPO / "data/data_analysis_lsoas7s.csv"))
    cost = pd.read_csv(REPO / "data/raw_data/prescriptions_epd/costs_2016_london.csv")
    ce = pd.read_csv(MODELS / "cost_estimates_diabetes.csv")
    ce = ce[ce.measure == "on-road, street view, weighted"].set_index("level")
    qof = pd.read_csv(REPO / "results/qof/qof_vs_prescribing_2016.csv").query("method == 'pearson'")
    dep = pd.read_csv(REPO / "results/qof/qof_depression_new_diagnoses_2016.csv").set_index("measure")
    labels = {"quantity / patient vs QOF prevalence": "quantity per patient vs % diagnosed",
              "items / patient vs QOF prevalence": "items per patient vs % diagnosed",
              "total quantity vs QOF register": "total quantity vs number diagnosed"}
    qrows = [{"condition": r.condition, "comparison": labels[r.comparison], "r": r.r} for r in qof.itertuples()]
    qrows.insert(len(qrows), {"condition": "depression", "comparison": "quantity per patient vs new diagnoses",
                              "r": round(float(dep.loc["qty / patient", "pearson"]), 3)})
    qrows.append({"condition": "anxiety", "comparison": "no QOF register", "r": None})
    corr_cols = {LABEL[t]: t for t, _ in TREATMENTS} | {o.replace("_prescriptions", "").replace("hypertension", "hypert."): o for o in OUTCOMES}
    c = wd[list(corr_cols.values())].corr().round(2)
    short = {"SAT_on_road_share_u": "aerial", "GSV_on_road_u": "street view", "SAT_off_road": "off-road", "SAT_NDVI": "NDVI"}
    fig2 = {"left": [box(wd[t].dropna() * 100, lab) for t, lab in short.items()],
            "right": [box(wd[t].dropna() * 100, LABEL[t]) for t in ("ESA_WHO_exposure", "WHO", "Natural_England1")]}
    data = {
        "n": {"wards": len(wd), "lsoas": nl},
        "colours": {"teal": TEAL, "blue": BLUE, "magenta": MAGENTA},
        "ramps": {name: [to_hex(steps(name)(i)) for i in range(7)] for name in ("teal", "blue", "magenta")},
        "models": {m: {lv: model_rows(m, lv) for lv in ("wards", "lsoas")} for m in ("dml", "psm", "ols")},
        "cost": {"cost_per_item": float(cost.actual_cost.sum() / cost["items"].sum()), "items_total": float(cost["items"].sum()),
                 "cost_total": float(cost.actual_cost.sum()),
                 "rows": {lv: {k: (bool(v) if isinstance(v, (bool,)) or k.endswith("significant") else float(v))
                               for k, v in ce.loc[lv].items() if k != "measure"} for lv in ("wards", "lsoas")}},
        "qof": qrows,
        "corr": {"cols": list(corr_cols), "r": c.values.tolist(), "n": len(wd)},
        "fig1": [box(wd[o].dropna(), o.replace("_prescriptions", "")) for o in OUTCOMES],
        "fig2": fig2,
        "maps": {**ward_shapes(),
                 "measures": [{"g": LABEL[t], "vals": wd[t].dropna().round(5).to_dict()} for t, _ in TREATMENTS],
                 "outcomes": [{"g": o.replace("_prescriptions", ""), "vals": wd[o].dropna().round(3).to_dict()} for o in OUTCOMES]},
    }
    html = (HERE / "results_page_template.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = REPO / "results" / "results.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
