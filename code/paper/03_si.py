"""SI tables and figures: OLS (S3-S4), prescribing against QOF diagnoses (S1), variable summaries (S2),
correlation matrices and the ward map / histogram pairs (S2-S4 figures).
Inputs : results/models/ols_{wards,lsoas}.csv, results/qof/*.csv, data/data_analysis_{wards,lsoas}7s.csv, ward boundaries
Outputs: results/paper/tables/si_*.tex, results/paper/figures/{dist_*.pdf, corr_{wards,lsoas}.png, mapdist_*.pdf}
Run    : python 03_si.py
Notes  : colours from palette.py; mapdist_* = ward map in seven classes (each about a seventh of the wards) next to its histogram.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

from palette import DIV, INK, MAGENTA, NODATA, SEQ, family, steps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "paper" / "tables"
FIG = REPO / "results" / "paper" / "figures"
OUTCOMES = ["diabetes", "hypertension", "asthma", "opioids", "total"]
HEAD = ["diabetes", "hypert.", "asthma", "opioids", "total"]
CONTROLS = [("median_age", "median age", "median age"), ("IMD_deprivation_score", "deprivation", "IMD deprivation score"),
            ("white_percent", "White \\%", "white percent"), ("building_density", "build.\\ dens.", "building_density")]
MEASURES = [  # model name, label, column in the analysis tables, x100 for display
    ("SAT_NDVI", "NDVI", "SAT_NDVI", 100), ("WHO", "WHO", "WHO", 100), ("ESA_WHO_exposure", "ESA WHO", "ESA_WHO_exposure", 100),
    ("Natural_England1", "NE", "Natural_England1", 100),
    ("SAT_on_road_share_u", "on-road, aerial", "SAT_on_road_share_u", 100),
    ("SAT_on_road_share_w", "on-road, aerial, weighted", "SAT_on_road_share_w", 100),
    ("GSV_on_road_u", "on-road, street view", "GSV_on_road_u", 100),
    ("GSV_on_road_w", "on-road, street view, weighted", "GSV_on_road_w", 100),
    ("SAT_off_road", "off-road", "SAT_off_road", 100)]
ACCESS = {"WHO", "ESA_WHO_exposure", "Natural_England1"}   # WHO and NE targets: share of residents with access, not share green


def coef_cell(c: float, p: float) -> str:
    num = f"{c:.3f}"
    num = num if num.startswith("-") else r"\phantom{-}" + num
    if p < 0.05:
        return (r"\cellcolor{negcell}" if c < 0 else r"\cellcolor{poscell}") + (r"\textbf{" + num + "}")
    return (num)


def ols_tables() -> str:
    w = pd.read_csv(REPO / "results/models/ols_wards.csv")
    l = pd.read_csv(REPO / "results/models/ols_lsoas.csv")
    out = []
    # baseline: controls only
    lines = [r"\begin{table}[hbt!]", r"\caption{" + (r"\textbf{OLS with the controls only.} Standardized coefficients; bold and shaded: $p<0.05$ (HC3 standard errors).") + r"}\label{ols_baseline}",
             r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l*{10}{l}}", r"\toprule",
             r" & \multicolumn{5}{c}{" + ("Wards (632)") + r"} & \multicolumn{5}{c}{" + ("LSOAs (4,835)") + r"} \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}", " & " + " & ".join(HEAD * 2) + r" \\", r"\midrule"]
    for key, lab, _ in CONTROLS:
        cells = []
        for d in (w, l):
            b = d[d.Label == "Baseline (controls only)"]
            for o in OUTCOMES:
                r = b[(b.Outcome == f"{o}_prescriptions") & (b.Control == key)].iloc[0]
                cells.append(coef_cell(r.Coeff, r.P))
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    r2 = []
    for d in (w, l):
        b = d[d.Label == "Baseline (controls only)"]
        r2 += [(f"{b[b.Outcome == f'{o}_prescriptions'].AdjR2.iloc[0]:.2f}") for o in OUTCOMES]
    lines += [r"\midrule", r"Adj.\ $R^2$ & " + " & ".join(r2) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out += lines
    # greenery: one model per measure (controls + that measure); wards on top, LSOAs below
    lines = [r"\begin{table}[hbt!]", r"\caption{" + (r"\textbf{OLS with the controls and one greenery measure at a time.} Standardized coefficient of the greenery measure; each cell is a separate model. Bold and shaded: $p<0.05$ (HC3 standard errors).") + r"}\label{ols_greenery}",
             r"\centering\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l*{5}{l}}", r"\toprule"]
    for i, (d, name) in enumerate(((w, "Wards (632)"), (l, "LSOAs (4,835)"))):
        if i:
            lines.append(r"\midrule[\heavyrulewidth]")
        lines += [(r"\textbf{" + name + "}") + " & " + " & ".join(HEAD) + r" \\", r"\midrule"]
        for key, lab, _, _ in MEASURES:
            cells = []
            for o in OUTCOMES:
                r = d[(d.Greenery == key) & (d.Outcome == f"{o}_prescriptions") & (d.Control == key)].iloc[0]
                cells.append(coef_cell(r.Coeff, r.P))
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
            if key == "Natural_England1":
                lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out + lines)


def qof_table() -> str:
    q = pd.read_csv(REPO / "results/qof/qof_vs_prescribing_2016.csv")
    q = q[q.method == "pearson"]
    dd = pd.read_csv(REPO / "results/qof/qof_depression_new_diagnoses_2016.csv")
    comps = [("quantity / patient vs QOF prevalence", "quantity per patient vs \\% diagnosed"),
             ("items / patient vs QOF prevalence", "items per patient vs \\% diagnosed"),
             ("total quantity vs QOF register", "total quantity vs number diagnosed")]
    lines = [r"\begin{table}[hbt!]",
             r"\caption{" + (r"\textbf{Prescribing outcomes against diagnoses.} Pearson correlation across the 637 London wards between each prescribing outcome and the QOF disease register of 2016/17 (share or number of patients diagnosed). QOF has no register for anxiety or opioid use.") + r"}\label{tab_qof}",
             r"\centering\footnotesize", r"\begin{tabular*}{0.85\linewidth}{@{\extracolsep{\fill}}lll}", r"\toprule",
             r"Condition & Comparison & $r$ \\", r"\midrule"]
    for cond in ["diabetes", "hypertension", "asthma", "depression"]:
        for i, (c, lab) in enumerate(comps):
            r = q[(q.condition == cond) & (q.comparison == c)].iloc[0]
            lines.append((cond if i == 0 else "") + " & " + (lab) + " & " + (f"{r.r:.2f}") + r" \\")
        if cond == "depression":
            r = dd[dd.measure == "qty / patient"].iloc[0]
            lines.append(" & " + ("quantity per patient vs new diagnoses") + " & " + (f"{r.pearson:.2f}") + r" \\")
        lines.append(r"\midrule")
    lines += [r"anxiety & " + ("no QOF register") + r" & \\", r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def spark(v: pd.Series, path: Path, colour: str = "#7a7975") -> None:
    fig, ax = plt.subplots(figsize=(1.6, 0.35))
    ax.hist(v.dropna(), bins=30, color=colour)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path)
    plt.close(fig)


def distributions() -> str:
    wd = pd.read_csv(REPO / "data" / "data_analysis_wards7s.csv")
    grey = "#7a7975"
    rows = [("Controls", [(lab, wd[col], 1, col, grey) for _, lab, col in CONTROLS]),
            ("Prescriptions (per patient per year)", [(o, wd[f"{o} prescriptions"], 1, f"{o}_rx", MAGENTA) for o in OUTCOMES]),
            ("Greenery (\\% green)", [(lab, wd[col], k, key, SEQ[family(key)](0.6)) for key, lab, col, k in MEASURES
                                         if key not in ACCESS]),
            ("Access to greenery (\\% of people)", [(lab, wd[col], k, key, SEQ[family(key)](0.6))
                                                     for key, lab, col, k in MEASURES if key in ACCESS])]
    lines = [r"\begin{table}[!hbt]",
             r"\caption{" + (r"Distributions and statistics across the 632 wards for the variables used in our models. Building density is log-transformed. All variables were standardized before being fed into the OLS models.") + r"}",
             r"\label{tab:variable_distrbutions}", r"\centering", r"\begin{tabular}{llllll}"]
    for name, items in rows:
        lines += [name + r" & Distribution & Min & Median & Max & Std \\ \midrule"]
        for lab, v, k, stem, colour in items:
            v = v.dropna() * k
            spark(v, FIG / f"dist_{stem}.pdf", colour)
            f = (lambda x: f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}")
            lines.append(lab + r" & \includegraphics[height=1.3em]{figures/dist_" + stem + ".pdf} & "
                         + " & ".join((f(x)) for x in (v.min(), v.median(), v.max(), v.std())) + r" \\")
        lines.append(r"\addlinespace[1em]")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def corr_figure(level: str) -> None:
    df = pd.read_csv(REPO / "data" / f"data_analysis_{level}7s.csv")
    cols = {lab.replace("\\", ""): col for _, lab, col in CONTROLS}
    cols |= {lab: col for _, lab, col, _ in MEASURES}
    cols |= {o: f"{o} prescriptions" for o in OUTCOMES}
    c = df[list(cols.values())].corr().values[1:, :-1]   # lower triangle without the diagonal: drop first row, last column
    labels = list(cols)
    n = len(labels) - 1
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(np.ma.array(c, mask=np.triu(np.ones_like(c, bool), 1)), cmap=DIV, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            ax.text(j, i, f"{c[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=6.5,
                    color="white" if abs(c[i, j]) > 0.65 else INK)
    ax.set_xticks(range(n), labels[:-1], rotation=90, fontsize=9)
    ax.set_yticks(range(n), labels[1:], fontsize=9)
    for b in (2.5, 11.5):   # group lines: controls | greenery | prescribing
        ax.axhline(b, color="white", lw=2)
    for b in (3.5, 12.5):
        ax.axvline(b, color="white", lw=2)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, shrink=0.6, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIG / f"corr_{level}.png", dpi=200)
    plt.close(fig)


SHP = REPO / "data/raw_data/geo_wards/London_Ward.shp"
UNITS = {"SAT_NDVI": ("NDVI", "mean NDVI", 1), "WHO": ("WHO", "% residents meeting WHO target", 100),
         "ESA_WHO_exposure": ("ESA WHO", "% residents meeting ESA WHO target", 100),
         "Natural_England1": ("NE", "% residents meeting NE target", 100),
         "SAT_on_road_share_u": ("On-road, aerial", "% of road buffer green", 100),
         "SAT_on_road_share_w": ("On-road, aerial, weighted", "% of road buffer green, weighted", 100),
         "GSV_on_road_u": ("On-road, street view", "% trees in street view", 100),
         "GSV_on_road_w": ("On-road, street view, weighted", "% trees in street view, weighted", 100),
         "SAT_off_road": ("Off-road", "% of ward in public green space", 100)}


def map_hist(ax_map, ax_hist, gdf: gpd.GeoDataFrame, v: pd.Series, ramp: str, title: str, unit: str) -> None:
    """Ward map in septiles (each colour about a seventh of the wards) and its histogram with the same colours."""
    cmap = steps(ramp)
    q = v.dropna().quantile([i / 7 for i in range(1, 7)]).to_numpy()
    cls = pd.Series(np.searchsorted(q, v.to_numpy(), side="right"), index=v.index).where(v.notna())
    g = gdf.assign(c=cls.reindex(gdf.index))
    g[g.c.isna()].plot(ax=ax_map, color=NODATA, linewidth=0)
    g[g.c.notna()].plot(ax=ax_map, color=[cmap(int(k)) for k in g.c.dropna()], linewidth=0)
    ax_map.set_axis_off()
    ax_map.set_title(title, fontsize=10, loc="left", color=INK, fontweight="bold")
    x = v.dropna()
    counts, edges = np.histogram(x, bins=24)
    mids = (edges[:-1] + edges[1:]) / 2
    ax_hist.bar(mids, counts, width=np.diff(edges) * 0.9, color=[cmap(int(np.searchsorted(q, m, side="right"))) for m in mids])
    ax_hist.axvline(x.median(), color=INK, ls="--", lw=0.9)
    ax_hist.set_xlabel(unit, fontsize=7.5, color=INK)
    ax_hist.tick_params(labelsize=7, colors="#52514e")
    ax_hist.spines[["top", "right", "left"]].set_visible(False)
    ax_hist.set_yticks([])


def mapdist_figure(items: list, path: Path) -> None:
    """items: (series indexed by ward code, ramp, title, unit); two map/histogram pairs per row."""
    gdf = gpd.read_file(SHP).to_crs(27700).set_index("GSS_CODE")
    rows = (len(items) + 1) // 2
    fig = plt.figure(figsize=(10, 2.5 * rows))
    gs = fig.add_gridspec(rows, 5, width_ratios=[1.5, 1.1, 0.22, 1.5, 1.1], wspace=0.15, hspace=0.45)
    for i, (v, ramp, title, unit) in enumerate(items):
        r, c = divmod(i, 2)
        map_hist(fig.add_subplot(gs[r, 3 * c]), fig.add_subplot(gs[r, 3 * c + 1]), gdf, v, ramp, title, unit)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def mapdist_figures() -> None:
    wd = pd.read_csv(REPO / "data" / "data_analysis_wards7s.csv").set_index("ward_code")
    mapdist_figure([(wd[f"{o} prescriptions"], "magenta", "Total (all prescribing)" if o == "total" else o.capitalize(),
                     "yearly quantity per patient") for o in OUTCOMES], FIG / "mapdist_prescriptions.pdf")
    for name, keep in (("ours", lambda k: family(k) == "teal"), ("official", lambda k: family(k) == "blue")):
        mapdist_figure([(wd[col] * UNITS[key][2], family(key), UNITS[key][0], UNITS[key][1])
                        for key, _, col, _ in MEASURES if keep(key)], FIG / f"mapdist_greenery_{name}.pdf")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "si_ols.tex").write_text(ols_tables())
    (OUT / "si_qof.tex").write_text(qof_table())
    (OUT / "si_distributions.tex").write_text(distributions())
    for lv in ("wards", "lsoas"):
        corr_figure(lv)
    mapdist_figures()
    print("wrote", [p.name for p in OUT.glob("si_*.tex")])
