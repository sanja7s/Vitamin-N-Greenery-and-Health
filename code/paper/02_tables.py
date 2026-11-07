"""LaTeX tables of the paper: Table 1 (PSM), Table 2 (double ML), wards on top and LSOAs below, and the diabetes drug list.

Inputs : results/models/psm_{wards,lsoas}.csv, results/models/dml_{wards,lsoas}.csv, data/raw_data/bnf_lists/diabetes.csv
Outputs: results/paper/tables/table1_psm.tex, table2_dml.tex, table_diabetes_drugs.tex
Run    : python 02_tables.py
Notes  : significant cells are shaded with the colours negcell / poscell (palette.py: CELL_NEG, CELL_POS);
         define them in the LaTeX preamble, e.g. \\definecolor{negcell}{HTML}{DCE4F3}.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "results" / "models"
OUT = REPO / "results" / "paper" / "tables"

OUTCOMES = ["diabetes", "hypertension", "asthma", "opioids", "total"]
HEAD = ["diabetes", "hypert.", "asthma", "opioids", "total"]
GROUPS = [
    ("Official measures", [("SAT_NDVI", "NDVI"), ("WHO", "WHO"), ("ESA_WHO_exposure", "ESA WHO"), ("Natural_England1", "NE")]),
    ("Our measures", [("SAT_on_road_share_u", "on-road, aerial"), ("SAT_on_road_share_w", "on-road, aerial, weighted"),
                      ("GSV_on_road_u", "on-road, street view"), ("GSV_on_road_w", "on-road, street view, weighted"),
                      ("SAT_off_road", "off-road")]),
]


def cell(value: float, se: float, sig: bool) -> str:
    """One table cell: effect in %, SE in small type; significant cells bold and shaded (blue = fewer, red = more)."""
    num = f"{value:.1f}".replace("-0.0", "0.0")
    num = num if num.startswith("-") else r"\phantom{-}" + num
    body = (r"\textbf{" + num + "}" if sig else num) + r"{\tiny$\pm$" + f"{se:.1f}" + "}"
    shade = (r"\cellcolor{negcell}" if value < 0 else r"\cellcolor{poscell}") if sig else ""
    return shade + body


MODEL = {  # file stem, effect column, label, caption text
    "psm": ("psm_{lv}", "ATE_pct", "tab_all", "PSM",
            r"\textbf{Associations between greenery measures and prescription rates: propensity score matching (PSM).} "
            r"Each cell shows the average treatment effect (ATE): the difference in prescribing between areas above and"
            r" below the citywide median of a measure, as \% of the below-median mean, with its standard error."),
    "dml": ("dml_{lv}", "Effect_pct", "tab_dml", "Double ML",
            r"\textbf{Associations between greenery measures and prescription rates: double machine learning (DML).}"
            r" Each cell shows the change in prescribing for one standard deviation more of a measure, as \% of the mean"
            r" prescribing, with its standard error."),
}
COMMON = (r" Both levels control for building density, deprivation, median age and the percentage of White residents. Bold and shaded: significant at"
          r" the 5\% level (blue: fewer prescriptions; red: more). \emph{Aerial}: green cover in the 10\,m road buffers;"
          r" \emph{street view}: share of trees in street view images; \emph{weighted}: roads weighted by how central they are.}")


def results_table(model: str) -> str:
    """One model, outcomes as columns, measures as rows; the ward block on top and the LSOA block under it."""
    stem, v, label, short, cap = MODEL[model]
    dfs = {lv: pd.read_csv(MODELS / f"{stem.format(lv=lv)}.csv") for lv in ("wards", "lsoas")}
    lines = [r"\begin{table*}[t!]", r"\caption{" + cap + COMMON + "}", r"\label{" + label + "}",
             r"\centering", r"\footnotesize", r"\begin{tabular}{l*{5}{l}}", r"\toprule"]
    for i, (lv, name) in enumerate((("wards", "Wards"), ("lsoas", "LSOAs"))):
        df = dfs[lv]
        if i:
            lines.append(r"\midrule[\heavyrulewidth]")
        lines += [r"\textbf{" + name + " (" + f"{int(df.N.max()):,}" + r")} & " + " & ".join(HEAD) + r" \\"]
        for gname, items in GROUPS:
            lines += [r"\midrule", r"\multicolumn{6}{l}{\emph{" + gname + r"}} \\"]
            for t, lab in items:
                r = df[df.Treatment == t].set_index("Outcome")
                cells = [cell(r.loc[f"{o}_prescriptions", v], r.loc[f"{o}_prescriptions", "SE_pct"],
                              bool(r.loc[f"{o}_prescriptions", "Significant"])) for o in OUTCOMES]
                lines.append(r"\quad " + lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def diabetes_drugs() -> str:
    """Methods table of the corrected diabetes list (data/raw_data/bnf_lists/diabetes.csv), two code/name pairs per row."""
    d = pd.read_csv(REPO / "data" / "raw_data" / "bnf_lists" / "diabetes.csv").sort_values("BNF_code")
    d["Drug_name"] = d.Drug_name.str[0].str.upper() + d.Drug_name.str[1:].str.lower()
    pairs = list(zip(d.BNF_code, d.Drug_name))
    half = (len(pairs) + 1) // 2
    rows = [f"{a} & {b} & " + (" & ".join(pairs[i + half]) if i + half < len(pairs) else " & ") + r" \\"
            for i, (a, b) in enumerate(pairs[:half])]
    return "\n".join([
        r"\begin{table*}[t!]", r"    \centering", r"    \footnotesize",
        r"    \resizebox{0.8\linewidth}{!}{\begin{tabular}{|l|l|l|l|}", r"            \hline",
        r"            \textbf{BNF Code} &\textbf{Drug name} & \textbf{BNF Code} &\textbf{Drug name} \\", r"            \hline",
        *("            " + r for r in rows), r"            \hline", r"    \end{tabular}}",
        r"    \caption{List of drugs counted as diabetes prescriptions (" + str(len(pairs)) + r" BNF chemical codes"
        r"). Five of them (albiglutide, chlorpropamide, metformin/rosiglitazone, miglitol,"
        r" tolazamide) were not prescribed in London in 2016.}",
        r"    \label{tab:diabetesdrugs}", r"\end{table*}", ""])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table1_psm.tex").write_text(results_table("psm"))
    (OUT / "table2_dml.tex").write_text(results_table("dml"))
    (OUT / "table_diabetes_drugs.tex").write_text(diabetes_drugs())
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
