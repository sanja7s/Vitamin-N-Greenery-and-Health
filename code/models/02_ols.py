"""OLS of each prescribing outcome on the controls, alone and with one greenery measure at a time (SI Tables S3-S4).

Inputs : data/data_analysis_{wards,lsoas}7s.csv; controls, outcomes and measures from causal.py
Outputs: results/models/ols_{wards,lsoas}.csv (one row per model term: coefficient, p, adjusted R², N)
Run    : python 02_ols.py   (seconds)
Notes  : y and X standardised; HC3 standard errors; each model drops areas with missing values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from causal import CONFOUNDERS, OUTCOMES, TREATMENTS, rename_columns  # noqa: E402

REPO = HERE.parents[1]
OUT_DIR = REPO / "results" / "models"
SPECS = [(None, "Baseline (controls only)"), *TREATMENTS]


def fit_ols(outcome: str, data: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    """OLS (HC3 standard errors) of one standardised outcome on standardised controls."""
    work = data.dropna(subset=controls + [outcome])
    X = work[controls].to_numpy(float)
    y = work[[outcome]].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    y = (y - y.mean(0)) / y.std(0)
    res = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")
    return pd.DataFrame({"Outcome": outcome, "Control": ["Intercept"] + list(controls), "Coeff": res.params,
                         "P": res.pvalues, "AdjR2": res.rsquared_adj, "N": len(work)})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for level in ("wards", "lsoas"):
        data = rename_columns(pd.read_csv(REPO / "data" / f"data_analysis_{level}7s.csv"))
        rows = []
        for col, label in SPECS:
            cols = list(CONFOUNDERS) + ([col] if col else [])
            rows += [fit_ols(o, data, cols).assign(Greenery=col or "", Label=label) for o in OUTCOMES]
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(OUT_DIR / f"ols_{level}.csv", index=False)
        g = out[out["Control"] == out["Greenery"]]
        print(level, len(data), "areas\n", g.pivot(index="Label", columns="Outcome", values="Coeff").round(3).to_string())


if __name__ == "__main__":
    main()
