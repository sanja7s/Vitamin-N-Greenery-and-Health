"""How diabetes prescribing and its cost might change with more greenery (Results: practical size of the findings).
Inputs : results/models/{dml,psm}_{wards,lsoas}.csv, data/data_analysis_{wards,lsoas}7s.csv,
         data/prescriptions/{ward,lsoa}_prescriptions_2016_diabetes.csv, data/raw_data/prescriptions_epd/costs_2016_london.csv
Outputs: results/models/cost_estimates_diabetes.csv (one row per level and greenery measure)
Run    : python 04_cost_estimates.py   (seconds)
Notes  : plus_sd = every area one SD greener (DML); to_median = areas below the median brought up to it (DML and PSM);
         items x the 2016 London cost per diabetes item. First-order: assumes the effect is causal and constant.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from causal import TREATMENTS  # noqa: E402

REPO = HERE.parents[1]
MODELS = REPO / "results" / "models"


def main() -> None:
    cost = pd.read_csv(REPO / "data/raw_data/prescriptions_epd/costs_2016_london.csv")
    cost = cost[cost.condition == "diabetes"]
    per_item = cost.actual_cost.sum() / cost["items"].sum()
    rows = []
    for lv, key, pkey in (("wards", "ward_code", "WD18CD"), ("lsoas", "lsoa_code", "LSOA11CD")):
        area = pd.read_csv(REPO / f"data/data_analysis_{lv}7s.csv").set_index(key)
        items = pd.read_csv(REPO / f"data/prescriptions/{lv[:-1]}_prescriptions_2016_diabetes.csv").set_index(pkey)["Total_items"]
        dml, psm = pd.read_csv(MODELS / f"dml_{lv}.csv"), pd.read_csv(MODELS / f"psm_{lv}.csv")
        for t, label in TREATMENTS:
            x = area[t].dropna()
            y = items.reindex(x.index).fillna(0)
            q = dml[(dml.Treatment == t) & (dml.Outcome == "diabetes_prescriptions")].iloc[0]
            p = psm[(psm.Treatment == t) & (psm.Outcome == "diabetes_prescriptions")].iloc[0]
            below = x <= x.median()
            plus_sd = q.Effect_pct / 100 * y.sum()
            lo, hi = sorted([(q.Effect_pct - 1.96 * q.SE_pct) / q.Effect_pct, (q.Effect_pct + 1.96 * q.SE_pct) / q.Effect_pct])
            rows.append({"level": lv, "measure": label, "dml_pct_per_sd": q.Effect_pct, "dml_significant": q.Significant,
                         "psm_pct": p.ATE_pct, "psm_significant": p.Significant, "items_2016": y.sum(),
                         "items_plus_sd": plus_sd, "items_plus_sd_low": plus_sd * hi, "items_plus_sd_high": plus_sd * lo,
                         "items_to_median_dml": (q.Effect_pct / 100 * (x.median() - x[below]) / q.Treatment_SD * y[below]).sum(),
                         "items_to_median_psm": p.ATE_pct / 100 * y[below].sum(), "cost_per_item": per_item})
    out = pd.DataFrame(rows)
    for c in ("items_plus_sd", "items_plus_sd_low", "items_plus_sd_high", "items_to_median_dml", "items_to_median_psm", "items_2016"):
        out[c.replace("items", "cost_gbp")] = out[c] * per_item
    out.to_csv(MODELS / "cost_estimates_diabetes.csv", index=False)
    print(f"London 2016 diabetes items: {cost['items'].sum():,.0f}, cost £{cost.actual_cost.sum() / 1e6:.0f} million (£{per_item:.2f} per item)")
    print(out[out.measure == "on-road, street view, weighted"][["level", "items_plus_sd", "cost_gbp_plus_sd", "cost_gbp_plus_sd_low",
                                                               "cost_gbp_plus_sd_high", "items_to_median_psm", "cost_gbp_to_median_psm"]].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
