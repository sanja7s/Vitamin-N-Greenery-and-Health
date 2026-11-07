"""Check the 2016 ward prescribing against the QOF disease registers of 2016-17 (SI Table S1).

QOF practice registers are allocated to wards with the same April-2016 registration shares as the prescribing (02):
ward register = Σ_gp register(gp) · f(gp, ward); prevalence = allocated register / allocated list size.
Inputs : data/raw_data/qof/PREVALENCE_1617.csv (NHS Digital QOF 2016-17), registrations and lookup (02), data/prescriptions/
Outputs: results/qof/qof_registers_ward_1617.csv, results/qof/qof_vs_prescribing_2016.csv
Run    : python 03_qof_check.py   (about 1 minute)
Notes  : no QOF register for anxiety or opioids. DEP counts diagnoses since April 2006 (18+), AST needs treatment in 12 months.
"""
import importlib

import pandas as pd

_alloc = importlib.import_module("02_allocate_to_areas_workers")
REPO_ROOT = _alloc.REPO_ROOT
QOF_PATH = REPO_ROOT / "data/raw_data/qof/PREVALENCE_1617.csv"
PRESC_DIR = REPO_ROOT / "data/prescriptions"
OUT_DIR = REPO_ROOT / "results/qof"
REGISTERS = {"diabetes": "DM", "hypertension": "HYP", "asthma": "AST", "depression": "DEP"}
YEAR = 2016


def qof_wards() -> pd.DataFrame:
    """Register, list size and prevalence (%) per condition and ward."""
    qof = pd.read_csv(QOF_PATH, dtype={"PRACTICE_CODE": str})
    qof = qof[qof["INDICATOR_GROUP_CODE"].isin(REGISTERS.values())]
    reg, lk = _alloc.load_registrations(), _alloc.london_lookup()
    share = reg.assign(f=reg["patients"] / reg.groupby("practice")["patients"].transform("sum"))
    share = share.merge(lk[["LSOA11CD", "WD18CD"]], left_on="lsoa", right_on="LSOA11CD")
    a = share.merge(qof, left_on="practice", right_on="PRACTICE_CODE")
    a["register"] = a["REGISTER"] * a["f"]
    a["list_size"] = a["PATIENT_LIST_SIZE"] * a["f"]
    w = a.groupby(["INDICATOR_GROUP_CODE", "WD18CD"])[["register", "list_size"]].sum().reset_index()
    w["prevalence_pct"] = 100 * w["register"] / w["list_size"]
    # share of each ward's registered patients whose practice reported to QOF
    covered = share[share["practice"].isin(qof["PRACTICE_CODE"])].groupby("WD18CD")["patients"].sum()
    w["qof_patient_coverage"] = w["WD18CD"].map(covered / share.groupby("WD18CD")["patients"].sum())
    w["condition"] = w["INDICATOR_GROUP_CODE"].map({v: k for k, v in REGISTERS.items()})
    return w


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w = qof_wards()
    w.to_csv(OUT_DIR / "qof_registers_ward_1617.csv", index=False)
    rows = []
    for cond in REGISTERS:
        q = w[w["condition"] == cond].set_index("WD18CD")
        p = pd.read_csv(PRESC_DIR / f"ward_prescriptions_{YEAR}_{cond}.csv").set_index("WD18CD")
        idx = q.index.intersection(p.index)
        for label, col, qcol in [("quantity / patient vs QOF prevalence", "Total_quantity_per_capita", "prevalence_pct"),
                                 ("items / patient vs QOF prevalence", "Total_items_per_capita", "prevalence_pct"),
                                 ("total quantity vs QOF register", "Total_quantity", "register")]:
            for method in ("pearson", "spearman"):
                rows.append({"condition": cond, "comparison": label, "method": method, "n_wards": len(idx),
                             "r": round(p.loc[idx, col].corr(q.loc[idx, qcol], method=method), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"qof_vs_prescribing_{YEAR}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
