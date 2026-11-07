"""Allocate practice-level prescribing (01) to LSOAs and wards by registered-patient share (Methods, eq. 1-2).

LSOA total = Σ_gp total(gp) · f(gp, a), f = patients(gp, a) / patients(gp); per-capita = total / Σ_gp patients(gp, a).
Wards (2018) sum their LSOAs' totals and patients. Opioid OME = quantity × mg_per_unit × ome_multiplier.
Inputs : data/raw_data/prescriptions_epd/{condition}_{YYYYMM}*.csv, gp_registrations/gp-reg-patients-all-lsoa.csv
         (NHS Digital, April 2016, wide layout), bnf_lists/opioids_OME.csv, ONS LSOA11 → WD18 lookup
Outputs: data/prescriptions/{lsoa,ward}_prescriptions_{year}_{condition}.csv,
         data/raw_data/prescriptions_epd/unallocated_{year}.csv (prescribers with no registered patients)
Run    : from 02_allocate_to_areas.ipynb
"""
from __future__ import annotations

import csv
import importlib

import pandas as pd

_epd = importlib.import_module("01_epd_download_workers")

REPO_ROOT = _epd.REPO_ROOT
EPD_DIR = _epd.OUT_DIR
REG_PATH = REPO_ROOT / "data/raw_data/gp_registrations/gp-reg-patients-all-lsoa.csv"
REG_LONG_PATH = REG_PATH.with_name(REG_PATH.stem + "_long.csv")
LOOKUP_PATH = (REPO_ROOT / "data/raw_data/lookups"
               / "Lower_Layer_Super_Output_Area_(2011)_to_Ward_(2018)_Lookup_in_England_and_Wales_v3.csv")
OUT_DIR = REPO_ROOT / "data" / "prescriptions"
CONDITIONS = _epd.CONDITIONS
MIN_EXTRA_PATIENTS = 100   # practices outside the London region are added when they hold ≥ this many London patients


def load_registrations() -> pd.DataFrame:
    """Practice × LSOA registered patients (long form, cached). Source rows come in pairs: codes, then counts."""
    if REG_LONG_PATH.exists():
        return pd.read_csv(REG_LONG_PATH, dtype={"practice": str, "lsoa": str})
    rows, pending = [], {}
    with open(REG_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for code, _name, kind, *vals in reader:
            if kind == "LSOA_CODE":
                pending[code] = vals
            elif kind == "All Patients":
                for lsoa, v in zip(pending.pop(code), vals):
                    if lsoa and v.strip():
                        rows.append((code, lsoa, int(v)))
    reg = pd.DataFrame(rows, columns=["practice", "lsoa", "patients"])
    reg.to_csv(REG_LONG_PATH, index=False)
    return reg


def london_lookup() -> pd.DataFrame:
    lk = pd.read_csv(LOOKUP_PATH, dtype=str)
    return lk.loc[lk["LAD18CD"].str.startswith("E09"), ["LSOA11CD", "WD18CD", "LAD18CD", "LAD18NM"]].drop_duplicates("LSOA11CD")


def extra_practices(reg: pd.DataFrame, region_practices: set[str]) -> list[str]:
    """Practices outside the London region holding ≥ MIN_EXTRA_PATIENTS London patients (for a second 01 pass)."""
    lon = reg[reg["lsoa"].isin(london_lookup()["LSOA11CD"])]
    by = lon.groupby("practice")["patients"].sum()
    return sorted(by[(~by.index.isin(region_practices)) & (by >= MIN_EXTRA_PATIENTS)].index)


def region_practices(year: int) -> set[str]:
    files = sorted(EPD_DIR.glob(f"practices_{year}??.csv"))
    return set(pd.concat([pd.read_csv(f, dtype=str) for f in files])["PRACTICE_CODE"])


def load_epd(year: int, condition: str) -> pd.DataFrame:
    """All monthly files of a condition (region and extra passes), as practice totals for the year."""
    files = sorted(EPD_DIR.glob(f"{condition}_{year}??.csv")) + sorted(EPD_DIR.glob(f"{condition}_{year}??__*.csv"))
    months = {int(f.stem.split("_")[1][:6]) for f in files}
    if len(months) < 12:
        raise FileNotFoundError(f"{condition} {year}: only months {sorted(months)} downloaded")
    df = pd.concat([pd.read_csv(f, dtype={"PRACTICE_CODE": str, "BNF_CODE": str, "BNF_CHEMICAL_SUBSTANCE": str})
                    for f in files], ignore_index=True)
    if condition == "opioids":
        ome = pd.read_csv(_epd.BNF_LISTS_DIR / "opioids_OME.csv", dtype={"bnf": str}).drop_duplicates("bnf").set_index("bnf")
        factor = df["BNF_CODE"].map(ome["mg_per_unit"] * ome["ome_multiplier"])
        df["OME"] = df["qty"] * factor.fillna(0.0)
    totals = df.groupby("PRACTICE_CODE")[[c for c in ["items", "qty", "adq", "OME"] if c in df]].sum()
    return totals.rename(columns={"items": "Total_items", "qty": "Total_quantity", "adq": "ADQ"})


def allocate(condition: str, year: int, reg: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (lsoa table, ward table, unallocated practice totals) for one condition and year."""
    reg = load_registrations() if reg is None else reg
    lk = london_lookup()
    totals = load_epd(year, condition)

    share = reg.assign(f=reg["patients"] / reg.groupby("practice")["patients"].transform("sum"))
    share = share[share["lsoa"].isin(lk["LSOA11CD"])]
    alloc = share.merge(totals, left_on="practice", right_index=True, how="inner")
    measures = list(totals.columns)
    for m in measures:
        alloc[m] = alloc[m] * alloc["f"]
    lsoa = alloc.groupby("lsoa")[measures].sum()
    patients = reg[reg["lsoa"].isin(lk["LSOA11CD"])].groupby("lsoa")["patients"].sum()
    lsoa = lsoa.reindex(patients.index).fillna(0.0)
    lsoa["Patient_count"] = patients
    lsoa = lsoa.reset_index().rename(columns={"lsoa": "LSOA11CD"})

    ward = lsoa.merge(lk[["LSOA11CD", "WD18CD"]], on="LSOA11CD").groupby("WD18CD")[measures + ["Patient_count"]].sum().reset_index()
    for tbl in (lsoa, ward):
        for m in measures:
            tbl[m + "_per_capita"] = tbl[m] / tbl["Patient_count"]
        tbl["year"] = year

    unallocated = totals[~totals.index.isin(reg["practice"])].assign(condition=condition)
    return lsoa, ward, unallocated


def run_year(year: int, conditions: list[str] = CONDITIONS) -> pd.DataFrame:
    """Write LSOA and ward tables for every condition; return the per-condition allocation report."""
    reg = load_registrations()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report, unalloc = [], []
    for cond in conditions:
        lsoa, ward, un = allocate(cond, year, reg)
        lsoa.to_csv(OUT_DIR / f"lsoa_prescriptions_{year}_{cond}.csv", index=False)
        ward.to_csv(OUT_DIR / f"ward_prescriptions_{year}_{cond}.csv", index=False)
        total_items = load_epd(year, cond)["Total_items"].sum()
        report.append({"condition": cond, "items_downloaded": total_items,
                       "items_allocated_to_London": ward["Total_items"].sum(),
                       "items_unallocated_no_registrations": un["Total_items"].sum(),
                       "share_allocated": ward["Total_items"].sum() / total_items})
        unalloc.append(un)
    pd.concat(unalloc).to_csv(EPD_DIR / f"unallocated_{year}.csv")
    return pd.DataFrame(report)
