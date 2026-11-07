"""Practice-level prescribing per condition from the NHSBSA English Prescribing Dataset (Methods).

Queries the open-data SQL API (one `EPD_YYYYMM` table per month) for London-region practices,
summing items / quantity / ADQ per practice × BNF chemical (opioids: × presentation, for OME).
Inputs : data/raw_data/bnf_lists/*.csv (condition → BNF codes)
Outputs: data/raw_data/prescriptions_epd/{condition}_{YYYYMM}.csv (one per month; a month is done
         iff its CSV exists), practices_{YYYYMM}.csv, epd_{year}_{condition}.csv (merged year)
Run    : from 01_epd_download.ipynb, or `python 01_epd_download_workers.py [year]`
"""
from __future__ import annotations

import gzip
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BNF_LISTS_DIR = REPO_ROOT / "data" / "raw_data" / "bnf_lists"
OUT_DIR = REPO_ROOT / "data" / "raw_data" / "prescriptions_epd"

API = "https://opendata.nhsbsa.net/api/3/action/datastore_search_sql"
REGION = "LONDON"                      # REGIONAL_OFFICE_NAME of the practices queried
CONDITIONS = ["diabetes", "hypertension", "asthma", "depression", "anxiety", "opioids", "total"]
CODES_PER_QUERY = 25                   # keeps most answers under the 32 000-row inline limit
RETRIES = 4
TIMEOUT_S = 600


def load_codes() -> dict[str, list[str]]:
    """Condition → BNF codes. Chemical (9-char) for five conditions, presentation (15-char) for opioids."""
    codes = {}
    for cond in CONDITIONS:
        if cond == "total":
            continue
        if cond == "opioids":
            codes[cond] = sorted(pd.read_csv(BNF_LISTS_DIR / "opioids_OME.csv", dtype=str)["bnf"].unique())
        else:
            codes[cond] = sorted(pd.read_csv(BNF_LISTS_DIR / f"{cond}.csv", dtype=str)["BNF_code"].unique())
    return codes


def _sql(resource: str, sql: str) -> pd.DataFrame:
    """Run one SQL query; large answers arrive as a gzipped CSV URL instead of inline rows."""
    url = API + "?" + urllib.parse.urlencode({"resource_id": resource, "sql": sql})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                payload = json.loads(r.read())
            if not payload.get("success"):
                raise RuntimeError(f"API error: {payload.get('error')}")
            res = payload["result"]
            res = res.get("result", res)
            if "records" in res:
                return pd.DataFrame(res["records"])
            if res.get("gc_urls"):
                parts = []
                for item in res["gc_urls"]:
                    with urllib.request.urlopen(item["url"], timeout=TIMEOUT_S) as g:
                        parts.append(pd.read_csv(io.BytesIO(gzip.decompress(g.read()))))
                return pd.concat(parts, ignore_index=True)
            raise RuntimeError(f"unexpected API answer: {str(res)[:300]}")
        except Exception as e:  # network hiccups and 5xx are common; back off and retry
            if attempt == RETRIES - 1:
                raise
            time.sleep(10 * (attempt + 1))
            last = e
    raise RuntimeError(last)


def _where(practice_codes: list[str] | None) -> str:
    if practice_codes:
        return "PRACTICE_CODE IN (" + ",".join(f"'{p}'" for p in practice_codes) + ")"
    return f"REGIONAL_OFFICE_NAME = '{REGION}'"


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def month_path(condition: str, ym: int, tag: str = "") -> Path:
    return OUT_DIR / f"{condition}_{ym}{tag}.csv"


def fetch_month(condition: str, ym: int, codes: dict[str, list[str]],
                practice_codes: list[str] | None = None, tag: str = "") -> Path:
    """Items, total quantity and ADQ per practice (× BNF code) for one condition and month."""
    out = month_path(condition, ym, tag)
    if out.exists():
        return out
    table = f"`EPD_{ym}`"
    where = _where(practice_codes)
    if condition == "total":
        sql = (f"SELECT PRACTICE_CODE, SUM(ITEMS) AS items, SUM(TOTAL_QUANTITY) AS qty, "
               f"SUM(ADQUSAGE) AS adq FROM {table} WHERE {where} GROUP BY PRACTICE_CODE")
        df = _sql(f"EPD_{ym}", sql)
    else:
        code_col = "BNF_CODE" if condition == "opioids" else "BNF_CHEMICAL_SUBSTANCE"
        parts = []
        for i in range(0, len(codes[condition]), CODES_PER_QUERY):
            chunk = ",".join(f"'{c}'" for c in codes[condition][i:i + CODES_PER_QUERY])
            sql = (f"SELECT PRACTICE_CODE, {code_col}, SUM(ITEMS) AS items, SUM(TOTAL_QUANTITY) AS qty, "
                   f"SUM(ADQUSAGE) AS adq FROM {table} WHERE {where} AND {code_col} IN ({chunk}) "
                   f"GROUP BY PRACTICE_CODE, {code_col}")
            parts.append(_sql(f"EPD_{ym}", sql))
        df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df.insert(0, "YEAR_MONTH", ym)
    _atomic_write(df, out)
    return out


def fetch_practices(ym: int, practice_codes: list[str] | None = None, tag: str = "") -> Path:
    """Practice directory (name, CCG, postcode) for the practices present in that month."""
    out = OUT_DIR / f"practices_{ym}{tag}.csv"
    if out.exists():
        return out
    sql = (f"SELECT PRACTICE_CODE, PRACTICE_NAME, PCO_CODE, PCO_NAME, POSTCODE, SUM(ITEMS) AS items "
           f"FROM `EPD_{ym}` WHERE {_where(practice_codes)} "
           f"GROUP BY PRACTICE_CODE, PRACTICE_NAME, PCO_CODE, PCO_NAME, POSTCODE")
    df = _sql(f"EPD_{ym}", sql)
    df.insert(0, "YEAR_MONTH", ym)
    _atomic_write(df, out)
    return out


def run_year(year: int, conditions: list[str] = CONDITIONS,
             practice_codes: list[str] | None = None, tag: str = "") -> None:
    """Download every month of `year`; `practice_codes` + `tag` add practices outside the region."""
    codes = load_codes()
    for m in range(1, 13):
        ym = year * 100 + m
        fetch_practices(ym, practice_codes, tag)
        for cond in conditions:
            fetch_month(cond, ym, codes, practice_codes, tag)
            print(f"{ym} {cond} done", flush=True)


def merge_year(year: int, condition: str, tags: tuple[str, ...] = ("",)) -> pd.DataFrame:
    """Concatenate the twelve monthly files (and any extra-practice files) of one condition."""
    frames = [pd.read_csv(month_path(condition, year * 100 + m, t), dtype={"PRACTICE_CODE": str})
              for m in range(1, 13) for t in tags if month_path(condition, year * 100 + m, t).exists()]
    if len(frames) < 12 * len(tags):
        raise FileNotFoundError(f"{condition} {year}: {len(frames)}/{12 * len(tags)} monthly files")
    df = pd.concat(frames, ignore_index=True)
    _atomic_write(df, OUT_DIR / f"epd_{year}_{condition}.csv")
    return df


if __name__ == "__main__":
    run_year(int(sys.argv[1]) if len(sys.argv) > 1 else 2016)
