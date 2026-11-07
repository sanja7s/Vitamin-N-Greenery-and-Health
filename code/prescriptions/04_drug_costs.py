"""2016 cost of the drugs on the diabetes list dispensed by London practices (NHSBSA EPD), for the cost estimates in Results.
Inputs : NHSBSA EPD open-data SQL API (EPD_201601...12), data/raw_data/bnf_lists (via 01_epd_download_workers.load_codes)
Outputs: data/raw_data/prescriptions_epd/costs_2016_london.csv (condition, month, BNF chemical: items, qty, nic, actual_cost)
Run    : python 04_drug_costs.py   (about 5 minutes)
Notes  : London = REGIONAL_OFFICE_NAME 'LONDON'; cost per item = actual_cost / items (NIC = list price before discounts).
"""
import importlib

import pandas as pd

dl = importlib.import_module("01_epd_download_workers")
OUT = dl.OUT_DIR / "costs_2016_london.csv"
CONDITIONS = ["diabetes"]


def main() -> None:
    codes, rows = dl.load_codes(), []
    for cond in CONDITIONS:
        for m in range(1, 13):
            ym = 201600 + m
            for i in range(0, len(codes[cond]), dl.CODES_PER_QUERY):
                chunk = ",".join(f"'{c}'" for c in codes[cond][i:i + dl.CODES_PER_QUERY])
                sql = (f"SELECT BNF_CHEMICAL_SUBSTANCE, SUM(ITEMS) AS items, SUM(TOTAL_QUANTITY) AS qty, SUM(NIC) AS nic, "
                       f"SUM(ACTUAL_COST) AS actual_cost FROM `EPD_{ym}` WHERE REGIONAL_OFFICE_NAME = '{dl.REGION}' "
                       f"AND BNF_CHEMICAL_SUBSTANCE IN ({chunk}) GROUP BY BNF_CHEMICAL_SUBSTANCE")
                d = dl._sql(f"EPD_{ym}", sql)
                if len(d):
                    rows.append(d.assign(condition=cond, ym=ym))
            print(cond, ym, flush=True)
    df = pd.concat(rows, ignore_index=True)
    dl._atomic_write(df, OUT)
    t = df.groupby("condition")[["items", "actual_cost"]].sum()
    print((t["actual_cost"] / t["items"]).rename("actual cost per item"))


if __name__ == "__main__":
    main()
