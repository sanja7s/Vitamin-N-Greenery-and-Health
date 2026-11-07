# Vitamin N

Code and data for:

> Šćepanović, S., Joglekar, S., Law, S., Quercia, D., Zhou, K., Battiston, A., & Schifanella, R.
> *Vitamin N: Benefits of Different Forms of Public Greenery for Urban Health.*
> [arXiv:2508.12998](https://arxiv.org/abs/2508.12998)

The paper links greenery in London to NHS prescribing in 2016, for 632 wards and 4,835 LSOAs. It proposes two kinds of public greenery, **on-road** (greenery seen along streets) and **off-road** (public parks and gardens), and compares them with four official measures (NDVI, WHO, ESA WHO, Natural England). Outcomes are the yearly quantity of drugs per registered patient for diabetes, hypertension, asthma and opioids, and all prescribing.

**[results/results.html](results/results.html)** is an interactive summary of all results (download it and open it in a browser): the measures, the main models for wards and LSOAs, the cost estimate, the check against diagnoses, and maps.

## Layout

Each folder's files are numbered in the order they run.

```text
code/
  prescriptions/    01 download practice-level prescribing (NHSBSA open-data API), 02 share it out to LSOAs and
                    wards by where each practice's patients live, 03 check against QOF diagnoses, 04 drug costs
  greenery/         01 total aerial greenery per area (NDVI), 02 aerial greenery per road, 03 on-road and off-road
                    aerial greenery per area, 04 street view on-road greenery per area
  analysis_tables/  01 wards, 02 LSOAs: prescribing + greenery + controls, one row per area
  models/           causal.py (estimators); 01 PSM and double ML (Tables 1-2), 02 OLS, 03 MGWR, 04 cost estimates
  paper/            01 figures, 02 tables, 03 SI tables and figures, 04 results page; palette.py (colours)
data/
  raw_data/         inputs (sources below)
  prescriptions/    2016 prescribing per LSOA and ward, per condition (output of prescriptions/02)
  greenery_metrics/ greenery per road segment, LSOA and ward (output of greenery/01-04), official measures
  data_analysis_{wards,lsoas}7s.csv   the tables every model reads
results/
  models/           model results (CSV)
  qof/              prescribing against QOF diagnoses
  paper/            figures and LaTeX tables of the paper and SI
  results.html      interactive summary
```

The logic of a step is in its `*_workers.py` file; the notebook with the same number runs it. Every script and notebook starts with its inputs, outputs and running time.

## Reproducing the results

Data files are stored with [Git LFS](https://git-lfs.com) (about 0.7 GB). Run `git lfs install` once before cloning.

```bash
conda env create -f environment.yml
conda activate vitamin_n
```

**From the analysis tables to the paper** (about 1.5 hours, most of it double ML for LSOAs). Run from each folder:

1. `code/analysis_tables/01_analysis_table_wards.ipynb`, `02_analysis_table_lsoas.ipynb` (rebuild the tables; optional, they are included)
2. `code/models/01_psm_dml.ipynb`, then `python 02_ols.py`, `python 03_mgwr_diabetes.py`, `python 04_cost_estimates.py`
3. `code/paper/`: `python 01_figures.py`, `02_tables.py`, `03_si.py`, `04_results_page.py`

**Prescribing from source.** `code/prescriptions/01` downloads the 2016 prescribing of London practices from the
NHSBSA open-data API (about 40 minutes); the downloaded files are included, so `02` (allocation), `03` (QOF check)
and `04` (costs) can run straight away.

**Greenery from source.** Steps `01`-`03` need the London Green Cover map (aerial images, 2016) as 2 km tiles in
`data/raw_data/green_cover_tiles/`; they are not included (about 13 GB). Step `04` needs only the included street view
tree shares and roads. The greenery files in `data/greenery_metrics/` are the ones used in the paper. Rerunning the
greenery steps with this code gives values that agree closely with them (r > 0.99) but not to the last digit.

The depression row "quantity per patient vs new diagnoses" in SI Table S1 comes from a separate
England-wide download that is not part of this repository; its result is included in
`results/qof/qof_depression_new_diagnoses_2016.csv`.

## Data sources

| Data | Source | Files |
|---|---|---|
| Prescribing per GP practice, 2016 | NHSBSA English Prescribing Dataset (Open Government Licence) | `raw_data/prescriptions_epd/` |
| Registered patients per practice and LSOA, April 2016 | NHS Digital, Patients Registered at a GP Practice (OGL) | `raw_data/gp_registrations/` |
| Diagnoses per practice, 2016/17 | NHS Digital, Quality and Outcomes Framework (OGL) | `raw_data/qof/` |
| Drug lists per condition | MedSat lists, checked against the EPD (see its README) | `raw_data/bnf_lists/` |
| Age, ethnicity, household deprivation | Census 2011, ONS (OGL) | `raw_data/controls/` |
| Building height per area | GLA building footprints | `raw_data/density_data/` |
| Ward and LSOA boundaries, LSOA → ward lookup | GLA London Datastore and ONS (OGL; contains OS data © Crown copyright) | `raw_data/geo_*`, `raw_data/lookups/` |
| Roads | OpenStreetMap (ODbL), London drive network downloaded with OSMnx | `raw_data/road_network/` |
| Road centrality (space syntax choice, 800 m) | Space syntax segment model of London within the M25 | `raw_data/M25/` |
| Public parks and gardens | OpenStreetMap (ODbL) | `raw_data/ward_public_green/` |
| Tree share per street view image | Google Street View images, segmented with SegNet (images not included) | `raw_data/GSV/` |
| Total greenery | GLA London Green Cover map (aerial tiles not included) | `greenery_metrics/SAT_greenery_total_*` |
| WHO, ESA WHO, Natural England | Battiston & Schifanella (2024) | `greenery_metrics/official_indices/` |

## Citation

If you use this code or data, please cite the paper (arXiv:2508.12998) and, for the official accessibility
measures, Battiston & Schifanella (2024).

## Licence

Code: MIT, see [LICENSE](LICENSE). Data keep the licences of their sources (table above).
