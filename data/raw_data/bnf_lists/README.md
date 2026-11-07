# Condition → BNF code lists

The drug lists that define each prescribing outcome. They start from the MedSat lists
(`sanja7s/MedSat`, `code/collate_data/NHS_prescription_parser/drug_names/`; DrugBank-derived, opioids from a curated
morphine-equivalent list) and were checked code by code against the NHSBSA English Prescribing Dataset (EPD).
Each `*_corrections.csv` maps every original row to its status and corrected code.

- `{diabetes,hypertension,asthma,depression,anxiety}.csv`: `BNF_code` is the 9-character chemical-substance code
  (`BNF_CHEMICAL_SUBSTANCE` in the EPD).
- `opioids_OME.csv`: 15-character presentation codes (`bnf`, all in BNF 4.7.2) with `mg_per_unit` and
  `ome_multiplier`; OME = quantity × mg_per_unit × ome_multiplier.

## What the checks changed

**Diabetes** (58 rows). Five drugs for other conditions were removed (desmopressin, which treats diabetes
*insipidus*; bromocriptine; colesevelam; hydrochlorothiazide and its potassium combination). Missing premixed and
other insulins, insulin degludec/liraglutide, empagliflozin/linagliptin and tirzepatide were added. Five listed drugs
were not prescribed in London in 2016 (albiglutide, chlorpropamide, metformin/rosiglitazone, miglitol, tolazamide)
and count nothing. Blood-glucose test strips are left out: they are monitoring supplies, not drugs.

**Hypertension** (112 rows). 25 rows for other conditions were removed (glaucoma eye drops, pulmonary arterial
hypertension drugs, ranitidine, tadalafil, an anal glyceryl trinitrate ointment, guanfacine for ADHD, adenosine),
with 4 unused duplicates. Diltiazem, minoxidil and terazosin were moved from their cream, hair-lotion and urology
codes to their blood-pressure codes, and 37 missing antihypertensives in BNF 2.2, 2.4, 2.5 and 2.6.2 were added
(lercanidipine, lacidipine, moxonidine, bumetanide, sotalol, combinations with diuretics, ...).

**Asthma** (51 rows). Six drugs pointed at a non-inhaled form or an unused code and now use their inhaled or
respiratory codes; 4 unused duplicates were removed; salbutamol, terbutaline, inhaled fluticasone propionate,
theophylline and aminophylline hydrate were added. Forms with no asthma use (skin, eye, ear, nose, ...) were kept and
are labelled with their form; 8 rows have no code in the EPD and count nothing.

**Depression** (32 rows, the whole of BNF 4.3). 21 of the original 31 codes did not match the drug named on the row
(12 were never used in the EPD, 9 belonged to another drug). Each drug now has its EPD code; citalopram
hydrochloride was added.

**Anxiety** (25 rows). Duloxetine was moved from its urinary-incontinence code to its antidepressant code, an unused
duplicate was removed, and pregabalin and sertraline were added. The list stays broad (most of its items are
antidepressants also in `depression.csv`), and QOF has no anxiety register to check it against.

**Opioids** (895 rows). 43 products had a wrong `mg_per_unit` and were fixed (oral liquids were given per 5 ml
although EPD quantity is in ml; fentanyl sublingual tablets had micrograms entered as mg; a few single products).
88 duplicates were removed and 143 products prescribed in 2014–2025 but not listed were added. The scope is BNF 4.7.2
only, so co-codamol and co-dydramol (4.7.1) and treatments for opioid dependence are not counted.

Depression and anxiety prescribing do not track diagnoses well (SI Table S1), so the paper's models use the
diabetes, hypertension, asthma and opioid lists, plus all prescribing.
