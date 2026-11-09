"""Estimators for greenery -> prescribing (Methods): propensity score matching (PSM) and double ML.

Estimators: 1-NN propensity-score matching ATE (logit PS, caliper 0.2 SD) with a
doubly-robust AIPW cross-check, both bootstrapped (iid or by spatial block); and
cross-fitted double ML for greenery kept continuous (effect per 1 SD).
Inputs : an analysis table (data/data_analysis_{wards,lsoas}7s.csv), read through `rename_columns`.
Outputs: one row per (treatment, outcome); `table1` formats the wide view.
Notes  : outcomes stay in natural units (quantity per registered patient per year).
         The bootstrap is not consistent for NN matching (Abadie & Imbens 2008);
         treat PSM intervals as indicative and lean on the AIPW columns.
"""
 
from __future__ import annotations
 
import warnings
from typing import Iterable, Optional, Sequence
 
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn import __version__ as _SKLEARN_VERSION
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
CONFOUNDERS: tuple[str, ...] = (
    "white_percent",
    "median_age",
    "IMD_deprivation_score",
    "building_density",
)
 
OUTCOMES: tuple[str, ...] = (
    "diabetes_prescriptions",
    "hypertension_prescriptions",
    "asthma_prescriptions",
    "opioids_prescriptions",
    "total_prescriptions",
)

# (column, label): the four widely used measures, then our five (Tables 1 and 2)
TREATMENTS: tuple[tuple[str, str], ...] = (
    ("SAT_NDVI", "NDVI"),
    ("WHO", "WHO"),
    ("ESA_WHO_exposure", "ESA WHO"),
    ("Natural_England1", "NE"),
    ("SAT_on_road_share_u", "on-road, aerial"),
    ("SAT_on_road_share_w", "on-road, aerial, weighted"),
    ("GSV_on_road_u", "on-road, street view"),
    ("GSV_on_road_w", "on-road, street view, weighted"),
    ("SAT_off_road", "off-road"),
)

# Unpenalized logistic regression: sklearn < 1.2 needs penalty="none", 1.2-1.7
# penalty=None, and >= 1.8 deprecates `penalty` in favour of C=np.inf.
_SK = tuple(int(v) for v in _SKLEARN_VERSION.split(".")[:2])
_NO_PENALTY = {"C": np.inf} if _SK >= (1, 8) else {"penalty": None if _SK >= (1, 2) else "none"}
 
 
# --------------------------------------------------------------------------
# data prep
# --------------------------------------------------------------------------
 
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Column names with spaces become underscores (e.g. 'diabetes prescriptions' -> 'diabetes_prescriptions')."""
    return df.rename(columns=lambda c: c.replace(" ", "_"))
 
 
def assert_variants_distinct(
    df: pd.DataFrame,
    cols: Sequence[str] = (
        "SAT_on_road_share_u",
        "SAT_on_road_share_w",
        "GSV_on_road_u",
        "GSV_on_road_w",
    ),
    max_r: float = 0.995,
) -> pd.DataFrame:
    """Guard against one greenery measure being silently copied into another.
 
    Returns the pairwise correlation matrix so it can be logged and reported.
    """
    present = [c for c in cols if c in df.columns]
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < 2:
                continue
            if np.allclose(pair[a], pair[b]):
                raise AssertionError(f"{a} and {b} are identical columns")
            r = float(pair.corr().iloc[0, 1])
            if r > max_r:
                raise AssertionError(f"{a} vs {b} suspiciously collinear (r={r:.4f})")
    return df[present].corr()
 
 
# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------
 
def logit_propensity(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unpenalized logistic PS on standardized covariates; returns logit(PS)."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(**_NO_PENALTY, max_iter=5000),
    )
    with warnings.catch_warnings():
        # sklearn 1.8 maps C=inf to penalty=None and then warns that C is ignored.
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore")
        p = model.fit(X, t).predict_proba(X)[:, 1]
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))
 
 
def _nearest(query: np.ndarray, pool: np.ndarray, caliper: float):
    """1-NN of each `query` point in `pool`; returns (indices, within_caliper)."""
    nn = NearestNeighbors(n_neighbors=1).fit(pool.reshape(-1, 1))
    dist, idx = nn.kneighbors(query.reshape(-1, 1))
    return idx.ravel(), dist.ravel() <= caliper
 
 
def psm_ate(
    y: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    caliper_sd: float = 0.2,
    details: bool = False,
):
    """1-NN propensity-score matching ATE, in the units of `y`.
 
    Each treated unit is matched to its nearest control on logit(PS) and vice
    versa; ATT and ATC are combined weighting by the number of *matched* units.
    """
    y, t, X = np.asarray(y, float), np.asarray(t, int), np.asarray(X, float)
    if t.min() == t.max():
        return (np.nan, None) if details else np.nan
 
    ps = logit_propensity(X, t)
    caliper = caliper_sd * ps.std(ddof=1)
    it, ic = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
 
    j_c, keep_t = _nearest(ps[it], ps[ic], caliper)   # treated -> control
    j_t, keep_c = _nearest(ps[ic], ps[it], caliper)   # control -> treated
    n1, n0 = int(keep_t.sum()), int(keep_c.sum())
    if n1 == 0 or n0 == 0:
        return (np.nan, None) if details else np.nan
 
    att = float((y[it[keep_t]] - y[ic[j_c[keep_t]]]).mean())
    atc = float((y[it[j_t[keep_c]]] - y[ic[keep_c]]).mean())
    ate = (att * n1 + atc * n0) / (n1 + n0)
 
    if not details:
        return ate
    return ate, {
        "att": att,
        "atc": atc,
        "treated_idx": np.concatenate([it[keep_t], it[j_t[keep_c]]]),
        "control_idx": np.concatenate([ic[j_c[keep_t]], ic[keep_c]]),
        "n_treated": len(it),
        "n_control": len(ic),
        "n_matched_treated": n1,
        "n_unmatched_treated": len(it) - n1,
    }
 
 
def balance(X: np.ndarray, t: np.ndarray, names: Sequence[str], matched: Optional[dict]):
    """Standardized mean differences before and after matching.
 
    Denominator is the pre-matching pooled SD throughout, so the two columns
    are directly comparable.  |SMD| < 0.1 is the usual target.  The matched
    sample pools treated->control and control->treated pairs, as in `psm_ate`.
    """
    X = np.asarray(X, float)
    xt, xc = X[t == 1], X[t == 0]
    pooled = np.sqrt((xt.var(0, ddof=1) + xc.var(0, ddof=1)) / 2)
    pooled[pooled == 0] = np.nan
    before = (xt.mean(0) - xc.mean(0)) / pooled
 
    if matched is None:
        after = np.full_like(before, np.nan)
    else:
        after = (
            X[matched["treated_idx"]].mean(0) - X[matched["control_idx"]].mean(0)
        ) / pooled
 
    return pd.DataFrame(
        {"covariate": list(names), "smd_before": before, "smd_after": after}
    )
 
 
def aipw_ate(y: np.ndarray, t: np.ndarray, X: np.ndarray, trim: float = 0.02) -> float:
    """Doubly-robust ATE cross-check.  The bootstrap is valid for this one."""
    y, t, X = np.asarray(y, float), np.asarray(t, int), np.asarray(X, float)
    ps = np.clip(1 / (1 + np.exp(-logit_propensity(X, t))), trim, 1 - trim)
    m1 = LinearRegression().fit(X[t == 1], y[t == 1]).predict(X)
    m0 = LinearRegression().fit(X[t == 0], y[t == 0]).predict(X)
    return float(np.mean(m1 - m0 + t * (y - m1) / ps - (1 - t) * (y - m0) / (1 - ps)))
 
 
# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------
 
def _resample(rng, n: int, blocks: Optional[np.ndarray]) -> np.ndarray:
    if blocks is None:
        return rng.integers(0, n, n)
    uniq = np.unique(blocks)
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([np.flatnonzero(blocks == b) for b in picked])
 
 
def _boot_once(y, t, X, blocks, caliper_sd, seed):
    """(PSM, AIPW) on one resample, so both estimators see the same draws."""
    ix = _resample(np.random.default_rng(seed), len(y), blocks)
    if t[ix].min() == t[ix].max():
        return np.nan, np.nan
    return psm_ate(y[ix], t[ix], X[ix], caliper_sd), aipw_ate(y[ix], t[ix], X[ix])


def _summarize(draws: np.ndarray, n_boot: int, alpha: float, name: str) -> dict:
    # Failed replicates (no match inside the caliper, one-arm resample) are not
    # random, so more than a few percent of them would bias the interval.
    valid = draws[np.isfinite(draws)]
    if len(valid) < 0.95 * n_boot:
        raise RuntimeError(f"{name}: only {len(valid)}/{n_boot} bootstrap replicates valid")
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "se": float(valid.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "boot_mean": float(valid.mean()),
        "n_boot_valid": int(len(valid)),
    }


def bootstrap_ci(
    y, t, X, *, blocks=None, n_boot=5000, caliper_sd=0.2, seed=0, n_jobs=-1,
    alpha=0.01,
):
    """Percentile bootstrap SE and (1-alpha) CI for both PSM and AIPW.

    Returns {"psm": {...}, "aipw": {...}}.  Blocks resample spatial units.
    """
    draws = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_boot_once)(y, t, X, blocks, caliper_sd, seed + i + 1)
        for i in range(n_boot)
    )
    draws = np.asarray(draws, float)
    return {
        "psm": _summarize(draws[:, 0], n_boot, alpha, "psm"),
        "aipw": _summarize(draws[:, 1], n_boot, alpha, "aipw"),
    }
 
 
# --------------------------------------------------------------------------
# one treatment x outcome cell
# --------------------------------------------------------------------------
 
def estimate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Sequence[str] = CONFOUNDERS,
    *,
    label: Optional[str] = None,
    block_col: Optional[str] = None,
    n_boot: int = 5000,
    caliper_sd: float = 0.2,
    seed: int = 0,
    n_jobs: int = -1,
    alpha: float = 0.01,
    with_balance: bool = False,
    threshold: Optional[float] = None,
):
    """ATE for one (treatment, outcome) pair, reported three ways.

    ATE_per_capita  absolute effect, in prescriptions per person per year
    ATE_pct         same effect as a % of the mean rate in below-median wards
                    (NaN if the outcome is not a non-negative rate)
    ATE_sd          same effect in SD units, for comparison across outcomes
    AIPW_*          doubly-robust estimate on the same bootstrap draws

    Treated = `treatment` > `threshold`; by default the citywide median over
    all rows of `df`, before listwise deletion.  Pass `threshold` when `df` is
    a subsample so the treatment keeps its full-sample meaning.
    """
    cols = list(confounders) + [treatment, outcome] + ([block_col] if block_col else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    citywide_median = float(df[treatment].median()) if threshold is None else float(threshold)
 
    work = df[cols].dropna()
    if len(work) < 20:
        raise ValueError(f"too few complete rows for {treatment} -> {outcome}")
 
    y = work[outcome].to_numpy(float)
    t = (work[treatment].to_numpy(float) > citywide_median).astype(int)
    X = work[list(confounders)].to_numpy(float)
    blocks = work[block_col].to_numpy() if block_col else None
 
    ate, detail = psm_ate(y, t, X, caliper_sd, details=True)
    if detail is None:
        raise ValueError(f"{treatment} -> {outcome}: no matched pairs inside the caliper")
    bal = balance(X, t, confounders, detail)
    boot = bootstrap_ci(
        y, t, X, blocks=blocks, n_boot=n_boot, caliper_sd=caliper_sd,
        seed=seed, n_jobs=n_jobs, alpha=alpha,
    )
    unc, unc_dr = boot["psm"], boot["aipw"]

    baseline = float(y[t == 0].mean())    # mean rate in below-median wards
    sd_y = float(y.std(ddof=1))
    # A % change needs a positive rate; standardized outcomes have mean ~0.
    scale_pct = 100.0 / baseline if (y.min() >= 0 and baseline > 0) else np.nan
    scale_sd = 1.0 / sd_y
 
    row = {
        "Treatment": treatment,
        "Label": label or treatment,
        "Outcome": outcome,
        "N": len(work),
        "N_treated": int(t.sum()),
        "N_control": int(len(t) - t.sum()),
        "N_unmatched_treated": detail["n_unmatched_treated"],
        "Citywide_median": citywide_median,
        "Baseline_control_mean": baseline,
        "ATE_per_capita": ate,
        "ATT_per_capita": detail["att"],
        "ATC_per_capita": detail["atc"],   # effect for below-median areas; drives the projection
        "SE_per_capita": unc["se"],
        "CI_low_per_capita": unc["ci_low"],
        "CI_high_per_capita": unc["ci_high"],
        "ATE_pct": ate * scale_pct,
        "SE_pct": unc["se"] * scale_pct,
        "ATE_sd": ate * scale_sd,
        "SE_sd": unc["se"] * scale_sd,
        "Significant": bool(unc["ci_low"] > 0 or unc["ci_high"] < 0),
        "Max_abs_SMD_after": float(np.nanmax(np.abs(bal["smd_after"]))),
        "Balanced": bool(np.nanmax(np.abs(bal["smd_after"])) < 0.1),
        "AIPW_per_capita": aipw_ate(y, t, X),
        "AIPW_SE_per_capita": unc_dr["se"],
        "AIPW_CI_low_per_capita": unc_dr["ci_low"],
        "AIPW_CI_high_per_capita": unc_dr["ci_high"],
        "AIPW_Significant": bool(unc_dr["ci_low"] > 0 or unc_dr["ci_high"] < 0),
        "N_boot_valid": unc["n_boot_valid"],
    }
    return (row, bal) if with_balance else row
 
 
def run_grid(
    df: pd.DataFrame,
    treatments: Iterable[tuple[str, str]] = TREATMENTS,
    outcomes: Sequence[str] = OUTCOMES,
    confounders: Sequence[str] = CONFOUNDERS,
    *,
    block_col: Optional[str] = None,
    n_boot: int = 5000,
    caliper_sd: float = 0.2,
    seed: int = 0,
    n_jobs: int = -1,
    alpha: float = 0.01,
    verbose: bool = True,
    thresholds: Optional[dict] = None,
) -> pd.DataFrame:
    """Every (treatment, outcome) pair; one row each, tested at level `alpha` (and FDR at `alpha`).

    `thresholds` maps treatment column -> cut-off, for subsamples (see `estimate`).
    """
    data = rename_columns(df)
    assert_variants_distinct(data)
    thresholds = thresholds or {}

    rows = []
    for treatment, label in treatments:
        if treatment not in data.columns:
            if verbose:
                print(f"skip missing treatment: {treatment}")
            continue
        for outcome in outcomes:
            row = estimate(
                data, treatment, outcome, confounders, label=label,
                block_col=block_col, n_boot=n_boot, caliper_sd=caliper_sd,
                seed=seed, n_jobs=n_jobs, alpha=alpha,
                threshold=thresholds.get(treatment),
            )
            if verbose:
                flag = "*" if row["Significant"] else " "
                bal = "" if row["Balanced"] else "  [UNBALANCED]"
                print(
                    f"{flag} {label:<22} {outcome:<28} "
                    f"ATE={row['ATE_per_capita']:8.3f}  "
                    f"({row['ATE_pct']:6.2f}%)  SE={row['SE_per_capita']:.3f}{bal}",
                    flush=True,
                )
            rows.append(row)
 
    out = pd.DataFrame(rows)
    return add_fdr(out, alpha) if len(out) else out
 
 
def add_fdr(
    results: pd.DataFrame,
    alpha: float = 0.01,
    est: str = "ATE_per_capita",
    se: str = "SE_per_capita",
) -> pd.DataFrame:
    """Benjamini-Hochberg flag across every (treatment, outcome) cell in `results`.

    p-values use a normal approximation (est / se), so for the PSM grid
    `Significant_FDR` can disagree with the percentile-CI `Significant`.
    """
    z = results[est] / results[se]
    p = 2 * (1 - _norm_cdf(np.abs(z)))
    order = np.argsort(p)
    passed = np.zeros(len(p), bool)
    below = np.flatnonzero(np.sort(p) <= alpha * np.arange(1, len(p) + 1) / len(p))
    if len(below):
        passed[order[: below[-1] + 1]] = True
    out = results.copy()
    out["p_approx"], out["Significant_FDR"] = p, passed
    return out
 
 
def _norm_cdf(x):
    from math import erf, sqrt
    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in np.atleast_1d(x)])
 
 
# --------------------------------------------------------------------------
# continuous treatment: partially linear model, double machine learning
# --------------------------------------------------------------------------

def _nuisance(learner: str, seed: int):
    """Out-of-fold learner for E[y|X] and E[d|X]: "linear" (OLS) or "forest" (random forest, the paper's)."""
    if learner == "linear":
        return LinearRegression()
    if learner == "forest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1
        )
    raise ValueError(f"unknown learner: {learner}")


def plr_effect(
    y: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    *,
    learner: str = "forest",
    n_folds: int = 5,
    n_rep: int = 5,
    seed: int = 0,
) -> dict:
    """Slope theta in  y = theta * d + g(X) + e,  by cross-fitted double ML.

    g(X) and E[d | X] are learned out-of-fold; theta is the regression of
    the outcome residual on the treatment residual (Robinson 1988;
    Chernozhukov et al. 2018, Econometrics Journal 21(1): C1-C68).  The SE is
    the influence-function sandwich.  Over `n_rep` random fold splits, theta
    is the median and the variance includes the split-to-split spread.
    """
    y, d, X = np.asarray(y, float), np.asarray(d, float), np.asarray(X, float)
    n = len(y)
    thetas, variances, d_r2 = [], [], []
    for r in range(n_rep):
        u, v = np.empty(n), np.empty(n)
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for train, test in folds.split(X):
            u[test] = y[test] - _nuisance(learner, seed + r).fit(X[train], y[train]).predict(X[test])
            v[test] = d[test] - _nuisance(learner, seed + r).fit(X[train], d[train]).predict(X[test])
        theta = float(u @ v / (v @ v))
        psi = (u - theta * v) * v
        thetas.append(theta)
        variances.append(np.mean(psi ** 2) / np.mean(v ** 2) ** 2 / n)
        d_r2.append(1 - np.mean(v ** 2) / np.var(d))
    thetas, variances = np.asarray(thetas), np.asarray(variances)
    theta = float(np.median(thetas))
    r2 = float(np.mean(d_r2))
    if r2 < -0.1:
        # Out-of-fold predictions worse than the mean: the learner is extrapolating.
        raise RuntimeError(f"learner {learner!r} unstable (treatment R2 = {r2:.2f})")
    return {
        "theta": theta,
        "se": float(np.sqrt(np.median(variances + (thetas - theta) ** 2))),
        "treatment_r2": r2,
    }


def estimate_continuous(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Sequence[str] = CONFOUNDERS,
    *,
    label: Optional[str] = None,
    learner: str = "forest",
    n_folds: int = 5,
    n_rep: int = 5,
    seed: int = 0,
    alpha: float = 0.01,
) -> dict:
    """Effect of a 1-SD increase in greenery, for one (treatment, outcome) pair.

    Effect_per_SD   prescriptions per person per year, per 1 SD of greenery
    Effect_pct      same, as % of the mean prescription rate
    Effect_sd       same, in SD units of the outcome
    Linear_*        same model with linear g(X), i.e. OLS with the confounders
    Treatment_R2    share of greenery variance explained by the confounders;
                    close to 1 means little independent variation to learn from

    `learner` sets how flexibly the confounders are adjusted for: "linear" or
    "forest" (random forest, allows interactions; the paper's).  The slope is
    assumed constant across the greenery distribution.
    """
    cols = list(confounders) + [treatment, outcome]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    work = df[cols].dropna()
    if len(work) < 20:
        raise ValueError(f"too few complete rows for {treatment} -> {outcome}")

    y = work[outcome].to_numpy(float)
    d_raw = work[treatment].to_numpy(float)
    sd_d = float(d_raw.std(ddof=1))
    d = (d_raw - d_raw.mean()) / sd_d
    X = work[list(confounders)].to_numpy(float)

    fit = plr_effect(y, d, X, learner=learner, n_folds=n_folds, n_rep=n_rep, seed=seed)
    lin = plr_effect(y, d, X, learner="linear", n_folds=n_folds, n_rep=1, seed=seed)

    z_crit = norm.ppf(1 - alpha / 2)
    mean_y = float(y.mean())
    scale_pct = 100.0 / mean_y if (y.min() >= 0 and mean_y > 0) else np.nan
    scale_sd = 1.0 / float(y.std(ddof=1))
    p = 2 * norm.sf(abs(fit["theta"] / fit["se"]))
    p_lin = 2 * norm.sf(abs(lin["theta"] / lin["se"]))

    return {
        "Treatment": treatment,
        "Label": label or treatment,
        "Outcome": outcome,
        "N": len(work),
        "Learner": learner,
        "Treatment_SD": sd_d,
        "Outcome_mean": mean_y,
        "Effect_per_SD": fit["theta"],
        "SE_per_SD": fit["se"],
        "CI_low_per_SD": fit["theta"] - z_crit * fit["se"],
        "CI_high_per_SD": fit["theta"] + z_crit * fit["se"],
        "p": float(p),
        "Significant": bool(p < alpha),
        "Effect_pct": fit["theta"] * scale_pct,
        "SE_pct": fit["se"] * scale_pct,
        "Effect_sd": fit["theta"] * scale_sd,
        "SE_sd": fit["se"] * scale_sd,
        "Linear_per_SD": lin["theta"],
        "Linear_SE_per_SD": lin["se"],
        "Linear_Significant": bool(p_lin < alpha),
        "Treatment_R2": fit["treatment_r2"],
    }


def run_grid_continuous(
    df: pd.DataFrame,
    treatments: Iterable[tuple[str, str]] = TREATMENTS,
    outcomes: Sequence[str] = OUTCOMES,
    confounders: Sequence[str] = CONFOUNDERS,
    *,
    learner: str = "forest",
    n_folds: int = 5,
    n_rep: int = 5,
    seed: int = 0,
    alpha: float = 0.01,
    verbose: bool = True,
) -> pd.DataFrame:
    """Every (treatment, outcome) pair with greenery as a continuous treatment."""
    data = rename_columns(df)
    assert_variants_distinct(data)

    rows = []
    for treatment, label in treatments:
        if treatment not in data.columns:
            if verbose:
                print(f"skip missing treatment: {treatment}")
            continue
        for outcome in outcomes:
            row = estimate_continuous(
                data, treatment, outcome, confounders, label=label, learner=learner,
                n_folds=n_folds, n_rep=n_rep, seed=seed, alpha=alpha,
            )
            if verbose:
                flag = "*" if row["Significant"] else " "
                print(
                    f"{flag} {label:<22} {outcome:<28} "
                    f"per SD={row['Effect_per_SD']:8.3f}  "
                    f"({row['Effect_pct']:6.2f}%)  SE={row['SE_per_SD']:.3f}  "
                    f"R2(T|X)={row['Treatment_R2']:.2f}",
                    flush=True,
                )
            rows.append(row)

    out = pd.DataFrame(rows)
    return add_fdr(out, alpha, est="Effect_per_SD", se="SE_per_SD") if len(out) else out


# --------------------------------------------------------------------------
# table output
# --------------------------------------------------------------------------
 
def table1(
    results: pd.DataFrame,
    value: str = "ATE_per_capita",
    err: str = "SE_per_capita",
    sig: str = "Significant",
) -> pd.DataFrame:
    """Wide treatments x outcomes view, cells formatted as -3.68±.33 (bold = significant)."""
    def cell(r):
        v = round(r[value], 2) + 0.0          # + 0.0 turns -0.00 into 0.00
        txt = f"{v:.2f}±{r[err]:.2f}".replace("±0.", "±.")
        return f"**{txt}**" if r[sig] else txt

    tmp = results.copy()
    dup = tmp.duplicated(["Label", "Outcome"])
    if dup.any():
        raise ValueError(f"repeated (Label, Outcome) rows, e.g. {tmp.loc[dup, ['Label', 'Outcome']].iloc[0].tolist()}")
    tmp["cell"] = tmp.apply(cell, axis=1)
    seen = list(dict.fromkeys(tmp["Outcome"]))
    cols = [o for o in OUTCOMES if o in seen] + [o for o in seen if o not in OUTCOMES]
    wide = tmp.pivot(index="Label", columns="Outcome", values="cell")
    wide = wide.reindex(index=list(dict.fromkeys(results["Label"])), columns=cols)
    return wide.rename(columns=lambda c: c.replace("_prescriptions", ""))
 

