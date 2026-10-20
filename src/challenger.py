"""Supervised challenger-detection model (the core new ML layer).

Task
----
Predict which ML niches are *challengers* — the ones breaking out and
taking within-sector share — using ONLY early-window signals.

Temporal design (no leakage):
    features : computed from the 2015-2019 panel window only
    label    : within-sector market-share gain from 2019 to 2025,
               thresholded at the sector median (top half = challenger)

Candidate models, compared by leave-one-out cross-validated AUC:
    1. L2 logistic regression on standardized features (interpretable)
    2. Random forest (small, regularised: depth 3, 500 trees)

The champion is refit on all niches and emits a challenger probability
for each niche, which the pipeline pairs with the 2026-2030 prospect
score from the forecasting layer.

CAVEAT: the bundled panel is benchmark-calibrated synthetic data, so the
classifier demonstrates the *method*; its AUC will be optimistic because
the features are computed from the same generating process that produced
the labels. On real data the identical pipeline runs unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_LABELS: dict[str, str] = {
    "cagr_early": "Early growth (CAGR '15\u2013'19)",
    "accel": "Growth acceleration",
    "invest_intensity": "Investment intensity",
    "invest_accel": "Investment acceleration",
    "patents_cagr": "Patent velocity",
    "adoption_2019": "Adoption level (2019)",
    "adoption_slope": "Adoption slope '15\u2013'19",
    "share_2019": "Within-sector share (2019)",
    "share_gain_early": "Early share gain '15\u2013'19",
}


def _cagr(years: np.ndarray, values: np.ndarray) -> float:
    """Log-linear fitted compound growth rate of a short series."""
    years = np.asarray(years, float)
    values = np.asarray(values, float)
    if len(values) < 3 or np.any(values <= 0):
        return 0.0
    slope, _ = np.polyfit(years, np.log(values), 1)
    return float(np.exp(slope) - 1.0)


def build_features(panel: pd.DataFrame, early_end: int = 2019) -> pd.DataFrame:
    """Nine early-window (<= 2019) signals per niche. No future data."""
    early = panel[panel["year"] <= early_end].copy()
    mid = early_end - 2  # split point for acceleration contrasts

    rows: dict[str, dict] = {}
    for niche, sub in early.groupby("niche"):
        sub = sub.sort_values("year")
        y = sub["year"].to_numpy()
        mkt = sub["market_size_usd_b"].to_numpy()
        inv = sub["investment_usd_b"].to_numpy()
        pat = sub["ai_patents_k"].to_numpy()
        adopt = sub["adoption_rate_pct"].to_numpy()

        pre = sub[sub["year"] <= mid]
        post = sub[sub["year"] > mid]
        invest_ratio = inv / mkt

        rows[niche] = {
            "cagr_early": _cagr(y, mkt),
            "accel": _cagr(post["year"], post["market_size_usd_b"])
            - _cagr(pre["year"], pre["market_size_usd_b"]),
            "invest_intensity": float(invest_ratio.mean()),
            "invest_accel": _cagr(post["year"], post["investment_usd_b"])
            - _cagr(pre["year"], pre["investment_usd_b"]),
            "patents_cagr": _cagr(y, pat),
            "adoption_2019": float(adopt[-1]),
            "adoption_slope": float((adopt[-1] - adopt[0]) / max(len(adopt) - 1, 1)),
            "share_2019": 0.0,          # filled below (needs sector totals)
            "share_gain_early": 0.0,     # filled below
        }

    # within-sector shares in the early window
    totals: dict[int, dict[str, float]] = {
        yr: sub.groupby("sector")["market_size_usd_b"].sum().to_dict()
        for yr, sub in early.groupby("year")
    }
    for niche, sub in early.groupby("niche"):
        sector = sub["sector"].iloc[0]
        s15 = float(
            sub[sub["year"] == early["year"].min()]["market_size_usd_b"].iloc[0]
            / totals[early["year"].min()][sector]
        )
        s19 = float(
            sub[sub["year"] == early_end]["market_size_usd_b"].iloc[0]
            / totals[early_end][sector]
        )
        rows[niche]["share_2019"] = s19
        rows[niche]["share_gain_early"] = s19 - s15

    return pd.DataFrame(rows).T


def build_labels(
    panel: pd.DataFrame, t0: int = 2019, t1: int = 2025
) -> tuple[pd.Series, pd.Series]:
    """Outcome: within-sector share gain t0 -> t1; label = above sector median."""
    gain: dict[str, float] = {}
    for niche, sub in panel.groupby("niche"):
        sector = sub["sector"].iloc[0]
        vals = {}
        for yr in (t0, t1):
            sector_total = float(
                panel[(panel["year"] == yr) & (panel["sector"] == sector)][
                    "market_size_usd_b"
                ].sum()
            )
            vals[yr] = float(
                sub[sub["year"] == yr]["market_size_usd_b"].iloc[0] / sector_total
            )
        gain[niche] = vals[t1] - vals[t0]

    gain_s = pd.Series(gain, name="share_gain")
    sectors = panel.groupby("niche")["sector"].first()
    label = pd.Series(index=gain_s.index, dtype=int)
    for niche in gain_s.index:
        sector = sectors[niche]
        sector_gains = gain_s[sectors == sector]
        label[niche] = int(gain_s[niche] > sector_gains.median())
    return label, gain_s


def train_challenger_models(
    X: pd.DataFrame, y: pd.Series, seed: int = 42
) -> dict:
    """LOOCV-compare logistic regression vs random forest; refit champion."""
    logit = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(C=0.5, max_iter=2000, random_state=seed),
            ),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=500, max_depth=3, random_state=seed, n_jobs=-1
    )
    models = {"Logistic Regression": logit, "Random Forest": forest}

    result: dict = {
        "auc": {},
        "oof_probs": {},
        "roc": {},
        "accuracy": {},
        "importances": {},
    }

    if y.nunique() < 2:
        # degenerate labels: no discrimination possible
        result["champion_name"] = "Logistic Regression"
        result["auc"] = {k: float("nan") for k in models}
        probs = pd.Series(0.5, index=X.index)
        result["probs"] = probs
        result["roc"] = {k: (np.array([0, 1]), np.array([0, 1])) for k in models}
        result["importances"] = {
            k: pd.Series(0.0, index=X.columns) for k in models
        }
        result["oof_probs"] = {k: probs.copy() for k in models}
        result["accuracy"] = {k: float("nan") for k in models}
        return result

    cv = LeaveOneOut()
    for name, model in models.items():
        oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        oof_s = pd.Series(oof, index=X.index)
        auc = float(roc_auc_score(y, oof_s))
        acc = float(((oof_s >= 0.5).astype(int) == y).mean())
        fpr, tpr, _ = roc_curve(y, oof_s)
        model.fit(X, y)
        result["auc"][name] = auc
        result["accuracy"][name] = acc
        result["oof_probs"][name] = oof_s
        result["roc"][name] = (fpr, tpr)
        if name == "Logistic Regression":
            coefs = model.named_steps["clf"].coef_[0]
            result["importances"][name] = pd.Series(
                coefs, index=X.columns
            ).sort_values(key=np.abs, ascending=False)
        else:
            result["importances"][name] = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

    champion = max(result["auc"], key=result["auc"].get)
    result["champion_name"] = champion
    model = models[champion]
    model.fit(X, y)
    result["probs"] = pd.Series(
        model.predict_proba(X)[:, 1], index=X.index, name="challenger_prob"
    )
    return result


def verdict_from_prob(p: float) -> str:
    if p >= 0.6:
        return "Predicted challenger"
    if p >= 0.4:
        return "Watch list"
    return "Established / holding"
