"""Composite prospect index for ML niches, scored WITHIN each sector.

The same five components as the sister industry project, min-max
normalised within each sector (Healthcare niches compete against
Healthcare niches; Agriculture against Agriculture):

    growth_momentum      forecast niche-market CAGR, 2025 -> 2030   30%
    market_scale         forecast 2030 niche market size ($B)       25%
    investment_momentum  forecast investment CAGR                   20%
    job_demand           forecast job-postings CAGR                 15%
    ai_maturity          projected 2030 ML adoption in the niche    10%
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENT_WEIGHTS: dict[str, float] = {
    "growth_momentum": 0.30,
    "market_scale": 0.25,
    "investment_momentum": 0.20,
    "job_demand": 0.15,
    "ai_maturity": 0.10,
}

COMPONENT_LABELS: dict[str, str] = {
    "growth_momentum": "Growth momentum (market CAGR)",
    "market_scale": "Niche scale (2030 $B)",
    "investment_momentum": "Investment momentum",
    "job_demand": "Job demand growth",
    "ai_maturity": "AI maturity (2030 adoption)",
}


def _minmax(values: np.ndarray) -> np.ndarray:
    mn, mx = values.min(), values.max()
    if mx - mn < 1e-12:
        return np.full(len(values), 0.5)
    return (values - mn) / (mx - mn)


def build_score_table(
    raw: dict[str, dict[str, float]], sectors: dict[str, str]
) -> pd.DataFrame:
    """Within-sector composite scores. raw = {niche: {component: value}}."""
    niches = list(raw.keys())
    components = list(COMPONENT_WEIGHTS.keys())

    raw_df = pd.DataFrame(
        {n: [raw[n][c] for c in components] for n in niches},
        index=components,
    ).T
    raw_df.index.name = "niche"

    rows = []
    for sector in sorted(set(sectors.values())):
        members = [n for n in niches if sectors[n] == sector]
        sub = raw_df.loc[members]
        norm = pd.DataFrame(index=sub.index)
        for comp in components:
            norm[comp] = _minmax(sub[comp].to_numpy())
        weights = np.array([COMPONENT_WEIGHTS[c] for c in components])
        scores = (norm[components].to_numpy() * weights).sum(axis=1) * 100.0

        for j, niche in enumerate(members):
            entry = {
                "niche": niche,
                "sector": sector,
                "score": round(float(scores[j]), 1),
            }
            for comp in components:
                entry[comp] = float(sub.loc[niche, comp])
                entry[f"{comp}_norm"] = float(norm.loc[niche, comp])
            rows.append(entry)

    out = pd.DataFrame(rows).set_index("niche")
    out["rank_sector"] = (
        out.groupby("sector")["score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    out = out.sort_values(["sector", "rank_sector"])
    out.index.name = "niche"
    return out
