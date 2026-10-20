"""Auto-generated markdown report for the niche challenger project."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .challenger import FEATURE_LABELS


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]

    def fmt_row(cells: list) -> str:
        return "| " + " | ".join(
            f"{str(c):<{widths[i]}}" for i, c in enumerate(cells)
        ) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _money(v: float) -> str:
    if v >= 1:
        return f"${v:,.2f}B"
    return f"${v * 1000:,.0f}M"


def generate_report(
    panel: pd.DataFrame,
    results: dict,
    scores: pd.DataFrame,
    preds: pd.DataFrame,
    model_info: dict,
    features: pd.DataFrame,
    outpath: Path,
    horizon: int,
    last_year: int,
    final_year: int,
) -> None:
    champion = model_info["champion_name"]
    auc_champ = model_info["auc"][champion]
    niches = list(preds.index)

    # ------------------------------------------------------ predicted lists
    def challengers_of(sector: str) -> list[str]:
        block = preds[preds["sector"] == sector]
        return [
            n for n in block.index if block.loc[n, "challenger_prob"] >= 0.5
        ]

    hc_ch, ag_ch = challengers_of("Healthcare"), challengers_of("Agriculture")

    def fmt_prob_list(names: list[str]) -> str:
        if not names:
            return "—"
        return ", ".join(
            f"**{n}** ({preds.loc[n, 'challenger_prob']:.2f})" for n in names
        )

    # ---------------------------------------------------------------- tables
    pool_rows = []
    sector_totals = (
        panel[panel["year"] == last_year].groupby("sector")["market_size_usd_b"].sum()
    )
    for niche in niches:
        sector = preds.loc[niche, "sector"]
        m25 = float(
            panel[(panel["niche"] == niche) & (panel["year"] == last_year)][
                "market_size_usd_b"
            ].iloc[0]
        )
        pool_rows.append(
            [niche, sector, _money(m25), f"{m25 / sector_totals[sector] * 100:.1f}%"]
        )
    pool_table = _md_table(
        ["Niche", "Sector", f"{last_year} market", "% of sector"], pool_rows
    )

    auc_rows = []
    for name in ("Logistic Regression", "Random Forest"):
        auc_rows.append(
            [
                name + (" *(champion)*" if name == champion else ""),
                f"{model_info['auc'][name]:.3f}",
                f"{model_info['accuracy'][name] * 100:.0f}%",
            ]
        )
    auc_table = _md_table(
        ["Model", "LOOCV AUC", "LOOCV accuracy @ 0.5"], auc_rows
    )

    top_features = model_info["importances"]["Logistic Regression"].head(5)
    feat_rows = [
        [
            FEATURE_LABELS.get(f, f),
            f"{v:+.2f}",
            "pushes toward challenger" if v > 0 else "pushes toward established",
        ]
        for f, v in top_features.items()
    ]
    feat_table = _md_table(["Feature (2015\u20132019 signals)", "Coef", "Reading"], feat_rows)

    pred_rows = []
    for niche in preds.sort_values(
        ["sector", "challenger_prob"], ascending=[True, False]
    ).index:
        r = preds.loc[niche]
        pred_rows.append(
            [
                niche, r["sector"],
                f"{r['challenger_prob']:.2f}", r["verdict"],
                "\u25b2 broke out" if r["label_true"] == 1 else "held",
                f"{r['share_gain'] * 100:+.1f} pp",
                f"{r['score']:.1f}",
            ]
        )
    pred_table = _md_table(
        ["Niche", "Sector", "P(challenger)", "Model verdict",
         "Actual 2019\u21922025", "Share gain", "Prospect score"],
        pred_rows,
    )

    fc_rows = []
    for niche in niches:
        res = results[(niche, "market_size_usd_b")]
        fc_rows.append(
            [
                niche, preds.loc[niche, "sector"],
                _money(float(res.hist_values[-1])),
                _money(float(res.point[-1])),
                f"[{_money(float(res.lo[-1]))} \u2013 {_money(float(res.hi[-1]))}]",
                _pct(res.horizon_cagr),
            ]
        )
    fc_table = _md_table(
        ["Niche", "Sector", f"{last_year} actual", f"{final_year} forecast",
         "90% CI", "Implied CAGR"],
        fc_rows,
    )

    md = f"""# The NEXT Challenger Niches inside ML
### Healthcare & Agriculture \u2014 a supervised challenger-detection model, {last_year + 1}\u2013{final_year}

*Generated automatically by `main.py` \u00b7 18 ML niches (10 healthcare, 8
agriculture) \u00b7 panel {int(panel['year'].min())}\u2013{last_year} \u00b7 forecast
horizon {horizon} years*

---

## 1. Executive Summary

A supervised classifier reads **only 2015\u20132019 signals** from each niche and
predicts which niches *break out* (gain within-sector share through {last_year}).
The champion model — **{champion}** — reaches a **leave-one-out AUC of
{auc_champ:.2f}** on held-out niches.

**Predicted challengers in Healthcare:** {fmt_prob_list(hc_ch)}

**Predicted challengers in Agriculture:** {fmt_prob_list(ag_ch)}

Pairing the classifier with the {last_year + 1}\u2013{final_year} ensemble
forecast, the confirmed challengers (high probability **and** high prospect
score) are:

{chr(10).join(f"- **{n}** \u2014 P(challenger) = {preds.loc[n, 'challenger_prob']:.2f}, prospect score {preds.loc[n, 'score']:.1f}, forecast CAGR {results[(n, 'market_size_usd_b')].horizon_cagr:+.1%}" for n in sorted(set(hc_ch + ag_ch), key=lambda x: -preds.loc[x, 'score']))}

**The question this project answers:** *inside Healthcare and Agriculture,
which ML/DS niches are the rising challengers \u2014 and can a model detect them
from early signals alone?*

---

## 2. The Niche Pool

{pool_table}

## 3. Method

**Temporal design (no leakage).** Features are computed from the
2015\u20132019 window only. The label is the outcome: within-sector market-share
gain from 2019 to {last_year}, thresholded at the sector median (top half =
challenger). The model therefore answers: *which early-window signals
separate niches that went on to break out?*

**Features (9).** Early growth (log-linear CAGR '15\u2013'19), growth
acceleration (CAGR '17\u2013'19 minus CAGR '15\u2013'17), investment intensity
(invest/market), investment acceleration, patent velocity, 2019 adoption
level, adoption slope, within-sector share (2019), early share gain.

**Models.** L2 logistic regression on standardised features (interpretable)
vs a regularised random forest (500 trees, depth 3). Both evaluated by
**leave-one-out cross-validation** (18 folds); the champion is refit on all
niches to emit final probabilities.

**Forecasting layer.** The same backtested ensemble as the sister industry
project: log-linear OLS growth regression + damped Holt exponential
smoothing, weights \u221d 1/MAPE from a 3-year holdout, 90% mixture-bootstrap
intervals; adoption via logistic S-curves. Prospect scores are min-max
normalised **within each sector**.

## 4. Model Card

{auc_table}

**Top signals (logistic coefficients, standardised):**

{feat_table}

> **Honest caveat:** the bundled panel is benchmark-calibrated synthetic data,
> so the features and labels come from the same generating process \u2014 the AUC
> is optimistic and demonstrates the *method*, not real-world skill. On real
> data the identical pipeline runs via `--from-csv`; expect AUC to fall and
> feature importances to become genuinely informative.

## 5. Predictions \u2014 all 18 niches

{pred_table}

![Challenger probability](../charts/01_challenger_probability.png)
![Verdict map](../charts/02_prob_vs_score.png)
![Feature importance](../charts/04_feature_importance.png)
![ROC curves](../charts/05_roc_curves.png)

## 6. Market Forecasts ({last_year + 1}\u2013{final_year})

{fc_table}

![Niche trajectories](../charts/03_market_trajectories.png)

## 7. Limitations

- **Synthetic calibrated inputs** \u2014 the classifier learns the data-generating
  process; treat AUC as an upper bound, and re-run on real panels before
  drawing conclusions.
- **n = 18 niches** \u2014 LOOCV is honest but high-variance at this size; the
  confidence interval on AUC is wide.
- **Label is relative** \u2014 "above sector-median share gain" defines
  challengers *within* this pool; different thresholds give different
  class balance.
- **Forecast layer** inherits the sister project's caveats: short series,
  iid bootstrap assumption, fixed adoption ceilings, no shock modelling.
- **Niche boundaries overlap** (e.g. ambient documentation touches admin
  automation) \u2014 some double counting is unavoidable.

## 8. Reproduce

```bash
pip install -r requirements.txt
python main.py                                  # full pipeline
python main.py --horizon 8 --seed 7             # different horizon / seed
python main.py --from-csv data/niche_panel.csv  # real data, same schema
```

## 9. Calibration References

- Grand View Research \u2014 *AI in Medical Imaging*, *AI in Drug Discovery*,
  *AI in Genomics/Precision Medicine* market reports
- CB Insights / PitchBook \u2014 ambient-AI scribe funding wave (2023\u201325)
- McKinsey \u2014 *The State of AI* (healthcare function adoption)
- MarketsandMarkets / Precedence \u2014 *Precision Agriculture*, *Agricultural
  Robots*, *Livestock Monitoring* market estimates
- Climate-tech MRV market reporting (soil carbon, Scope-3 verification)
- US BLS \u2014 *Data Scientists* occupational outlook (jobs calibration)
- Sister project: *The NEXT Top Industries for DS/ML* (same forecasting engine)
"""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(md, encoding="utf-8")
