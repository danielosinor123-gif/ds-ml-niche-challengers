"""Pipeline entry point for the niche challenger project.

Stages:
    1. build (or load) the niche panel (18 ML niches x 2 sectors)
    2. ensemble-forecast market size, investment, job postings 2026-2030
    3. fit logistic S-curves to niche adoption and project forward
    4. score niches WITHIN each sector on the composite prospect index
    5. train the challenger classifier (early signals -> breakout label),
       LOOCV-compare logistic regression vs random forest
    6. render charts
    7. write the markdown report

Usage:
    python main.py
    python main.py --horizon 8 --seed 7
    python main.py --from-csv data/niche_panel.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

from src import visuals
from src.challenger import build_features, build_labels, train_challenger_models, verdict_from_prob
from src.data import build_panel, load_panel_from_csv
from src.forecast import ensemble_forecast, fit_logistic, logistic_value
from src.report import generate_report
from src.scoring import build_score_table

FORECAST_METRICS = ["market_size_usd_b", "investment_usd_b", "job_postings_k"]

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
REPORT_DIR = OUT_DIR / "reports"


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _enable_utf8_console() -> None:
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict challenger ML niches inside Healthcare & Agriculture."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--from-csv", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _enable_utf8_console()
    for d in (DATA_DIR, CHART_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- 1. data
    _banner("[1/7] Building the niche panel dataset")
    if args.from_csv:
        panel = load_panel_from_csv(args.from_csv)
        print(f"  loaded external panel: {args.from_csv}")
    else:
        panel = build_panel(seed=args.seed)
        print(
            f"  {panel['niche'].nunique()} ML niches "
            f"({panel[panel['sector'] == 'Healthcare']['niche'].nunique()} healthcare, "
            f"{panel[panel['sector'] == 'Agriculture']['niche'].nunique()} agriculture)"
        )
    panel.to_csv(DATA_DIR / "niche_panel.csv", index=False)
    print(f"  saved: data/niche_panel.csv  ({len(panel)} rows)")

    niches = list(panel["niche"].unique())
    sectors = panel.groupby("niche")["sector"].first().to_dict()
    last_year = int(panel["year"].max())
    final_year = last_year + args.horizon

    # -------------------------------------------------------- 2. forecasts
    _banner(f"[2/7] Ensemble forecasting {last_year + 1}-{final_year}")
    results: dict = {}
    seed_counter = 0
    for niche in niches:
        sub = panel[panel["niche"] == niche].sort_values("year")
        for metric in FORECAST_METRICS:
            seed_counter += 1
            results[(niche, metric)] = ensemble_forecast(
                niche=niche,
                metric=metric,
                years=sub["year"].to_numpy(),
                values=sub[metric].to_numpy(),
                horizon=args.horizon,
                seed=args.seed * 1000 + seed_counter,
            )
    print(f"  forecast {len(results)} series (18 niches x 3 metrics)")

    # -------------------------------------------------------- 3. adoption
    _banner("[3/7] Fitting adoption S-curves")
    adoption_proj: dict = {}
    for niche in niches:
        sub = panel[panel["niche"] == niche].sort_values("year")
        ceiling, k, midpoint, _ = fit_logistic(
            sub["year"].to_numpy(), sub["adoption_rate_pct"].to_numpy()
        )
        adoption_proj[niche] = logistic_value(ceiling, k, midpoint, final_year)
    print(f"  projected adoption for {len(adoption_proj)} niches")

    # --------------------------------------------------------- 4. scoring
    _banner("[4/7] Composite prospect scores (within sector)")
    raw = {}
    for niche in niches:
        raw[niche] = {
            "growth_momentum": results[(niche, "market_size_usd_b")].horizon_cagr,
            "market_scale": float(results[(niche, "market_size_usd_b")].point[-1]),
            "investment_momentum": results[(niche, "investment_usd_b")].horizon_cagr,
            "job_demand": results[(niche, "job_postings_k")].horizon_cagr,
            "ai_maturity": adoption_proj[niche],
        }
    scores = build_score_table(raw, sectors=sectors)
    scores.to_csv(OUT_DIR / "scores.csv")

    forecast_rows = []
    for (niche, metric), res in results.items():
        for i, year in enumerate(res.years):
            forecast_rows.append(
                {
                    "niche": niche, "metric": metric, "year": year,
                    "point": round(float(res.point[i]), 4),
                    "ci90_lo": round(float(res.lo[i]), 4),
                    "ci90_hi": round(float(res.hi[i]), 4),
                }
            )
    pd.DataFrame(forecast_rows).to_csv(OUT_DIR / "forecasts.csv", index=False)
    print("  saved: outputs/scores.csv, outputs/forecasts.csv")

    # ------------------------------------------------- 5. challenger model
    _banner("[5/7] Training the challenger-detection classifier")
    features = build_features(panel)
    labels, share_gain = build_labels(panel)
    model_info = train_challenger_models(features, labels, seed=args.seed)

    for name, auc in model_info["auc"].items():
        print(f"  {name:<22} LOOCV AUC = {auc:.3f}   accuracy = {model_info['accuracy'][name]*100:.0f}%")
    champion = model_info["champion_name"]
    print(f"  champion model: {champion}")

    preds = pd.DataFrame(index=features.index)
    preds["sector"] = [sectors[n] for n in preds.index]
    preds["challenger_prob"] = model_info["probs"].reindex(preds.index)
    preds["verdict"] = [verdict_from_prob(p) for p in preds["challenger_prob"]]
    preds["label_true"] = labels.reindex(preds.index)
    preds["share_gain"] = share_gain.reindex(preds.index)
    preds["score"] = scores["score"].reindex(preds.index)
    preds["growth_momentum"] = scores["growth_momentum"].reindex(preds.index)
    preds["market_scale"] = scores["market_scale"].reindex(preds.index)
    preds = preds.sort_values(["sector", "challenger_prob"], ascending=[True, False])
    preds.to_csv(OUT_DIR / "predictions.csv")
    features.to_csv(OUT_DIR / "challenger_features.csv")
    print("  saved: outputs/predictions.csv, outputs/challenger_features.csv")

    # ---------------------------------------------------------- 6. charts
    _banner("[6/7] Rendering charts")
    visuals.plot_probability_ranking(preds, CHART_DIR / "01_challenger_probability.png")
    print("  saved: outputs/charts/01_challenger_probability.png")
    visuals.plot_prob_vs_score(preds, CHART_DIR / "02_prob_vs_score.png")
    print("  saved: outputs/charts/02_prob_vs_score.png")
    visuals.plot_trajectories(
        panel, results, preds, last_year, final_year,
        CHART_DIR / "03_market_trajectories.png",
    )
    print("  saved: outputs/charts/03_market_trajectories.png")
    visuals.plot_feature_importance(model_info, CHART_DIR / "04_feature_importance.png")
    print("  saved: outputs/charts/04_feature_importance.png")
    visuals.plot_roc(model_info, CHART_DIR / "05_roc_curves.png")
    print("  saved: outputs/charts/05_roc_curves.png")

    # --------------------------------------------------------- 7. report
    _banner("[7/7] Writing the markdown report")
    generate_report(
        panel=panel,
        results=results,
        scores=scores,
        preds=preds,
        model_info=model_info,
        features=features,
        outpath=REPORT_DIR / "summary_report.md",
        horizon=args.horizon,
        last_year=last_year,
        final_year=final_year,
    )
    print("  saved: outputs/reports/summary_report.md")

    # ---------------------------------------------------------- summary
    _banner(f"RESULT — Predicted challenger niches ({champion}, LOOCV AUC {model_info['auc'][champion]:.2f})")
    for sector in ("Healthcare", "Agriculture"):
        block = preds[preds["sector"] == sector]
        print(f"  [{sector}]")
        for niche, r in block.iterrows():
            tag = "*" if r["challenger_prob"] >= 0.5 else " "
            print(
                f"   {tag} P={r['challenger_prob']:.2f}  {niche:<44} "
                f"score {r['score']:5.1f}   (share {'+' if r['share_gain'] >= 0 else ''}{r['share_gain']*100:.1f} pp 19->{last_year})"
            )
    print()
    print("  Outputs:")
    print("    data/niche_panel.csv              - input panel dataset")
    print("    outputs/predictions.csv            - challenger probabilities & verdicts")
    print("    outputs/challenger_features.csv    - early-window model features")
    print("    outputs/scores.csv / forecasts.csv - prospect scores & forecasts + CI")
    print("    outputs/charts/*.png               - 5 charts")
    print("    outputs/reports/summary_report.md  - full written report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
