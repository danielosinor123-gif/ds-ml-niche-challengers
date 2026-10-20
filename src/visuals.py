"""Charts for the niche challenger project.

Five publication-ready figures:
    01 challenger probability ranking (both sectors)
    02 challenger probability vs 2026-2030 prospect score (quadrant map)
    03 market trajectories per sector (challengers emphasised)
    04 feature importance (logistic coefficients + forest importances)
    05 LOOCV ROC curves for both candidate models
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .challenger import FEATURE_LABELS

SECTOR_COLORS = {
    "Healthcare": "#00A499",
    "Agriculture": "#27AE60",
}

SHORT_NAMES = {
    "Medical Imaging & Diagnostics": "Imaging & Diagnostics",
    "Clinical Decision Support": "Clinical Decision Support",
    "Admin Automation & Revenue Cycle": "Admin & Revenue Cycle",
    "Drug Discovery & Development": "Drug Discovery AI",
    "Genomics & Precision Medicine": "Genomics & Precision Med",
    "Ambient Clinical Documentation": "Ambient Documentation",
    "Remote Patient Monitoring & Wearables": "Remote Patient Monitoring",
    "Virtual Assistants & Mental-Health AI": "Virtual Assistants/MH",
    "Synthetic Data & Trial Digital Twins": "Synthetic Data & Twins",
    "Outbreak & Epidemiological Forecasting": "Epidemiological F'casting",
    "Remote Sensing & Crop Scouting": "Crop Scouting/Remote Sensing",
    "Yield Prediction & Farm Analytics": "Yield Prediction",
    "Livestock Monitoring & Health Vision": "Livestock Monitoring",
    "Autonomous Equipment & Field Robotics": "Ag Field Robotics",
    "ML-Guided Seed Breeding & Gene Editing": "Seed Breeding AI",
    "Carbon & Sustainability Analytics (MRV)": "Carbon & MRV",
    "Indoor & Vertical Farming Optimization": "Indoor Farming AI",
    "Pest & Disease Early Warning": "Pest & Disease Warning",
}

_CHALLENGER_ZONE = "#27AE60"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D0D0",
            "axes.grid": True,
            "grid.color": "#E8E8E8",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10.5,
            "legend.frameon": False,
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "axes.labelcolor": "#333333",
            "text.color": "#222222",
        }
    )


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _footer(fig, text: str) -> None:
    fig.text(0.012, 0.008, text, fontsize=7.2, color="#999999", ha="left")


def _gradient_colors(n: int, hex_a: str, hex_b: str) -> list[str]:
    a = np.array(matplotlib.colors.to_rgb(hex_a))
    b = np.array(matplotlib.colors.to_rgb(hex_b))
    return [matplotlib.colors.to_hex((1 - w) * a + w * b) for w in np.linspace(0, 1, n)]


def _sector_gradient(sector: str, n: int) -> list[str]:
    if sector == "Healthcare":
        return _gradient_colors(n, "#00A499", "#3D2C6D")
    return _gradient_colors(n, "#8CBF26", "#145A32")


# --------------------------------------------------------------------------
def plot_probability_ranking(
    preds: pd.DataFrame, outpath: Path
) -> None:
    """Horizontal bars of challenger probability, grouped by sector."""
    _style()
    order = ["Healthcare", "Agriculture"]
    blocks = []
    for sector in order:
        sub = preds[preds["sector"] == sector].sort_values(
            "challenger_prob", ascending=False
        )
        blocks.append(sub)
    df = pd.concat(blocks)
    n = len(df)

    fig, ax = plt.subplots(figsize=(11.5, 7.6), dpi=170)
    positions = np.arange(n)[::-1]

    # challenger zone (prob >= 0.5)
    ax.axvspan(0.5, 1.0, color=_CHALLENGER_ZONE, alpha=0.06, zorder=1)
    ax.axvline(0.5, color="#27AE60", ls=":", lw=1.2)

    bars = ax.barh(
        positions, df["challenger_prob"],
        color=[SECTOR_COLORS[s] for s in df["sector"]],
        height=0.62, zorder=3,
    )
    for bar, p in zip(bars, df["challenger_prob"]):
        bar.set_alpha(1.0 if p >= 0.5 else 0.55)
        ax.text(
            p + 0.012, bar.get_y() + bar.get_height() / 2, f"{p:.2f}",
            va="center", fontsize=8.6, fontweight="bold", color="#333333",
        )

    labels = [
        f"{SHORT_NAMES.get(n, n)}  \u00b7  {s.lower()}"
        for n, s in zip(df.index, df["sector"])
    ]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Predicted challenger probability (LOOCV-selected champion model)")
    ax.set_title("Predicted Challenger Probability by ML Niche")
    ax.grid(axis="y", visible=False)
    _despine(ax)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _footer(
        fig,
        "Green zone = predicted challenger (prob \u2265 0.5) \u00b7 bars muted below "
        "0.5 \u00b7 probabilities from the champion classifier refit on all 18 niches",
    )
    fig.savefig(outpath)
    plt.close(fig)


def plot_prob_vs_score(preds: pd.DataFrame, outpath: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11.8, 7.6), dpi=170)

    x = preds["challenger_prob"].to_numpy()
    y = preds["score"].to_numpy()
    med_y = float(np.median(y))

    ax.axvline(0.5, color="#B5B5B5", ls="--", lw=1)
    ax.axhline(med_y, color="#B5B5B5", ls="--", lw=1)

    for niche, row in preds.iterrows():
        c = SECTOR_COLORS[row["sector"]]
        ax.scatter(
            row["challenger_prob"], row["score"],
            s=110 + row["score"] * 3.2, color=c, alpha=0.85,
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        ax.annotate(
            SHORT_NAMES.get(niche, niche),
            xy=(row["challenger_prob"], row["score"]),
            xytext=(7, 5), textcoords="offset points",
            fontsize=7.8, color="#333333",
        )

    ax.text(0.985, 0.97, "CONFIRMED CHALLENGERS\nhigh probability \u00d7 high prospect",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color="#B5B5B5", fontweight="bold")
    ax.text(0.015, 0.97, "ESTABLISHED HEAVYWEIGHTS\nbig but not rising",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color="#B5B5B5", fontweight="bold")
    ax.text(0.985, 0.05, "EMERGING WATCH LIST",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="#B5B5B5", fontweight="bold")
    ax.text(0.015, 0.05, "FADING",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
            color="#B5B5B5", fontweight="bold")

    ax.set_xlim(-0.04, 1.12)
    ax.set_xlabel("Predicted challenger probability")
    ax.set_ylabel("2026\u20132030 prospect score (within sector, 0\u2013100)")
    ax.set_title("Challenger Probability \u00d7 Prospect Score \u2014 the Verdict Map")
    handles = [
        matplotlib.lines.Line2D([], [], color=c, marker="o", ls="", ms=9, label=s)
        for s, c in SECTOR_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    _despine(ax)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _footer(fig, "Bubble size = within-sector prospect score \u00b7 dashed lines = "
                 "0.5 probability and median prospect score")
    fig.savefig(outpath)
    plt.close(fig)


def plot_trajectories(
    panel: pd.DataFrame,
    results: dict,
    preds: pd.DataFrame,
    last_year: int,
    final_year: int,
    outpath: Path,
    metric: str = "market_size_usd_b",
) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), dpi=170)
    fmt = "${v:,.2f}B" if metric == "market_size_usd_b" else "{v:,.1f}k"

    for ax, sector in zip(axes, ["Healthcare", "Agriculture"]):
        niches = preds[preds["sector"] == sector].index.tolist()
        colors = _sector_gradient(sector, max(len(niches), 2))
        color_map = {n: colors[i] for i, n in enumerate(niches)}

        # challengers drawn last so they sit on top
        ordered = sorted(
            niches, key=lambda n: preds.loc[n, "challenger_prob"] >= 0.5
        )

        boundary = last_year + 0.5
        for niche in ordered:
            res = results[(niche, metric)]
            emph = preds.loc[niche, "challenger_prob"] >= 0.5
            c = color_map[niche]
            lw, alpha = (2.2, 1.0) if emph else (1.0, 0.40)
            ax.plot(
                res.hist_years, res.hist_values, color=c, lw=lw,
                alpha=alpha, zorder=4 if emph else 2.5, label=niche if emph else None,
            )
            ax.plot(
                res.years, res.point, color=c, lw=lw * 0.9, ls="--",
                alpha=alpha, zorder=4 if emph else 2.5,
            )
            if emph:
                ax.fill_between(
                    res.years, res.lo, res.hi, color=c, alpha=0.11, lw=0, zorder=1.5,
                )
                ax.annotate(
                    fmt.format(v=res.point[-1]),
                    xy=(res.years[-1], res.point[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=c, va="center",
                )

        ax.axvline(boundary, color="#B5B5B5", ls=":", lw=1)
        ax.set_title(f"{sector} niches \u2014 challengers emphasised")
        ax.set_xlabel("Year")
        ax.set_ylabel("Niche DS/ML market ($B)")
        ax.legend(loc="upper left", fontsize=7.6)
        _despine(ax)

    fig.suptitle(
        "Niche Market Trajectories \u2014 History & Ensemble Forecast",
        fontsize=14, fontweight="bold", x=0.055, ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    _footer(
        fig,
        "History solid, 2026\u20132030 ensemble forecast dashed with 90% bootstrap CI "
        "\u00b7 emphasised = predicted challengers (prob \u2265 0.5)",
    )
    fig.savefig(outpath)
    plt.close(fig)


def plot_feature_importance(model_info: dict, outpath: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=170)

    logit = model_info["importances"]["Logistic Regression"].sort_values(key=np.abs)
    ax = axes[0]
    colors = ["#C0392B" if v < 0 else "#1F6FB2" for v in logit.values]
    ax.barh(
        range(len(logit)), logit.values, color=colors, height=0.62, zorder=3,
    )
    ax.set_yticks(range(len(logit)))
    ax.set_yticklabels(
        [FEATURE_LABELS.get(c, c) for c in logit.index], fontsize=8.6
    )
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_title("Logistic regression coefficients (standardised)")
    ax.set_xlabel("Coefficient (blue = challenger-positive)")
    ax.grid(axis="y", visible=False)
    _despine(ax)

    rf = model_info["importances"]["Random Forest"].sort_values()
    ax = axes[1]
    ax.barh(
        range(len(rf)), rf.values, color="#00A499", height=0.62, zorder=3,
    )
    ax.set_yticks(range(len(rf)))
    ax.set_yticklabels(
        [FEATURE_LABELS.get(c, c) for c in rf.index], fontsize=8.6
    )
    ax.set_title("Random forest importances (impurity reduction)")
    ax.set_xlabel("Relative importance")
    ax.set_xlim(0, rf.max() * 1.18)
    ax.grid(axis="y", visible=False)
    _despine(ax)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _footer(
        fig,
        "Features computed from the 2015\u20132019 window only \u00b7 labels = "
        "within-sector share gain 2019\u21922025 above sector median",
    )
    fig.savefig(outpath)
    plt.close(fig)


def plot_roc(model_info: dict, outpath: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 7.0), dpi=170)

    for name, (fpr, tpr) in model_info["roc"].items():
        auc = model_info["auc"][name]
        is_champ = name == model_info["champion_name"]
        ax.plot(
            fpr, tpr, lw=2.4 if is_champ else 1.8,
            color="#00A499" if is_champ else "#888888",
            label=f"{name} \u2014 LOOCV AUC = {auc:.3f}" + (" (champion)" if is_champ else ""),
        )
    ax.plot([0, 1], [0, 1], color="#CCCCCC", ls="--", lw=1)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Leave-One-Out ROC \u2014 Challenger Detection")
    ax.legend(loc="lower right", fontsize=9.5)
    _despine(ax)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _footer(
        fig,
        "Out-of-fold probabilities over 18 leave-one-out folds \u00b7 the champion is "
        "refit on all niches to produce final probabilities",
    )
    fig.savefig(outpath)
    plt.close(fig)
