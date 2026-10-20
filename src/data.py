"""Benchmark-calibrated niche-level panel: ML/DS niches inside two sectors.

18 niches (10 in Healthcare, 8 in Agriculture) x 2015-2025 x 5 indicators:

    market_size_usd_b   Annual DS/ML spend within the niche ($B)
    investment_usd_b    Funding & R&D capital directed at the niche ($B)
    job_postings_k      DS/ML job postings per year (thousands)
    adoption_rate_pct   Share of relevant firms deploying ML in the niche (%)
    ai_patents_k        ML-related patents granted per year (thousands)

Niche 2025 calibration targets (public estimates, global):
    Healthcare (~$21B sector, matches the sister industry project):
      Medical imaging & diagnostics AI     ~$4.9B  (largest, most mature)
      Administrative / revenue-cycle AI     ~$3.2B  (incl. GenAI coding)
      Drug discovery & development AI      ~$3.6B  (fast, VC-heavy)
      Clinical decision support            ~$3.0B  (mature)
      Genomics & precision medicine        ~$2.4B
      Remote patient monitoring            ~$1.7B
      Ambient clinical documentation       ~$0.9B  (post-2023 GenAI boom)
      Virtual assistants / mental health   ~$1.1B
      Synthetic data & trial digital twins ~$0.6B
      Outbreak & epi forecasting           ~$0.5B
    Agriculture (~$3.5B sector):
      Remote sensing & crop scouting       ~$0.75B (mature)
      Yield prediction & farm analytics    ~$0.65B (mature)
      Autonomous equipment & robotics     ~$0.55B
      Livestock monitoring & health vision ~$0.42B
      Indoor / vertical farming optim.     ~$0.31B
      ML-guided seed breeding / gene edit. ~$0.28B
      Pest & disease early warning        ~$0.29B
      Carbon & sustainability MRV          ~$0.22B (fastest, carbon markets)

HONESTY NOTE: values are *calibrated estimates* (synthetic but realistic).
The challenger classifier in src/challenger.py therefore demonstrates the
METHOD; on real data the same pipeline runs unchanged (--from-csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_YEARS = list(range(2015, 2026))
SEED = 42

HEALTHCARE = "Healthcare"
AGRICULTURE = "Agriculture"

NICHE_CALIBRATION: dict[str, dict] = {
    # ------------------------------------------------------- healthcare
    "Medical Imaging & Diagnostics": {
        "sector": HEALTHCARE,
        "base_market": 1.00, "market_cagr": 0.171,
        "invest_alpha": 0.55, "invest_extra": 0.010,
        "base_jobs": 2.00, "jobs_cagr": 0.17,
        "adopt_ceiling": 85.0, "adopt_mid": 2018.0, "adopt_k": 0.55,
        "base_patents": 0.45, "patents_cagr": 0.20,
        "events": {2020: 1.04},
    },
    "Clinical Decision Support": {
        "sector": HEALTHCARE,
        "base_market": 0.80, "market_cagr": 0.141,
        "invest_alpha": 0.50, "invest_extra": 0.010,
        "base_jobs": 1.60, "jobs_cagr": 0.15,
        "adopt_ceiling": 78.0, "adopt_mid": 2019.0, "adopt_k": 0.50,
        "base_patents": 0.30, "patents_cagr": 0.18,
        "events": {},
    },
    "Admin Automation & Revenue Cycle": {
        "sector": HEALTHCARE,
        "base_market": 0.70, "market_cagr": 0.165,
        "invest_alpha": 0.60, "invest_extra": 0.015,
        "base_jobs": 1.40, "jobs_cagr": 0.19,
        "adopt_ceiling": 72.0, "adopt_mid": 2019.5, "adopt_k": 0.50,
        "base_patents": 0.15, "patents_cagr": 0.16,
        "events": {2023: 1.18},
    },
    "Drug Discovery & Development": {
        "sector": HEALTHCARE,
        "base_market": 0.19, "market_cagr": 0.34,
        "invest_alpha": 0.85, "invest_extra": 0.020,
        "base_jobs": 0.90, "jobs_cagr": 0.28,
        "adopt_ceiling": 68.0, "adopt_mid": 2021.0, "adopt_k": 0.50,
        "base_patents": 0.50, "patents_cagr": 0.30,
        "events": {2020: 1.10, 2021: 1.06},
    },
    "Genomics & Precision Medicine": {
        "sector": HEALTHCARE,
        "base_market": 0.15, "market_cagr": 0.32,
        "invest_alpha": 0.75, "invest_extra": 0.015,
        "base_jobs": 0.80, "jobs_cagr": 0.26,
        "adopt_ceiling": 70.0, "adopt_mid": 2020.5, "adopt_k": 0.50,
        "base_patents": 0.40, "patents_cagr": 0.28,
        "events": {},
    },
    "Ambient Clinical Documentation": {
        "sector": HEALTHCARE,
        "base_market": 0.010, "market_cagr": 0.52,
        "invest_alpha": 0.90, "invest_extra": 0.030,
        "base_jobs": 0.35, "jobs_cagr": 0.45,
        "adopt_ceiling": 88.0, "adopt_mid": 2022.8, "adopt_k": 0.90,
        "base_patents": 0.10, "patents_cagr": 0.45,
        "events": {2023: 1.30},
    },
    "Remote Patient Monitoring & Wearables": {
        "sector": HEALTHCARE,
        "base_market": 0.09, "market_cagr": 0.34,
        "invest_alpha": 0.65, "invest_extra": 0.015,
        "base_jobs": 0.60, "jobs_cagr": 0.24,
        "adopt_ceiling": 66.0, "adopt_mid": 2021.2, "adopt_k": 0.50,
        "base_patents": 0.25, "patents_cagr": 0.26,
        "events": {2020: 1.12},
    },
    "Virtual Assistants & Mental-Health AI": {
        "sector": HEALTHCARE,
        "base_market": 0.05, "market_cagr": 0.37,
        "invest_alpha": 0.70, "invest_extra": 0.020,
        "base_jobs": 0.45, "jobs_cagr": 0.30,
        "adopt_ceiling": 74.0, "adopt_mid": 2021.5, "adopt_k": 0.60,
        "base_patents": 0.15, "patents_cagr": 0.32,
        "events": {2020: 1.15, 2023: 1.10},
    },
    "Synthetic Data & Trial Digital Twins": {
        "sector": HEALTHCARE,
        "base_market": 0.02, "market_cagr": 0.40,
        "invest_alpha": 0.80, "invest_extra": 0.020,
        "base_jobs": 0.25, "jobs_cagr": 0.34,
        "adopt_ceiling": 60.0, "adopt_mid": 2022.0, "adopt_k": 0.55,
        "base_patents": 0.12, "patents_cagr": 0.35,
        "events": {2020: 1.15},
    },
    "Outbreak & Epidemiological Forecasting": {
        "sector": HEALTHCARE,
        "base_market": 0.09, "market_cagr": 0.19,
        "invest_alpha": 0.55, "invest_extra": 0.005,
        "base_jobs": 0.30, "jobs_cagr": 0.16,
        "adopt_ceiling": 62.0, "adopt_mid": 2019.2, "adopt_k": 0.50,
        "base_patents": 0.08, "patents_cagr": 0.18,
        "events": {2020: 1.30},
    },
    # ------------------------------------------------------- agriculture
    "Remote Sensing & Crop Scouting": {
        "sector": AGRICULTURE,
        "base_market": 0.22, "market_cagr": 0.130,
        "invest_alpha": 0.50, "invest_extra": 0.005,
        "base_jobs": 0.30, "jobs_cagr": 0.14,
        "adopt_ceiling": 76.0, "adopt_mid": 2018.6, "adopt_k": 0.50,
        "base_patents": 0.12, "patents_cagr": 0.17,
        "events": {},
    },
    "Yield Prediction & Farm Analytics": {
        "sector": AGRICULTURE,
        "base_market": 0.18, "market_cagr": 0.137,
        "invest_alpha": 0.45, "invest_extra": 0.005,
        "base_jobs": 0.28, "jobs_cagr": 0.13,
        "adopt_ceiling": 72.0, "adopt_mid": 2019.0, "adopt_k": 0.50,
        "base_patents": 0.10, "patents_cagr": 0.15,
        "events": {},
    },
    "Livestock Monitoring & Health Vision": {
        "sector": AGRICULTURE,
        "base_market": 0.030, "market_cagr": 0.30,
        "invest_alpha": 0.65, "invest_extra": 0.015,
        "base_jobs": 0.10, "jobs_cagr": 0.24,
        "adopt_ceiling": 64.0, "adopt_mid": 2021.5, "adopt_k": 0.55,
        "base_patents": 0.08, "patents_cagr": 0.26,
        "events": {2020: 1.08},
    },
    "Autonomous Equipment & Field Robotics": {
        "sector": AGRICULTURE,
        "base_market": 0.035, "market_cagr": 0.32,
        "invest_alpha": 0.80, "invest_extra": 0.020,
        "base_jobs": 0.12, "jobs_cagr": 0.27,
        "adopt_ceiling": 62.0, "adopt_mid": 2021.8, "adopt_k": 0.50,
        "base_patents": 0.14, "patents_cagr": 0.30,
        "events": {},
    },
    "ML-Guided Seed Breeding & Gene Editing": {
        "sector": AGRICULTURE,
        "base_market": 0.014, "market_cagr": 0.35,
        "invest_alpha": 0.85, "invest_extra": 0.025,
        "base_jobs": 0.06, "jobs_cagr": 0.30,
        "adopt_ceiling": 58.0, "adopt_mid": 2022.2, "adopt_k": 0.55,
        "base_patents": 0.16, "patents_cagr": 0.32,
        "events": {2022: 1.10},
    },
    "Carbon & Sustainability Analytics (MRV)": {
        "sector": AGRICULTURE,
        "base_market": 0.008, "market_cagr": 0.38,
        "invest_alpha": 0.75, "invest_extra": 0.030,
        "base_jobs": 0.05, "jobs_cagr": 0.32,
        "adopt_ceiling": 56.0, "adopt_mid": 2022.6, "adopt_k": 0.60,
        "base_patents": 0.05, "patents_cagr": 0.36,
        "events": {2021: 1.18},
    },
    "Indoor & Vertical Farming Optimization": {
        "sector": AGRICULTURE,
        "base_market": 0.018, "market_cagr": 0.33,
        "invest_alpha": 0.70, "invest_extra": 0.015,
        "base_jobs": 0.08, "jobs_cagr": 0.26,
        "adopt_ceiling": 68.0, "adopt_mid": 2020.8, "adopt_k": 0.55,
        "base_patents": 0.07, "patents_cagr": 0.28,
        "events": {2020: 1.06},
    },
    "Pest & Disease Early Warning": {
        "sector": AGRICULTURE,
        "base_market": 0.08, "market_cagr": 0.135,
        "invest_alpha": 0.50, "invest_extra": 0.005,
        "base_jobs": 0.12, "jobs_cagr": 0.15,
        "adopt_ceiling": 66.0, "adopt_mid": 2019.6, "adopt_k": 0.50,
        "base_patents": 0.07, "patents_cagr": 0.18,
        "events": {2020: 1.10},
    },
}

PANEL_COLUMNS = [
    "niche", "sector", "year", "market_size_usd_b", "investment_usd_b",
    "job_postings_k", "adoption_rate_pct", "ai_patents_k",
]


def build_panel(seed: int = SEED) -> pd.DataFrame:
    """Generate the niche x sector x year panel with reproducible noise."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for niche, cfg in NICHE_CALIBRATION.items():
        for i, year in enumerate(HISTORY_YEARS):
            event = cfg["events"].get(year, 1.0)

            market = (
                cfg["base_market"]
                * (1.0 + cfg["market_cagr"]) ** i
                * event
                * np.exp(rng.normal(0.0, 0.045))
            )
            invest = (
                market
                * cfg["invest_alpha"]
                * (1.0 + cfg["invest_extra"]) ** i
                * np.exp(rng.normal(0.0, 0.060))
            )
            jobs = (
                cfg["base_jobs"]
                * (1.0 + cfg["jobs_cagr"]) ** i
                * event
                * np.exp(rng.normal(0.0, 0.050))
            )
            adoption = cfg["adopt_ceiling"] / (
                1.0 + np.exp(-cfg["adopt_k"] * (year - cfg["adopt_mid"]))
            )
            adoption = float(np.clip(adoption + rng.normal(0.0, 0.8), 1.0, 99.0))
            patents = (
                cfg["base_patents"]
                * (1.0 + cfg["patents_cagr"]) ** i
                * np.exp(rng.normal(0.0, 0.070))
            )

            rows.append(
                {
                    "niche": niche,
                    "sector": cfg["sector"],
                    "year": year,
                    "market_size_usd_b": round(float(market), 4),
                    "investment_usd_b": round(float(invest), 4),
                    "job_postings_k": round(float(jobs), 4),
                    "adoption_rate_pct": round(adoption, 2),
                    "ai_patents_k": round(float(patents), 4),
                }
            )

    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


def load_panel_from_csv(path: str) -> pd.DataFrame:
    """Load an external niche panel with the same schema as build_panel()."""
    df = pd.read_csv(path)
    missing = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {path} is missing required columns: {missing}. "
            f"Expected schema: {PANEL_COLUMNS}"
        )
    return df
