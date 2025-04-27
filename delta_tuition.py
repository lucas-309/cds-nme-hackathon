#!/usr/bin/env python3
"""
delta_tuition.py
─────────────────────────────────────────────────────────
Additive-growth tuition predictor for three school types:

      • Private
      • Public In-State
      • Public Out-of-State

Model (no linear regression)
----------------------------
  1. For each type, compute the yearly mean tuition across all states.
  2. Average the year-over-year **dollar** changes:     Δ̄
  3. Forecast:      cost̂(Y) = last_cost + Δ̄ · (Y − last_year)

Public API
----------
    estimate_tuition(school_type: str, year: int) -> float
    list_types() -> set[str]

CSV requirements
----------------
* Columns:  Year , State , Type , Length , Expense , Value
* Tuition rows are identified by Expense == "Fees/Tuition".
"""

from pathlib import Path
from functools import lru_cache
import sys
import numpy as np
import pandas as pd

CSV_PATH = Path("archive/overall_tuition.csv")          # ← change if needed
TYPE_ALIASES = {
    "private": "Private",
    "public in-state": "Public In-State",
    "public in-state": "Public In-State",
    "public instate": "Public In-State",
    "public out-of-state": "Public Out-of-State",
    "public out-of-state": "Public Out-of-State",
    "public outofstate": "Public Out-of-State",
}

# ── 1. Load & preprocess CSV (memoised) ────────────────────────────
@lru_cache
def _load_table() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    needed = {"Year", "State", "Type", "Expense", "Value"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df[df["Expense"].str.strip().str.lower() == "fees/tuition"]
    df = df.dropna(subset=["Year", "Type", "Value"])

    df["Year"]  = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    if (df["Value"] <= 0).any():
        raise ValueError("All tuition values must be positive.")

    # normalise Type names via aliases
    df["Type"] = (df["Type"].str.strip()
                            .str.replace(r"\s+", " ", regex=True)
                            .str.title())
    df["Type"] = df["Type"].map(lambda x: TYPE_ALIASES.get(x.lower(), x))

    return df


# ── 2. Build additive models per Type (memoised) ───────────────────
@lru_cache
def _build_models():
    tbl = _load_table()

    # mean tuition per type per year (averaged over all states/lengths)
    yearly = (
        tbl.groupby(["Type", "Year"])["Value"]
        .mean()
        .sort_index()
        .reset_index()
    )

    models = {}  # {type: (last_year, last_cost, avg_delta)}
    for t, grp in yearly.groupby("Type"):
        costs = grp["Value"].to_numpy(float)
        years = grp["Year"].to_numpy(int)
        avg_delta = np.mean(costs[1:] - costs[:-1]) if len(costs) > 1 else 0.0
        models[t] = (years[-1], costs[-1], avg_delta)

    return models


# ── 3. Public helpers ──────────────────────────────────────────────
def list_types() -> set[str]:
    """Valid type strings (use exactly as shown)."""
    return set(_build_models().keys())


def estimate_tuition(school_type: str, year: int) -> float:
    """
    Parameters
    ----------
    school_type : one of list_types()  (case-insensitive OK via aliases)
    year        : future (or past) academic year as int

    Returns
    -------
    Estimated tuition (float).
    """
    if not isinstance(year, int):
        raise TypeError("year must be int")

    st_norm = TYPE_ALIASES.get(school_type.lower(), school_type.title())
    models = _build_models()

    if st_norm not in models:
        raise KeyError(f"Unknown school_type: {school_type!r}")

    last_year, last_cost, avg_delta = models[st_norm]
    return last_cost + avg_delta * (year - last_year)


# ── 4. Interactive CLI when run directly ───────────────────────────
def _cli():
    print("🎓 Tuition predictor — additive growth (Δ̄) model\n")
    print("Valid types:", ", ".join(sorted(list_types())))
    try:
        while True:
            # year
            ytxt = input("\nEnter target year (e.g. 2030): ").strip()
            if not ytxt.isdigit():
                print("✖  Year must be digits only.")
                continue
            tgt_year = int(ytxt)

            # type
            stype = input("Enter school type: ").strip()
            try:
                est = estimate_tuition(stype, tgt_year)
                print(f"Estimated tuition for {stype.title()} in {tgt_year}: "
                      f"${est:,.2f}")
            except Exception as e:
                print("✖", e)

    except KeyboardInterrupt:
        print("\nGood-bye!")


if __name__ == "__main__":
    _cli()
