"""
tuition_predictor.py
────────────────────
Additive-growth tuition forecast (no linear regression, no caching).

Public API
----------
    estimate_tuition(kind, year, program=None) -> float
    list_programs(kind) -> set[str]

Command-line mode
-----------------
    python tuition_predictor.py
        ↳ prompts in a loop until Ctrl-C.
"""

from pathlib import Path
import sys
from functools import lru_cache
import numpy as np
import pandas as pd
from delta_tuition import estimate_tuition

# ───────────────────────────────────────────────────────────────
# 0.  Paths & basic config
# ───────────────────────────────────────────────────────────────
CSV_PATHS = {
    "g": Path("archive/tuition_graduate.csv"),
    "u": Path("archive/undergraduate_package.csv"),
}
PROG_COL = {"g": "school", "u": "component"}


# ───────────────────────────────────────────────────────────────
# 1.  Internal helpers (memoised)
# ───────────────────────────────────────────────────────────────
@lru_cache
def _load_df(kind: str) -> pd.DataFrame:
    if kind not in {"g", "u"}:
        raise ValueError("kind must be 'g' or 'u'")

    path = CSV_PATHS[kind]
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    col_prog = PROG_COL[kind]
    df = pd.read_csv(path, dtype={col_prog: str}).rename(columns=str.strip)

    required = {"academic.year", col_prog, "cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df.dropna(subset=required)
    df["academic.year"] = pd.to_numeric(df["academic.year"], errors="coerce")
    df["cost"]          = pd.to_numeric(df["cost"],          errors="coerce")
    df = df.dropna(subset=required)

    if (df["cost"] <= 0).any():
        raise ValueError("All costs must be positive.")

    # undergraduate → keep only 'Total'
    if kind == "u":
        df = df[df[col_prog] == "Total"]
        if df.empty:
            raise ValueError('No rows with component == "Total" found.')
        df[col_prog] = "Total"

    return df


@lru_cache
def _build_models(kind: str):
    """
    Returns dict[program] -> (last_year, last_cost, avg_delta)
    using additive average yearly dollar increase.
    """
    df = _load_df(kind)
    col_prog = PROG_COL[kind]

    models = {}
    for prog, g in df.groupby(col_prog):
        g = g.sort_values("academic.year")
        years = g["academic.year"].to_numpy(int)
        costs = g["cost"].to_numpy(float)

        avg_delta = np.mean(costs[1:] - costs[:-1]) if len(costs) > 1 else 0.0
        models[prog] = (years[-1], costs[-1], avg_delta)
    return models


# ───────────────────────────────────────────────────────────────
# 2.  Public API
# ───────────────────────────────────────────────────────────────
def list_programs(kind: str) -> set[str]:
    """Return the set of valid program names (graduate) or {'Total'} (undergrad)."""
    return set(_build_models(kind.lower()).keys())


def estimate_tuition2(kind: str, year: int, program: str | None = None) -> float:
    """
    kind    : "g" (graduate) or "u" (undergraduate)
    year    : int academic year
    program : required for graduate
    """
    kind = kind.lower()
    if kind not in {"g", "u"}:
        raise ValueError("kind must be 'g' or 'u'")
    if not isinstance(year, int):
        raise TypeError("year must be int")

    models = _build_models(kind)

    if kind == "g":
        if program is None:
            raise ValueError("program is required for graduate estimation")
        if program not in models:
            raise KeyError(f"Unknown program: {program!r}")
        key = program
    else:
        key = "Total"

    last_year, last_cost, avg_delta = models[key]
    return last_cost + avg_delta * (year - last_year)


# ───────────────────────────────────────────────────────────────
# 3.  Interactive CLI (only when run directly)
# ───────────────────────────────────────────────────────────────
def _cli_loop():
    print("🎓 Tuition predictor — additive growth model")
    kind = ""
    while kind not in {"u", "g"}:
        kind = input("Predict Graduate (G) or Undergraduate (U) tuition? ").strip().lower()
        if kind not in {"u", "g"}:
            print("✖  Please type 'G' or 'U'.\n")

    kind_full = "graduate" if kind == "g" else "undergraduate"

    try:
        while True:
            yr_str = input("Target academic year (e.g. 2030): ").strip()
            if not yr_str.isdigit():
                print("✖  Year must be digits only. Try again.\n")
                continue
            year = int(yr_str)

            if kind == "g":
                prog = input("Graduate program name (exactly as in CSV): ").strip()
            else:
                prog = None  # ignored

            try:
                est = estimate_tuition2(kind, year, prog)
                label = prog if prog else "Total"
                print(f"→ Estimated {kind_full} cost for {label} in {year}:  ${est:,.2f}\n")
            except Exception as e:
                print(f"✖  {e}\n")

    except KeyboardInterrupt:
        print("\nGood-bye!")


if __name__ == "__main__":
    _cli_loop()
