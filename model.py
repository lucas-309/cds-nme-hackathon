#!/usr/bin/env python3
"""
Conservative tuition predictor  (additive yearly growth, no linear regression)
──────────────────────────────────────────────────────────────────────────────
Datasets
  G  → tuition_graduate.csv
       • academic.year , school     , cost
  U  → archive/undergraduate_package.csv
       • academic.year , component  , cost
         (only rows where component == "Total" are used)

Model
  For each series, compute the *average annual dollar increase* Δ̄.
  Forecast:  cost̂(Y) = last_cost + Δ̄ · (Y − last_year)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ─────────────── file locations ──────────────────────────
CSV_PATHS = {
    "g": Path("archive/tuition_graduate.csv"),
    "u": Path("archive/undergraduate_package.csv"),
}

# ─────────────── dataset prompt ──────────────────────────
kind = ""
while kind not in {"u", "g"}:
    kind = input("Predict Graduate (G) or Undergraduate (U) tuition? ").strip().lower()
    if kind not in {"u", "g"}:
        print("✖  Please type “G” for graduate or “U” for undergraduate.\n")

kind_full   = "graduate"      if kind == "g" else "undergraduate"
program_col = "school"        if kind == "g" else "component"
csv_path    = CSV_PATHS[kind]

if not csv_path.exists():
    sys.exit(f"[fatal] CSV not found: {csv_path}")

# ─────────────── load & validate ─────────────────────────
df = pd.read_csv(csv_path)

need = {"academic.year", program_col, "cost"}
miss = need - set(df.columns)
if miss:
    sys.exit(f"[fatal] CSV missing columns: {sorted(miss)}")

df = df.dropna(subset=need)
df["academic.year"] = pd.to_numeric(df["academic.year"], errors="coerce")
df["cost"]          = pd.to_numeric(df["cost"],          errors="coerce")
df = df.dropna(subset=need)

if (df["cost"] <= 0).any():
    sys.exit("[fatal] All costs must be positive.")

# undergraduate → keep only "Total"
if kind == "u":
    df = df[df[program_col] == "Total"]
    if df.empty:
        sys.exit('[fatal] No rows with component == "Total" found.')
    df[program_col] = "Total"           # single key

# ─────────────── build additive model ───────────────────
models = {}   # {program: dict(last_year, last_cost, avg_delta)}
for prog, grp in df.groupby(program_col):
    grp   = grp.sort_values("academic.year")
    years = grp["academic.year"].to_numpy(int)
    costs = grp["cost"].to_numpy(float)

    avg_delta = np.mean(costs[1:] - costs[:-1]) if len(costs) > 1 else 0.0
    models[prog] = {"last_year": years[-1], "last_cost": costs[-1], "avg_delta": avg_delta}

print(f"\n[info] Loaded {kind_full} data — {len(models)} "
      f"{'programs' if kind=='g' else 'component(s)'} ready.")

# ─────────────── interactive loop ───────────────────────
print("\n🔮  Tuition predictor — press Ctrl-C to quit.\n")
try:
    while True:
        # year
        ytxt = input("Enter target academic year (e.g. 2030): ").strip()
        if not ytxt.isdigit():
            print("✖  Year must be digits only. Try again.\n")
            continue
        year = int(ytxt)

        # program name (graduate only)
        if kind == "g":
            prog = input("Enter program name (exactly as in CSV): ").strip()
            if prog not in models:
                print(f"✖  Unknown program “{prog}”. Try again.\n")
                continue
        else:
            prog = "Total"

        m           = models[prog]
        years_ahead = year - m["last_year"]
        est_cost    = m["last_cost"] + m["avg_delta"] * years_ahead
        print(f"→ Estimated {kind_full} cost for {prog} in {year}:  ${est_cost:,.2f}\n")

except KeyboardInterrupt:
    print("\nGood-bye!")
