#!/usr/bin/env python3
"""
predict_tuition.py
Predict graduate-school tuition for a given major and future year.
"""

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_PATH   = "archive/tuition_graduate.csv"   # ← change if your file lives elsewhere
MODEL_PATH  = "tuition_model.pkl"      # saved model for quick reuse


# ────────────────────────── 1. DATA LOADER ──────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read CSV and do minimal cleaning."""
    df = pd.read_csv(path)
    required_cols = {"academic.year", "school", "cost"}
    missing       = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain {missing} columns.")
    df = df.dropna(subset=required_cols)          # remove incomplete rows
    df["academic.year"] = df["academic.year"].astype(int)           # ensure numeric year
    return df


# ────────────────────────── 2. MODEL TRAINING ───────────────────────
def build_pipeline() -> Pipeline:
    """Pipeline = One-Hot encode Major ➜ feed to LinearRegression."""
    preproc = ColumnTransformer(
        transformers=[("major_ohe", OneHotEncoder(handle_unknown="ignore"), ["school"])],
        remainder="passthrough"       # keep the Year column untouched
    )
    return Pipeline(steps=[
        ("preprocess", preproc),
        ("regressor",  LinearRegression())
    ])


def train_and_save(df: pd.DataFrame, path: str = MODEL_PATH):
    """Train pipeline on full data, report hold-out MAE, then save."""
    X, y = df[["academic.year", "school"]], df["cost"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["school"]
    )
    pipe = build_pipeline().fit(X_train, y_train)
    mae  = mean_absolute_error(y_test, pipe.predict(X_test))
    print(f"[info] Model trained. Hold-out MAE ≈ ${mae:,.0f}")
    joblib.dump(pipe, path)
    print(f"[info] Saved to “{path}”.")
    return pipe


# ────────────────────────── 3. PREDICTION LOOP ──────────────────────
def predict_loop(model: Pipeline):
    """Tiny CLI for interactive predictions."""
    print("\n🔮  Tuition Predictor – type Ctrl-C to quit.\n")
    while True:
        try:
            yr = int(input("Enter target year (e.g. 2030): ").strip())
            mj = input("Enter major exactly as in the dataset (e.g. “Computer Science”): ").strip()
            sample = pd.DataFrame({"academic.year": [yr], "Major": [mj]})
            pred   = model.predict(sample)[0]
            print(f"→ Estimated tuition for {mj} in {yr}:  ${pred:,.2f}\n")
        except ValueError:
            print("  ✖  Year must be an integer. Try again.\n")
        except KeyboardInterrupt:
            print("\nGood-bye!")
            break
        except Exception as e:
            print(f"  ✖  {e}\n")


# ────────────────────────── 4. MAIN ─────────────────────────────────
def main():
    if os.path.exists(MODEL_PATH):
        print("[info] Loading cached model …")
        model = joblib.load(MODEL_PATH)
    else:
        print("[info] No saved model found – training from scratch …")
        df    = load_data()
        model = train_and_save(df)

    predict_loop(model)


if __name__ == "__main__":
    main()
