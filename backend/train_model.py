"""
Trains a small model to predict Customer Satisfaction Rating (1-5)
from ticket attributes. Run once: python train_model.py
Produces model.joblib used by main.py for the "numeric" prediction endpoint.

Schema-agnostic by design: this dataset comes in a few variants (the real
Kaggle CSV vs. any regenerated/synthetic version) with slightly different
available columns. Rather than hardcoding a fixed feature list that breaks
the moment a column is named or typed differently, this script inspects
the actual CSV at runtime and uses whatever categorical/numeric candidate
columns are actually present and actually usable.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error

DATA_PATH = "../data/customer_support_tickets.csv"
MODEL_PATH = "model.joblib"
TARGET = "Customer Satisfaction Rating"

# Candidate features, in priority order. Not all of these exist in every
# variant of this dataset (e.g. the real Kaggle CSV has no numeric-hours
# response-time column — only a raw timestamp — so that candidate simply
# won't be picked up there). We use whichever candidates are actually
# present and actually parseable, instead of assuming a fixed schema.
CATEGORICAL_CANDIDATES = [
    "Product Purchased",
    "Ticket Type",
    "Ticket Priority",
    "Ticket Channel",
]
NUMERIC_CANDIDATES = [
    "Customer Age",
    "Resolution Duration (hrs)",   # derived from timestamps when available
    "First Response Time (hrs)",   # only exists in some dataset variants
    "Time to Resolution (hrs)",    # only exists in some dataset variants
]


def add_derived_resolution_duration(df):
    """Mirrors the same derivation in main.py — see that docstring for why
    negative durations are dropped rather than averaged in."""
    if "First Response Time" not in df.columns or "Time to Resolution" not in df.columns:
        return df
    frt = pd.to_datetime(df["First Response Time"], errors="coerce")
    ttr = pd.to_datetime(df["Time to Resolution"], errors="coerce")
    if frt.notna().mean() < 0.05 or ttr.notna().mean() < 0.05:
        return df
    duration_hrs = (ttr - frt).dt.total_seconds() / 3600.0
    duration_hrs = duration_hrs.where(duration_hrs >= 0)
    df["Resolution Duration (hrs)"] = duration_hrs
    return df


def resolve_features(df: pd.DataFrame):
    """Pick real, usable feature columns from the actual dataframe."""
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]

    numeric = []
    for c in NUMERIC_CANDIDATES:
        if c not in df.columns:
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        # If a column that's supposed to be numeric is mostly unparsable
        # (e.g. it's actually a raw timestamp string in this dataset
        # variant), it's not safe to use as a numeric feature — skip it
        # rather than silently feeding NaNs into the model.
        if coerced.notna().mean() > 0.5:
            numeric.append(c)

    return categorical, numeric


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_derived_resolution_duration(df)

    if TARGET not in df.columns:
        raise SystemExit(
            f"Column '{TARGET}' not found in {DATA_PATH}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[TARGET])
    if len(df) == 0:
        raise SystemExit(
            f"No rows have a value for '{TARGET}' after dropping missing rows — "
            f"can't train. Check the dataset."
        )

    categorical, numeric = resolve_features(df)
    features = categorical + numeric

    if not features:
        raise SystemExit(
            f"None of the expected feature columns were found or usable. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"Using categorical features: {categorical}")
    print(f"Using numeric features: {numeric}")
    print(f"Training on {len(df)} rows (rows with a satisfaction rating)")

    encoders = {}
    X = df[features].copy()
    for col in categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if pd.to_numeric(X[col], errors="coerce").notna().any() else 0)

    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"Validation accuracy: {acc:.3f} | MAE: {mae:.3f}")

    joblib.dump(
        {"model": clf, "encoders": encoders, "features": features, "categorical": categorical, "numeric": numeric},
        MODEL_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
