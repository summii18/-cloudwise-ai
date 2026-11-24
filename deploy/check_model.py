"""Confirm the saved model loads and still predicts a known label."""

import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("error", category=InconsistentVersionWarning)

MODEL_PATH = Path(__file__).resolve().parent.parent / "cloudwise_model.pkl"
EXPECTED_FEATURES = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes",
]
ALLOWED_LABELS = {"Keep Running", "Sleep Mode", "Shutdown"}


def main():
    model = joblib.load(MODEL_PATH)
    names = list(model.feature_names_in_)
    if names != EXPECTED_FEATURES:
        raise SystemExit(f"Unexpected features: {names}")

    sample = pd.DataFrame(
        [[20.0, 30.0, 5, 10]],
        columns=EXPECTED_FEATURES,
    )
    label = model.predict(sample)[0]
    if label not in ALLOWED_LABELS:
        raise SystemExit(f"Unexpected prediction: {label}")

    print(f"model_ok {type(model).__name__} {label}")


if __name__ == "__main__":
    main()
