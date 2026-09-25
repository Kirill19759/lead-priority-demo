"""Обучение простой модели на учебной выборке demo."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample_leads.csv"
MODEL = ROOT / "models" / "lead_priority.joblib"

FEATURES_CAT = ["source", "device", "city"]
FEATURES_NUM = ["pages_viewed", "session_minutes", "repeat_visit", "form_filled"]
TARGET = "target_deal"


def main() -> None:
    df = pd.read_csv(DATA)
    x = df[FEATURES_CAT + FEATURES_NUM]
    y = df[TARGET]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
                        ("num", "passthrough", FEATURES_NUM),
                    ]
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, random_state=42),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, proba)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "features_cat": FEATURES_CAT,
            "features_num": FEATURES_NUM,
            "valid_roc_auc": float(auc),
        },
        MODEL,
    )
    print(f"saved {MODEL} valid ROC-AUC={auc:.3f}")


if __name__ == "__main__":
    main()
