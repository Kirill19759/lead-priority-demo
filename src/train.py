"""Генерация учебной выборки и обучение модели демо."""

from __future__ import annotations

import math
import random
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


def generate_sample(n: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    sources = ["direct", "ads", "organic", "referral", "email"]
    devices = ["desktop", "mobile", "tablet"]
    cities = ["Екатеринбург", "Москва", "Тюмень", "Челябинск", "Пермь"]
    rows = []
    for i in range(1, n + 1):
        source = random.choices(sources, weights=[15, 30, 25, 18, 12])[0]
        device = random.choices(devices, weights=[48, 47, 5])[0]
        city = random.choice(cities)
        pages = random.randint(1, 18)
        mins = round(max(0.5, random.gauss(pages * 1.1, 4)), 1)
        repeat = int(random.random() < 0.30)
        form = int(random.random() < (0.12 + 0.04 * min(pages, 12) / 12 + 0.18 * repeat))
        logit = -2.4
        logit += {"ads": 0.15, "organic": 0.45, "referral": 0.70, "direct": 0.35, "email": 0.55}[source]
        logit += {"desktop": 0.35, "mobile": 0.05, "tablet": 0.15}[device]
        logit += 0.10 * min(pages, 12)
        logit += 0.04 * min(max(mins, 0), 25)
        logit += 0.80 * repeat + 1.20 * form
        if city == "Москва":
            logit += 0.15
        p = 1 / (1 + math.exp(-logit))
        target = int(random.random() < p)
        day = 1 + (i % 28)
        hour = 9 + (i % 10)
        rows.append(
            {
                "lead_id": f"L{i:04d}",
                "created_at": f"2026-08-{day:02d} {hour:02d}:00",
                "source": source,
                "device": device,
                "city": city,
                "pages_viewed": pages,
                "session_minutes": mins,
                "repeat_visit": repeat,
                "form_filled": form,
                "target_deal": target,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample()
    df.to_csv(DATA, index=False)
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
    proba_valid = pipe.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, proba_valid)
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
    full_score = pipe.predict_proba(x)[:, 1]
    groups = pd.cut(
        full_score,
        bins=[-0.01, 0.40, 0.60, 1.01],
        labels=["низкий", "средний", "высокий"],
    )
    print(f"saved {MODEL}")
    print(f"rows={len(df)} valid ROC-AUC={auc:.3f}")
    print("groups:", groups.value_counts().to_dict())


if __name__ == "__main__":
    main()
