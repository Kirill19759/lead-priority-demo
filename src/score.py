"""Оценка приоритета заявок по уже обученной модели."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "lead_priority.joblib"
REQUIRED = ["source", "device", "city", "pages_viewed", "session_minutes", "repeat_visit", "form_filled"]


def load_bundle():
    if not MODEL.exists():
        raise FileNotFoundError("Сначала запустите: python src/train.py")
    return joblib.load(MODEL)


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_bundle()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("В таблице нет колонок: " + ", ".join(missing))
    x = df[bundle["features_cat"] + bundle["features_num"]]
    out = df.copy()
    out["priority_score"] = bundle["pipeline"].predict_proba(x)[:, 1]
    out["priority_group"] = pd.cut(
        out["priority_score"],
        bins=[-0.01, 0.40, 0.60, 1.01],
        labels=["низкий", "средний", "высокий"],
    )
    return out.sort_values("priority_score", ascending=False).reset_index(drop=True)
