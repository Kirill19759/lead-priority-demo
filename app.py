"""Локальное демо: загрузка CSV → приоритет → выгрузка."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.score import REQUIRED, load_bundle, score_frame

SAMPLE = Path(__file__).resolve().parent / "data" / "sample_leads.csv"

st.set_page_config(page_title="tagiltsev_ml · приоритет заявок", layout="wide")
st.title("Приоритет входящих заявок")
st.caption("Демонстрационный сервис tagiltsev_ml. Не промышленное внедрение у конкретного клиента.")

bundle = load_bundle()
st.write(
    f"Модель обучена на учебной выборке демо. "
    f"Качество на отложенной части этой выборки: ROC-AUC {bundle['valid_roc_auc']:.2f}. "
    f"На ваших данных цифра будет другой."
)

st.subheader("Что нужно в файле")
st.code(", ".join(REQUIRED))

uploaded = st.file_uploader("Загрузите CSV с заявками", type=["csv"])
use_sample = st.checkbox("Показать учебный пример", value=uploaded is None)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(SAMPLE)
else:
    st.stop()

st.write(f"Строк во входе: {len(df)}")

try:
    ranked = score_frame(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Высокий", int((ranked["priority_group"] == "высокий").sum()))
c2.metric("Средний", int((ranked["priority_group"] == "средний").sum()))
c3.metric("Низкий", int((ranked["priority_group"] == "низкий").sum()))
c4.metric("Средняя оценка", f"{ranked['priority_score'].mean():.2f}")

st.dataframe(ranked.head(30), use_container_width=True)
st.download_button(
    "Скачать ранжированный список",
    ranked.to_csv(index=False).encode("utf-8-sig"),
    file_name="leads_priority.csv",
    mime="text/csv",
)
