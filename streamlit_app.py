"""Interactive companion to the static report. Run: `uv run streamlit run streamlit_app.py`.

Everything shown is computed by the growth_engine package on a seeded synthetic event stream, or
on a journeys CSV you upload (user_id, path, converted) for the attribution section. Nothing
here describes a real programme.
"""

from __future__ import annotations

# Community Cloud runs this file from a plain checkout; make the src/ package importable there.
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from growth_engine.attribution import compare, removal_effects, spearman
from growth_engine.cohorts import average_curve, cohort_table
from growth_engine.economics import blended, channel_economics
from growth_engine.events import CHANNEL_EFFECT, load_journeys_csv, synthetic_dataset
from growth_engine.uplift import by_segment, two_proportion

st.set_page_config(page_title="Growth Analytics Engine", layout="wide")
st.title("Growth Analytics Engine")
st.caption("Attribution, cohorts, uplift and unit economics. Synthetic data unless you upload.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Journeys CSV (user_id, path, converted)", type=["csv"])
    seed = st.number_input("Seed (synthetic)", value=42, min_value=0, step=1)
    users = st.slider("Prospects (synthetic)", 500, 20000, 6000, 500)
    months = st.slider("Months (synthetic)", 6, 36, 18, 1)
    horizon = st.slider("LTV horizon (months)", 1, 12, 6, 1)

if upload is not None:
    ds = load_journeys_csv(upload.getvalue().decode("utf-8", errors="replace"), source=upload.name)
    for w in ds.warnings:
        st.warning(w)
    st.info("Uploaded journeys support the attribution section only; the rest needs purchases.")
else:
    ds = synthetic_dataset(n_users=int(users), months=int(months), seed=int(seed))
st.write(f"**Source:** {ds.source} — {len(ds.journeys):,} journeys")

st.header("1. Attribution")
models = compare(ds.journeys)
channels = list(models["markov"])
st.dataframe(
    {"channel": channels, **{m: [round(c[ch], 1) for ch in channels] for m, c in models.items()}},
    hide_index=True,
)
st.bar_chart({"removal effect": removal_effects(ds.journeys)})
if all(c in CHANNEL_EFFECT for c in channels):
    rho = {m: round(spearman(c, CHANNEL_EFFECT), 2) for m, c in models.items()}
    st.write("Rank agreement with the generator's true channel effects (Spearman):", rho)

if ds.purchases:
    st.header("2. Cohorts")
    table = cohort_table(ds.journeys, ds.purchases, ds.months)
    ret = average_curve(table, table.retention)
    ltv = average_curve(table, table.cumulative_revenue)
    st.line_chart({"retention": ret[~np.isnan(ret)], "cumulative revenue": ltv[~np.isnan(ltv)]})

    st.header("3. Uplift")
    ate = two_proportion(ds.journeys)
    st.metric(
        "Average effect (pts)",
        f"{ate.uplift * 100:+.1f}",
        f"95% CI {ate.ci_low * 100:+.1f} to {ate.ci_high * 100:+.1f}",
    )
    seg = by_segment(ds.journeys)
    st.dataframe(
        {
            "segment": list(seg),
            "uplift (pts)": [round(e.uplift * 100, 1) for e in seg.values()],
            "ci low": [round(e.ci_low * 100, 1) for e in seg.values()],
            "ci high": [round(e.ci_high * 100, 1) for e in seg.values()],
            "significant": [e.significant for e in seg.values()],
        },
        hide_index=True,
    )

    st.header("4. Unit economics")
    try:
        rows = channel_economics(ds.spend, models["markov"], table, int(horizon))
    except ValueError as exc:
        st.error(str(exc))
    else:
        rows.append(blended(rows, table))
        st.dataframe(
            {
                "channel": [r.channel for r in rows],
                "spend": [r.spend for r in rows],
                "attributed": [round(r.conversions, 1) for r in rows],
                "CAC": [round(r.cac) if np.isfinite(r.cac) else None for r in rows],
                "LTV": [round(r.ltv) for r in rows],
                "LTV:CAC": [
                    round(r.ltv_to_cac, 2) if np.isfinite(r.ltv_to_cac) else None for r in rows
                ],
                "payback month": [r.payback_month for r in rows],
            },
            hide_index=True,
        )
