"""LNL demos: landing page. Run with `streamlit run Home.py` from this folder."""

import streamlit as st

from lib import auth, theme

st.set_page_config(page_title="LNL Demos", page_icon="▣", layout="wide")
theme.apply()
auth.require_passcode()

theme.bar("LNL ▸ STATISTICAL MODELS ▸ DEMONSTRATIONS", "LOCAL · localhost")

# Demo type -> demos. Each demo: (id, page file, title, what it shows, lecture)
DEMOS = {
    "INFERENCE  ·  CONFIDENCE INTERVALS & HYPOTHESIS TESTS": [
        ("0301", "pages/0301_CI_Coverage.py", "Unbiasedness & CI coverage",
         "Fixed X and β; a new ε each simulation. The average estimate settles on β, "
         "and about 95% of the intervals contain β.",
         "0001-03-01 inference lecture"),
    ],
}

st.markdown('<div class="note">Pick a demo type, then click <b>▶ OPEN DEMO</b>. '
            'On the demo page, set the parameters at the top and click <b>▶ RUN</b>.</div>', unsafe_allow_html=True)

for topic, demos in DEMOS.items():
    theme.side_header(topic)
    cols = st.columns(3)
    for k, (i, path, title, what, lecture) in enumerate(demos):
        with cols[k % 3].container(border=True):
            st.markdown(f'<div class="card-id">{i}  ·  {lecture}</div>'
                        f'<div class="card-title">{title}</div>'
                        f'<div class="card-text">{what}</div>', unsafe_allow_html=True)
            if st.button("▶  OPEN DEMO", key=f"open_{i}", type="primary", width="stretch"):
                st.switch_page(path)
