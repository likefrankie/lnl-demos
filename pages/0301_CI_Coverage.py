"""Demo 0301: repeated samples for a fixed X.

Each simulation keeps X and beta fixed, draws a new error vector, refits OLS, and builds
the confidence intervals. Over many simulations the average of the estimates approaches
the true beta (E[beta_hat] = beta), and about (1 - alpha) of the intervals contain it.
"""

import time
from datetime import datetime

import numpy as np
import streamlit as st

from lib import auth, plots, simulate, theme

st.set_page_config(page_title="CI Coverage · LNL Demos", page_icon="▣", layout="wide")
theme.apply()
auth.require_passcode()

MODEL = simulate.MODELS["mlr2"]
S = st.session_state

# --------------------------------------------------------------------- controls
st.sidebar.page_link("Home.py", label="◂  ALL DEMOS")
ph_bar = st.empty()

with st.container(border=True):
    st.markdown(f'<div class="side-h">SIMULATION PARAMETERS  ▸  MODEL  {MODEL.formula}</div>', unsafe_allow_html=True)
    c = st.columns([1, 1, 1, 1, 1, 1, 1.6])
    b0 = c[0].number_input("β₀  intercept (true)", value=1.0, step=0.5, format="%.2f")
    b1 = c[1].number_input("β₁  slope on x₁ (true)", value=2.0, step=0.5, format="%.2f")
    b2 = c[2].number_input("β₂  slope on x₂ (true)", value=-0.5, step=0.5, format="%.2f")
    sigma = c[3].number_input("σ  error std. deviation", min_value=0.1, max_value=50.0, value=3.0, step=0.5, format="%.2f")
    n = int(c[4].number_input("n  sample size", min_value=5, max_value=2000, value=30, step=5))
    total = int(c[5].number_input("number of simulations", min_value=10, max_value=5000, value=500, step=50))
    with c[6]:
        run = st.button("▶  RUN", type="primary", width="stretch")
        s1, s2 = st.columns(2)
        step = s1.button("+1", width="stretch")
        reset = s2.button("RESET", width="stretch")

    with st.expander("MORE SETTINGS  ·  corr(x₁, x₂), seed, confidence level, critical value, speed, reference density"):
        e = st.columns(6)
        rho = e[0].slider("ρ  corr(x₁, x₂)", -0.95, 0.95, 0.30, 0.05)
        seed = int(e[1].number_input("seed (fixes X)", value=42, step=1))
        level = e[2].select_slider("confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
                                   format_func=lambda v: f"{v:.0%}")
        crit_kind = e[3].radio("critical value", ["t", "z"], horizontal=True,
                               format_func=lambda k: "t  (n − p − 1 df)" if k == "t" else "z  (normal)")
        delay = e[4].slider("pause per simulation (s)", 0.0, 1.0, 0.05, 0.01)
        ref = e[5].radio("reference density (row 4)", ["both", "t", "z"], horizontal=True,
                         format_func=lambda r: {"t": "t(n−p−1)", "z": "N(0,1)", "both": "both"}[r])
    st.markdown('<div class="note">RUN draws samples until the number of simulations is reached; +1 draws one. '
                'Any click stops a run. Changing a parameter starts over.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------- state
beta = np.array([b0, b1, b2])
cfg = (b0, b1, b2, sigma, n, rho, seed, level, crit_kind, total)
if reset or S.get("cfg") != cfg:
    S.cfg = cfg
    S.X = MODEL.design(n, rho=rho, seed=seed)
    S.rng = np.random.default_rng(seed + 1)
    S.est, S.se, S.L, S.U, S.hit = [], [], [], [], []
    S.last = None

X = S.X
p = X.shape[1] - 1
df = n - p - 1
crit = simulate.critical_value(level, df, crit_kind)
crit_label = f"t({df}) = {crit:.3f}" if crit_kind == "t" else f"z = {crit:.3f}"


def advance():
    d = simulate.draw_and_fit(X, beta, sigma, S.rng, crit)
    S.est.append(d.b)
    S.se.append(d.se)
    S.L.append(d.L)
    S.U.append(d.U)
    S.hit.append(d.hit)
    S.last = d


# --------------------------------------------------------------------- layout
ph_kpi = st.empty()
left, right = st.columns([5, 7], gap="small")
with left:
    ph_data = st.empty()
with right:
    ph_ci = st.empty()
    ph_cov = st.empty()
ph_grid = st.empty()
st.markdown(
    '<div class="note"><b>HOW TO READ.</b> X and β are fixed. Each simulation draws a new error vector ε, so y = Xβ + ε '
    'changes, and so do the estimates, their standard errors, and the interval estimate ± crit · SE. Top row: the cumulative average of the estimates settles on '
    'the true β, illustrating that the OLS estimator is unbiased, E[estimate] = β. Bottom row: each bar is one simulation\'s interval; about '
    f'{level:.0%} of the bars cover the true β (green) and the rest miss it (red). Third row: the running share of intervals that cover β, which settles inside the shaded band around {level:.0%}. Fourth row: the histogram of (estimate − β) / SE across simulations, which follows the t distribution with n − p − 1 df; with a small n it has heavier tails than N(0, 1), which is why z intervals under-cover. Fifth and sixth rows: the left and right tails zoomed in, with the z and t critical values marked and the share of draws beyond the z critical value. Switch the critical value to z with a '
    'small n to see the intervals under-cover.</div>', unsafe_allow_html=True)


def fmt(v, d=3):
    return f"{v:,.{d}f}"


def data_table():
    k = len(S.est)
    idx = list(range(n)) if n <= 10 else list(range(5)) + [None] + list(range(n - 5, n))
    head = "".join(f'<th>{h}</th>' for h in ["OBS", *MODEL.x_names, "ε", "y"])
    rows = []
    for i in idx:
        if i is None:
            rows.append(f'<tr class="gap"><td colspan="{p + 3}">· · ·  {n - 10} rows  · · ·</td></tr>')
            continue
        xs = "".join(f'<td class="fixed">{fmt(X[i, j + 1])}</td>' for j in range(p))
        if S.last is None:
            e = yv = '<td class="fixed">—</td>'
        else:
            e = f'<td class="fresh" style="color:#ffa028">{fmt(S.last.eps[i])}</td>'
            yv = f'<td class="fresh" style="color:#ffa028">{fmt(S.last.y[i])}</td>'
        rows.append(f'<tr><td class="l fixed">{i + 1}</td>{xs}{e}{yv}</tr>')
    body = f'<table class="term"><tr>{head}</tr>{"".join(rows)}</table>'
    return theme.panel(f"SAMPLE  ▸  SIM #{k:,}", body, right="X FIXED  ·  ε, y REDRAWN")


def ci_table():
    k = len(S.est)
    head = "".join(f'<th{" class=l" if h == "PARAM" else ""}>{h}</th>'
                   for h in ["PARAM", "TRUE β", "ESTIMATE", "SE", "L", "U", "WIDTH", "CONTAINS β"])
    rows = []
    for j, nm in enumerate(MODEL.names):
        if S.last is None:
            cells = "".join('<td class="fixed">—</td>' for _ in range(5)) + '<td class="fixed">—</td>'
        else:
            d = S.last
            cells = (f'<td class="fresh" style="color:#ffa028">{fmt(d.b[j], 4)}</td><td class="fresh" style="color:#ffa028">{fmt(d.se[j], 4)}</td>'
                     f'<td class="fresh" style="color:#ffa028">{fmt(d.L[j], 4)}</td><td class="fresh" style="color:#ffa028">{fmt(d.U[j], 4)}</td>'
                     f'<td>{fmt(d.U[j] - d.L[j], 4)}</td><td>{theme.tag(bool(d.hit[j]), "YES", "NO")}</td>')
        rows.append(f'<tr><td class="l" style="color:{theme.PARAM_COLORS[j]}">{nm}</td><td>{fmt(beta[j], 4)}</td>{cells}</tr>')
    sig = "—" if S.last is None else fmt(S.last.sigma_hat, 4)
    body = f'<table class="term"><tr>{head}</tr>{"".join(rows)}</table>'
    return theme.panel(f"ESTIMATES & {level:.0%} CONFIDENCE INTERVALS  ▸  SIM #{k:,}", body,
                       right=f"σ EST = {sig}  ·  σ TRUE = {sigma:g}  ·  {crit_label}")


def coverage_table():
    k = len(S.est)
    head = "".join(f'<th{" class=l" if h == "PARAM" else ""}>{h}</th>'
                   for h in ["PARAM", "TRUE β", "AVG ESTIMATE", "AVG − TRUE", "HITS", "SIMS", "COVERAGE", "STATUS"])
    rows = []
    band = 2.0 * np.sqrt(level * (1 - level) / k) if k else np.nan
    for j, nm in enumerate(MODEL.names):
        if k == 0:
            cells = "".join('<td class="fixed">—</td>' for _ in range(6))
        else:
            est = np.array(S.est)[:, j]
            hits = int(np.array(S.hit)[:, j].sum())
            cov = hits / k
            ok = abs(cov - level) <= band
            status = theme.tag(ok, "IN BAND", "OFF BAND").replace("tag out", "tag warn")
            cells = (f'<td>{fmt(est.mean(), 4)}</td><td>{est.mean() - beta[j]:+.4f}</td><td>{hits:,}</td><td>{k:,}</td>'
                     f'<td style="color:{theme.C["amber"]}">{cov:.1%}</td><td>{status}</td>')
        rows.append(f'<tr><td class="l" style="color:{theme.PARAM_COLORS[j]}">{nm}</td><td>{fmt(beta[j], 4)}</td>{cells}</tr>')
    body = f'<table class="term"><tr>{head}</tr>{"".join(rows)}</table>'
    right = f"TARGET {level:.0%}" + ("" if k == 0 else f"  ±  {band:.1%}  (2 SE)")
    return theme.panel("RUNNING TOTALS  ▸  ALL SIMULATIONS", body, right=right)


def render(status, charts=True):
    k = len(S.est)
    now = datetime.now().strftime("%H:%M:%S")
    ph_bar.markdown(
        f'<div class="term-bar"><span>LNL ▸ 0301 ▸ UNBIASEDNESS &amp; CONFIDENCE INTERVAL COVERAGE</span>'
        f'<span>SIM {k:,} / {total:,}  ●  {status}  ●  {now}</span></div>', unsafe_allow_html=True)
    ph_kpi.markdown(theme.kpis([
        ("MODEL", "MLR · p = 2", False),
        ("n", f"{n}", False),
        ("DF  n−p−1", f"{df}", False),
        ("LEVEL", f"{level:.0%}", True),
        ("CRITICAL", crit_label, True),
        ("σ", f"{sigma:g}", False),
        ("ρ(x₁,x₂)", f"{rho:+.2f}", False),
        ("SIMULATIONS", f"{k:,}", True),
    ]), unsafe_allow_html=True)
    ph_data.markdown(data_table(), unsafe_allow_html=True)
    ph_ci.markdown(ci_table(), unsafe_allow_html=True)
    ph_cov.markdown(coverage_table(), unsafe_allow_html=True)

    if not charts:
        return
    names = MODEL.names
    est = np.array(S.est).reshape(-1, p + 1)
    L = np.array(S.L).reshape(-1, p + 1)
    U = np.array(S.U).reshape(-1, p + 1)
    se = np.array(S.se).reshape(-1, p + 1)
    hit = np.array(S.hit, dtype=bool).reshape(-1, p + 1)
    ph_grid.plotly_chart(plots.estimates_grid(est, se, L, U, hit, beta, names, total, level, df, ref),
                         width="stretch", config={"displayModeBar": False}, key=f"grid_{k}_{status}")


# --------------------------------------------------------------------- run
if step and len(S.est) < total:
    advance()

if run and len(S.est) < total:
    every = max(1, total // 50)     # redraw the charts about 50 times per run; tables update every simulation
    while len(S.est) < total:
        advance()
        k = len(S.est)
        render("RUNNING", charts=(k % every == 0 or k == total))
        if delay:
            time.sleep(delay)

render("COMPLETE" if len(S.est) >= total else "IDLE")
