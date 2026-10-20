"""Plotly figures shared by the demos."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .theme import C, PARAM_COLORS, style_figure


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _segments(x, lo, hi):
    """Vertical segments x=const from lo to hi, separated by None for a single trace."""
    xs = np.repeat(x, 3).astype(object)
    ys = np.column_stack([lo, hi, np.full(len(lo), np.nan)]).ravel().astype(object)
    xs[2::3] = None
    ys[2::3] = None
    return xs, ys


def _waiting(fig):
    """Message shown before the first simulation, so an empty chart doesn't look broken."""
    fig.add_annotation(text="NO SIMULATIONS YET  ▸  PRESS ▶ RUN OR +1 IN THE SIDEBAR", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(color=C["amber"], size=14),
                       bgcolor="rgba(0,0,0,0.85)", bordercolor=C["amber"], borderwidth=1, borderpad=8)


def estimates_grid(est, se, L, U, hit, beta, names, total, level, df, ref="both"):
    """6 x p grid, one column per coefficient. Rows 1-3 share the simulation axis.
    Row 1: each estimate (dots), cumulative average, true beta.
    Row 2: each simulation's CI as a vertical bar (green covers, red misses) around the true beta.
    Row 3: running coverage, with a +/- 2 binomial-SE band around the confidence level.
    Row 4: histogram of (estimate - beta) / SE over the simulations so far, with the reference density
    ref = "t" (t with df degrees of freedom), "z" (standard normal), or "both".
    Rows 5-6: the left and right tails of row 4, zoomed, always with both densities and the z and t
    critical values, plus the share beyond the z critical value: simulated, under t, and under N(0, 1)."""
    k = len(names)
    S = est.shape[0]
    ref_label = {"t": f"t({df})", "z": "N(0, 1)", "both": f"t({df}) & N(0, 1)"}[ref]
    titles = [f"{nm}  ▸  ESTIMATES & CUMULATIVE AVERAGE  vs  TRUE {nm}" for nm in names] + \
             [f"{nm}  ▸  {level:.0%} CONFIDENCE INTERVALS" for nm in names] +              [f"{nm}  ▸  RUNNING COVERAGE  (TARGET {level:.0%} ± 2 SE)" for nm in names] +              [f"{nm}  ▸  (ESTIMATE − TRUE {nm}) / SE  vs  {ref_label}" for nm in names] +              [f"{nm}  ▸  LEFT TAIL  ·  t({df}) vs N(0, 1)" for nm in names] +              [f"{nm}  ▸  RIGHT TAIL  ·  t({df}) vs N(0, 1)" for nm in names]
    fig = make_subplots(rows=6, cols=k, vertical_spacing=0.05, horizontal_spacing=0.05,
                        subplot_titles=titles, row_heights=[0.19, 0.19, 0.15, 0.17, 0.15, 0.15])
    alpha2 = (1.0 - level) / 2.0
    z_crit, t_crit = stats.norm.ppf(1.0 - alpha2), stats.t.ppf(1.0 - alpha2, df)
    edges = np.arange(-5.0, 5.0001, 0.2)
    mids = (edges[:-1] + edges[1:]) / 2
    tails = {5: (-5.0, -1.4), 6: (1.4, 5.0)}
    notes = []
    grid = np.linspace(-5, 5, 401)
    s_all = np.arange(1, total + 1)
    band = 2.0 * np.sqrt(level * (1.0 - level) / s_all)
    sims = np.arange(1, S + 1)
    bar_w = float(np.clip(700.0 / max(total, 1), 0.5, 2.5))     # thinner bars when there are many simulations

    for j in range(k):
        col = j + 1
        color = PARAM_COLORS[j]
        if S:
            cum = np.cumsum(est[:, j]) / sims
            fig.add_trace(go.Scatter(x=sims, y=est[:, j], mode="markers",
                                       marker=dict(size=3, color=_rgba(color, 0.35)),
                                       hovertemplate="sim %{x}<br>estimate = %{y:.4f}<extra></extra>"), row=1, col=col)
            fig.add_trace(go.Scatter(x=sims, y=cum, mode="lines", line=dict(color=color, width=2.2),
                                     hovertemplate="sim %{x}<br>avg estimate = %{y:.4f}<extra></extra>"), row=1, col=col)
            fig.add_trace(go.Scatter(x=[S], y=[cum[-1]], mode="markers", marker=dict(size=7, color=color),
                                     hovertemplate="sim %{x}<br>avg estimate = %{y:.4f}<extra></extra>"), row=1, col=col)

            h = hit[:, j]
            for mask, c in ((h, C["green"]), (~h, C["red"])):
                if mask.any():
                    xs, ys = _segments(sims[mask], L[mask, j], U[mask, j])
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_rgba(c, 0.55 if c == C["green"] else 0.95), width=bar_w),
                                               hoverinfo="skip"), row=2, col=col)
            fig.add_trace(go.Scatter(x=sims, y=est[:, j], mode="markers", marker=dict(size=2, color=_rgba(C["txt"], 0.6)),
                                       hovertemplate="sim %{x}<br>estimate = %{y:.4f}<extra></extra>"), row=2, col=col)

        fig.add_trace(go.Scatter(x=s_all, y=np.minimum(level + band, 1.0), mode="lines", line=dict(width=0),
                                 hoverinfo="skip"), row=3, col=col)
        fig.add_trace(go.Scatter(x=s_all, y=np.maximum(level - band, 0.0), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_rgba(C["amber"], 0.10), hoverinfo="skip"), row=3, col=col)
        fig.add_hline(y=level, line=dict(color=C["amber"], width=1.2, dash="dot"), row=3, col=col)
        if S:
            cov = np.cumsum(hit[:, j]) / sims
            fig.add_trace(go.Scatter(x=sims, y=cov, mode="lines", line=dict(color=color, width=2),
                                     hovertemplate="sim %{x}<br>coverage = %{y:.1%}<extra></extra>"), row=3, col=col)
            fig.add_trace(go.Scatter(x=[S], y=[cov[-1]], mode="markers", marker=dict(size=7, color=color),
                                     hovertemplate="sim %{x}<br>coverage = %{y:.1%}<extra></extra>"), row=3, col=col)
        fig.update_yaxes(range=[max(0.0, level - 0.35), 1.01], tickformat=".0%", row=3, col=col)

        if S:
            tstat = (est[:, j] - beta[j]) / se[:, j]
            fig.add_trace(go.Histogram(x=tstat, histnorm="probability density", xbins=dict(start=-5, end=5, size=0.25),
                                       marker=dict(color=_rgba(color, 0.55), line=dict(color=C["panel"], width=0.5)),
                                       hovertemplate="%{x}<br>density = %{y:.3f}<extra></extra>"), row=4, col=col)
        if ref in ("t", "both"):
            fig.add_trace(go.Scatter(x=grid, y=stats.t.pdf(grid, df), mode="lines", line=dict(color=C["txt"], width=1.8),
                                     hovertemplate=f"t({df}) pdf<br>%{{x:.2f}}: %{{y:.3f}}<extra></extra>"), row=4, col=col)
        if ref in ("z", "both"):
            fig.add_trace(go.Scatter(x=grid, y=stats.norm.pdf(grid), mode="lines",
                                     line=dict(color=C["amber"], width=1.6, dash="dash" if ref == "both" else "solid"),
                                     hovertemplate="N(0,1) pdf<br>%{x:.2f}: %{y:.3f}<extra></extra>"), row=4, col=col)
        fig.update_xaxes(range=[-5, 5], row=4, col=col)
        fig.update_yaxes(range=[0, 0.62], row=4, col=col)

        tstat = (est[:, j] - beta[j]) / se[:, j] if S else np.array([])
        dens = np.histogram(tstat, bins=edges)[0] / (max(S, 1) * 0.2)    # density relative to all simulations
        for row, (lo, hi) in tails.items():
            xs = np.linspace(lo, hi, 200)
            pt, pz = stats.t.pdf(xs, df), stats.norm.pdf(xs)
            keep = (mids > lo) & (mids < hi)
            if S:
                fig.add_trace(go.Bar(x=mids[keep], y=dens[keep], width=0.2, marker=dict(color=_rgba(color, 0.55),
                                     line=dict(color=C["panel"], width=0.5)),
                                     hovertemplate="%{x:.1f}<br>density = %{y:.4f}<extra></extra>"), row=row, col=col)
            fig.add_trace(go.Scatter(x=xs, y=pt, mode="lines", line=dict(color=C["txt"], width=1.8),
                                     hovertemplate=f"t({df}) pdf<br>%{{x:.2f}}: %{{y:.4f}}<extra></extra>"), row=row, col=col)
            fig.add_trace(go.Scatter(x=xs, y=pz, mode="lines", line=dict(color=C["amber"], width=1.6, dash="dash"),
                                     hovertemplate="N(0,1) pdf<br>%{x:.2f}: %{y:.4f}<extra></extra>"), row=row, col=col)
            sign = -1 if row == 5 else 1
            fig.add_vline(x=sign * z_crit, line=dict(color=C["amber"], width=1, dash="dot"), row=row, col=col)
            fig.add_vline(x=sign * t_crit, line=dict(color=C["txt"], width=1, dash="dot"), row=row, col=col)
            ymax = max(pt.max(), pz.max(), dens[keep].max() if S else 0.0)
            fig.update_xaxes(range=[lo, hi], row=row, col=col)
            fig.update_yaxes(range=[0, 1.15 * ymax], row=row, col=col)
            emp = (np.mean(tstat < -z_crit) if row == 5 else np.mean(tstat > z_crit)) if S else np.nan
            rel = "<" if row == 5 else ">"
            emp_txt = "—" if not S else f"{emp:.1%}"
            notes.append(dict(axis=(row - 1) * k + col, row=row, text=(
                f"SHARE {rel} {sign * z_crit:+.3f} (z crit)<br>SIM {emp_txt}  ·  t({df}) {stats.t.sf(z_crit, df):.1%}"
                f"  ·  N(0,1) {alpha2:.1%}<br>t crit {sign * t_crit:+.3f}  ·  z crit {sign * z_crit:+.3f}")))

        for row in (1, 2):
            fig.add_hline(y=beta[j], line=dict(color=C["txt"], width=1.2, dash="dot"), row=row, col=col)
        fig.add_annotation(x=1, y=beta[j], xref=f"x{col if col > 1 else ''} domain", yref=f"y{col if col > 1 else ''}",
                           text=f"TRUE {beta[j]:g}", showarrow=False, xanchor="right", yanchor="bottom",
                           font=dict(color=C["txt"], size=10))

    if not S:
        _waiting(fig)
    for row in (1, 2, 3):
        fig.update_xaxes(range=[0, total + 1], row=row)
        for col in range(1, k + 1):     # rows 1-3 zoom together within each column
            if row < 3:
                fig.update_xaxes(matches=f"x{2 * k + col}", showticklabels=False, row=row, col=col)
    fig.update_xaxes(title_text="simulation", title_font=dict(size=10, color=C["dim"]), row=3)
    for row in (4, 5, 6):
        fig.update_xaxes(title_text="(estimate − true β) / SE", title_font=dict(size=10, color=C["dim"]), row=row)
    fig = style_figure(fig, height=1900)
    fig.update_layout(barmode="overlay")
    for nt in notes:     # added after style_figure so they keep their own font
        ax = "" if nt["axis"] == 1 else nt["axis"]
        left = nt["row"] == 6     # right-tail note sits on the left of the panel, and vice versa
        fig.add_annotation(xref=f"x{ax} domain", yref=f"y{ax} domain", x=0.02 if left else 0.98, y=0.97,
                           xanchor="left" if left else "right", yanchor="top", text=nt["text"], showarrow=False,
                           align="left" if left else "right", font=dict(size=10, color=C["txt"]),
                           bgcolor="rgba(0,0,0,0.6)")
    return fig


def coverage_figure(hit, names, level, total):
    """Running share of intervals that contain the true beta, with a +/- 2 binomial-SE band around the level."""
    S = hit.shape[0]
    fig = go.Figure()
    s = np.arange(1, total + 1)
    band = 2.0 * np.sqrt(level * (1.0 - level) / s)
    fig.add_trace(go.Scatter(x=s, y=np.minimum(level + band, 1.0), mode="lines", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s, y=np.maximum(level - band, 0.0), mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=_rgba(C["amber"], 0.10), hoverinfo="skip"))
    fig.add_hline(y=level, line=dict(color=C["amber"], width=1.2, dash="dot"))
    if S:
        sims = np.arange(1, S + 1)
        for j, nm in enumerate(names):
            cov = np.cumsum(hit[:, j]) / sims
            fig.add_trace(go.Scatter(x=sims, y=cov, mode="lines", name=nm, line=dict(color=PARAM_COLORS[j], width=2),
                                     hovertemplate=f"{nm}<br>sim %{{x}}<br>coverage %{{y:.1%}}<extra></extra>"))
    else:
        _waiting(fig)
    fig.update_xaxes(range=[0, total + 1], title_text="simulation", title_font=dict(size=10, color=C["dim"]))
    fig.update_yaxes(range=[max(0.0, level - 0.35), 1.01], tickformat=".0%")
    fig.update_layout(title=dict(text=f"RUNNING COVERAGE  ▸  SHARE OF INTERVALS CONTAINING THE TRUE β  (TARGET {level:.0%} ± 2 SE)",
                                 font=dict(size=11, color=C["amber"]), x=0.01, y=0.96))
    fig = style_figure(fig, height=300)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=1, xanchor="right", y=1.12, font=dict(size=11)))
    return fig
