"""Static HTML report — the Pages artefact. Every number on the page is computed by this module
from the dataset it describes and embedded as JSON for the Executive Shell's KPI strip. No
inline script or style; a small external script (report.js) mounts the shell."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template

import numpy as np

from .attribution import compare, removal_effects, spearman
from .cohorts import CohortTable, average_curve, cohort_table
from .economics import ChannelEconomics, blended, channel_economics
from .events import CHANNEL_EFFECT, Dataset, synthetic_dataset
from .uplift import Effect, by_segment, qini, two_proportion

SHELL_DIR = Path(__file__).parent / "shell"
REPORT_JS = Path(__file__).parent / "report.js"
TEMPLATES = Path(__file__).parent / "templates"
LTV_HORIZON = 6


@dataclass(frozen=True)
class Analysis:
    dataset: Dataset
    attribution: dict[str, dict[str, float]]
    removal: dict[str, float]
    cohorts: CohortTable
    ate: Effect
    segments: dict[str, Effect]
    qini: list[tuple[float, float]]
    economics: list[ChannelEconomics]
    blended: ChannelEconomics
    truth_agreement: dict[str, float]  # Spearman rank correlation of each model with the generator


def analyse(ds: Dataset, horizon: int = LTV_HORIZON) -> Analysis:
    table = cohort_table(ds.journeys, ds.purchases, ds.months)
    attribution = compare(ds.journeys)
    econ = channel_economics(ds.spend, attribution["markov"], table, horizon)
    return Analysis(
        dataset=ds,
        attribution=attribution,
        removal=removal_effects(ds.journeys),
        cohorts=table,
        ate=two_proportion(ds.journeys),
        segments=by_segment(ds.journeys),
        qini=qini(ds.journeys),
        economics=econ,
        blended=blended(econ, table),
        truth_agreement={
            m: spearman(credit, CHANNEL_EFFECT)
            for m, credit in attribution.items()
            if all(c in CHANNEL_EFFECT for c in credit)
        },
    )


def _money(v: float) -> str:
    return "&infin;" if not np.isfinite(v) else f"${v:,.0f}"


def _pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


def _row(cells: list[str], head: bool = False) -> str:
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def _label(name: str) -> str:
    return html.escape(name.replace("_", " "))


def _bars(values: dict[str, float], unit: str, width: int = 380) -> str:
    """Horizontal bar chart; one row per key, labelled with the value."""
    row_h, pad_l, pad_r = 26, 96, 44
    height = row_h * len(values) + 8
    top = max(values.values(), default=0.0) or 1.0
    rows = []
    for i, (k, v) in enumerate(values.items()):
        y = 4 + i * row_h
        w = (width - pad_l - pad_r) * v / top
        rows.append(
            f'<text class="gae-tick" x="{pad_l - 6}" y="{y + 16}" text-anchor="end">'
            f"{_label(k)}</text>"
            f'<rect class="gae-bar" x="{pad_l}" y="{y + 4}" width="{w:.1f}" height="{row_h - 10}"/>'
            f'<text class="gae-tick" x="{pad_l + w + 4:.1f}" y="{y + 16}">'
            f"{v:.3f}{unit}</text>"
        )
    return (
        f'<svg class="gae-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Bar chart">{"".join(rows)}</svg>'
    )


def _lines(
    series: list[tuple[str, list[tuple[float, float]], str]],
    x_label: str,
    y_fmt: str = "{:.0f}",
    width: int = 720,
    height: int = 240,
) -> str:
    pad = 40
    xs = [x for _, pts, _ in series for x, _ in pts]
    ys = [y for _, pts, _ in series for _, y in pts]
    x_lo, x_hi = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y_lo, y_hi = min(0.0, min(ys)), max(ys) if max(ys) > min(0.0, min(ys)) else 1.0

    def sx(x: float) -> float:
        return pad + (x - x_lo) / (x_hi - x_lo) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - y_lo) / (y_hi - y_lo) * (height - 2 * pad)

    paths = []
    for label, pts, cls in series:
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        safe = html.escape(label)
        paths.append(f'<polyline class="{cls}" points="{d}"><title>{safe}</title></polyline>')
    axis = (
        f'<path class="gae-axis" d="M{pad},{pad} L{pad},{height - pad} '
        f'L{width - pad},{height - pad}"/>'
    )
    labels = (
        f'<text class="gae-tick" x="{pad - 4}" y="{pad + 4}" text-anchor="end">'
        f"{y_fmt.format(y_hi)}</text>"
        f'<text class="gae-tick" x="{pad - 4}" y="{height - pad + 4}" text-anchor="end">'
        f"{y_fmt.format(y_lo)}</text>"
        f'<text class="gae-tick" x="{width - pad}" y="{height - pad + 14}" text-anchor="end">'
        f"{html.escape(x_label)}</text>"
    )
    return (
        f'<svg class="gae-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Line chart">{axis}{"".join(paths)}{labels}</svg>'
    )


def _attribution_table(a: Analysis) -> str:
    models = list(a.attribution)
    channels = list(a.attribution["markov"])
    head = _row(["Channel", *[m.replace("_", " ") for m in models], "Markov vs last"], head=True)
    body = []
    for c in channels:
        last, mk = a.attribution["last_touch"][c], a.attribution["markov"][c]
        delta = (mk - last) / last if last > 0 else float("inf")
        body.append(
            _row(
                [
                    _label(c),
                    *[f"{a.attribution[m][c]:.1f}" for m in models],
                    "n/a" if not np.isfinite(delta) else f"{delta * 100:+.0f}%",
                ]
            )
        )
    total = sum(a.attribution["markov"].values())
    body.append(_row(["<strong>Total</strong>", *[f"{total:.0f}"] * len(models), ""]))
    if a.truth_agreement:
        body.append(
            _row(
                [
                    "Rank agreement with generator truth (Spearman &rho;)",
                    *[f"{a.truth_agreement[m]:+.2f}" for m in models],
                    "",
                ]
            )
        )
    return f'<table class="gae-table"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def _cohort_heatmap(t: CohortTable, max_cols: int = 12) -> str:
    cols = min(max_cols, t.retention.shape[1])
    head = _row(["Cohort", "Users", *[f"M{k}" for k in range(cols)]], head=True)
    body = []
    for i, c in enumerate(t.cohorts):
        cells = []
        for k in range(cols):
            v = t.retention[i, k]
            if np.isnan(v):
                cells.append('<td class="gae-heat gae-heat--na">&middot;</td>')
            else:
                level = min(9, int(v * 10))
                cells.append(f'<td class="gae-heat" data-level="{level}">{v * 100:.0f}%</td>')
        body.append(f"<tr><td>month {c}</td><td>{t.sizes[i]}</td>{''.join(cells)}</tr>")
    return (
        f'<table class="gae-table gae-table--heat"><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _uplift_table(segments: dict[str, Effect]) -> str:
    head = _row(
        ["Segment", "Treated", "Control", "Rate T", "Rate C", "Uplift", "95% CI"], head=True
    )
    body = "".join(
        _row(
            [
                _label(s),
                str(e.n_treated),
                str(e.n_control),
                _pct(e.rate_treated),
                _pct(e.rate_control),
                f"{e.uplift * 100:+.1f} pts",
                f"[{e.ci_low * 100:+.1f}, {e.ci_high * 100:+.1f}]"
                + (" &check;" if e.significant else ""),
            ]
        )
        for s, e in segments.items()
    )
    return f'<table class="gae-table"><thead>{head}</thead><tbody>{body}</tbody></table>'


def _econ_table(rows: list[ChannelEconomics], blend: ChannelEconomics, horizon: int) -> str:
    head = _row(
        [
            "Channel",
            "Spend",
            "Attributed (Markov)",
            "CAC",
            f"LTV at M{horizon}",
            "LTV:CAC",
            "Payback",
        ],
        head=True,
    )
    body = []
    for r in [*rows, blend]:
        pb = "not within window" if r.payback_month is None else f"month {r.payback_month}"
        cls = ' class="gae-blend"' if r.channel == "blended" else ""
        body.append(
            f"<tr{cls}>"
            + "".join(
                f"<td>{c}</td>"
                for c in [
                    _label(r.channel),
                    _money(r.spend),
                    f"{r.conversions:.1f}",
                    _money(r.cac),
                    _money(r.ltv),
                    "&infin;" if not np.isfinite(r.ltv_to_cac) else f"{r.ltv_to_cac:.2f}",
                    pb,
                ]
            )
            + "</tr>"
        )
    return f'<table class="gae-table"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def _kpi_json(a: Analysis, horizon: int) -> str:
    last, mk = a.attribution["last_touch"], a.attribution["markov"]
    swing = max(mk, key=lambda c: abs(mk[c] - last[c]))
    return json.dumps(
        {
            "users": len(a.dataset.journeys),
            "converted": sum(1 for j in a.dataset.journeys if j.converted),
            "swingChannel": swing,
            "swingLast": round(last[swing], 1),
            "swingMarkov": round(mk[swing], 1),
            "ate": round(a.ate.uplift, 4),
            "ateLow": round(a.ate.ci_low, 4),
            "ateHigh": round(a.ate.ci_high, 4),
            "bestSegment": max(a.segments, key=lambda s: a.segments[s].uplift),
            "bestSegmentUplift": round(max(e.uplift for e in a.segments.values()), 4),
            "ltvToCac": round(a.blended.ltv_to_cac, 2),
            "ltvHorizon": horizon,
            "truthAgreement": {m: round(v, 2) for m, v in a.truth_agreement.items()},
            "worstChannel": min(
                a.economics, key=lambda r: r.ltv_to_cac if np.isfinite(r.ltv_to_cac) else 1e9
            ).channel,
        }
    )


def render_html(a: Analysis, pages: str, horizon: int = LTV_HORIZON) -> str:
    tmpl = Template((TEMPLATES / "page.html").read_text(encoding="utf-8"))
    ds = a.dataset
    ret = average_curve(a.cohorts, a.cohorts.retention)
    ltv = average_curve(a.cohorts, a.cohorts.cumulative_revenue)
    ret_pts = [(float(k), float(v)) for k, v in enumerate(ret) if not np.isnan(v)]
    ltv_pts = [(float(k), float(v)) for k, v in enumerate(ltv) if not np.isnan(v)]
    q = a.qini
    random_line = [(0.0, 0.0), (1.0, q[-1][1])]
    return tmpl.substitute(
        pages=html.escape(pages),
        kpi_json=_kpi_json(a, horizon),
        source=html.escape(ds.source),
        users=f"{len(ds.journeys):,}",
        converted=f"{sum(1 for j in ds.journeys if j.converted):,}",
        months=ds.months,
        purchases=f"{len(ds.purchases):,}",
        attribution_table=_attribution_table(a),
        removal_chart=_bars(a.removal, ""),
        heatmap=_cohort_heatmap(a.cohorts),
        retention_chart=_lines(
            [("retention", ret_pts, "gae-line gae-line--ok")], "months since acquisition", "{:.0%}"
        ),
        ltv_chart=_lines(
            [("cumulative revenue per user", ltv_pts, "gae-line gae-line--accent")],
            "months since acquisition",
            "${:,.0f}",
        ),
        ate=f"{a.ate.uplift * 100:+.1f}",
        ate_low=f"{a.ate.ci_low * 100:+.1f}",
        ate_high=f"{a.ate.ci_high * 100:+.1f}",
        n_treated=f"{a.ate.n_treated:,}",
        n_control=f"{a.ate.n_control:,}",
        uplift_table=_uplift_table(a.segments),
        qini_chart=_lines(
            [
                ("targeted by segment uplift", q, "gae-line gae-line--accent"),
                ("random targeting", random_line, "gae-line gae-line--muted"),
            ],
            "share of population targeted",
            "{:.0f}",
        ),
        horizon=horizon,
        econ_table=_econ_table(a.economics, a.blended, horizon),
    )


def write_report(
    out: Path,
    seed: int = 42,
    n_users: int = 6000,
    months: int = 18,
    pages: str = "https://freddricklogan.github.io/growth-analytics-engine/",
) -> Path:
    """Generate the report into `out/` (index.html + src/ assets + report.json)."""
    ds = synthetic_dataset(n_users=n_users, months=months, seed=seed)
    a = analyse(ds)
    out.mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    shutil.copy(SHELL_DIR / "exec-shell.css", out / "src" / "exec-shell.css")
    shutil.copy(SHELL_DIR / "exec-shell.js", out / "src" / "exec-shell.js")
    shutil.copy(REPORT_JS, out / "src" / "report.js")
    shutil.copy(TEMPLATES / "report.css", out / "src" / "report.css")
    (out / "index.html").write_text(render_html(a, pages), encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps(
            {
                "source": ds.source,
                "attribution": a.attribution,
                "removal_effects": a.removal,
                "ate": a.ate.__dict__,
                "segments": {s: e.__dict__ for s, e in a.segments.items()},
                "economics": [r.__dict__ for r in [*a.economics, a.blended]],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return out / "index.html"
