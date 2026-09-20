"""Cohort retention and lifetime-value curves by acquisition month."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .events import Journey, Purchase

FloatArray = np.ndarray


@dataclass(frozen=True)
class CohortTable:
    cohorts: list[int]
    sizes: list[int]
    retention: FloatArray  # [cohort, month_since] fraction active; NaN where unobserved
    cumulative_revenue: FloatArray  # [cohort, month_since] revenue per acquired user, cumulative


def cohort_table(
    journeys: Sequence[Journey], purchases: Sequence[Purchase], months: int
) -> CohortTable:
    """Rows are acquisition months; columns are months since acquisition."""
    converted = {j.user_id: j.cohort_month for j in journeys if j.converted}
    cohorts = sorted(set(converted.values()))
    if not cohorts:
        raise ValueError("no converted users")
    row = {c: i for i, c in enumerate(cohorts)}
    sizes = [sum(1 for m in converted.values() if m == c) for c in cohorts]
    active = np.zeros((len(cohorts), months))
    revenue = np.zeros((len(cohorts), months))
    seen: set[tuple[int, int]] = set()
    for p in purchases:
        c = converted.get(p.user_id)
        if c is None or p.month < c:
            continue
        k = p.month - c
        revenue[row[c], k] += p.revenue
        if (p.user_id, k) not in seen:
            seen.add((p.user_id, k))
            active[row[c], k] += 1
    size_col = np.array(sizes, dtype=float)[:, None]
    retention = active / size_col
    cumulative = np.cumsum(revenue, axis=1) / size_col
    for i, c in enumerate(cohorts):
        observed = months - c
        retention[i, observed:] = np.nan
        cumulative[i, observed:] = np.nan
    return CohortTable(cohorts, sizes, retention, cumulative)


def average_curve(table: CohortTable, matrix: FloatArray) -> FloatArray:
    """Size-weighted mean down each column over the cohorts that observed that column."""
    w = np.array(table.sizes, dtype=float)[:, None]
    mask = ~np.isnan(matrix)
    num = np.nansum(matrix * w, axis=0)
    den = (mask * w).sum(axis=0)
    out = np.full(matrix.shape[1], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def ltv_at(table: CohortTable, horizon: int) -> float:
    """Average cumulative revenue per converted user at `horizon` months since acquisition,
    using only cohorts old enough to have observed it."""
    curve = average_curve(table, table.cumulative_revenue)
    if horizon < 0 or horizon >= len(curve) or np.isnan(curve[horizon]):
        raise ValueError(f"horizon {horizon} not observed by any cohort")
    return float(curve[horizon])
