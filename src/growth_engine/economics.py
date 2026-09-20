"""Unit economics per channel: CAC from attributed conversions, LTV from the cohort curve,
the LTV:CAC ratio and the month the cohort pays back its acquisition cost."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .cohorts import CohortTable, average_curve


@dataclass(frozen=True)
class ChannelEconomics:
    channel: str
    spend: float
    conversions: float
    cac: float  # inf when no conversions are attributed
    ltv: float
    ltv_to_cac: float
    payback_month: int | None  # None when the observed curve never reaches CAC


def payback_month(curve: np.ndarray, cac: float) -> int | None:
    for k, v in enumerate(curve):
        if not np.isnan(v) and v >= cac:
            return k
    return None


def channel_economics(
    spend: Mapping[str, float],
    attributed: Mapping[str, float],
    table: CohortTable,
    horizon: int,
) -> list[ChannelEconomics]:
    curve = average_curve(table, table.cumulative_revenue)
    if horizon < 0 or horizon >= len(curve) or np.isnan(curve[horizon]):
        raise ValueError(f"horizon {horizon} not observed")
    ltv = float(curve[horizon])
    rows: list[ChannelEconomics] = []
    for channel, conv in attributed.items():
        s = float(spend.get(channel, 0.0))
        cac = s / conv if conv > 0 else float("inf")
        rows.append(
            ChannelEconomics(
                channel, s, conv, cac, ltv, _ratio(ltv, cac), payback_month(curve, cac)
            )
        )
    return rows


def _ratio(ltv: float, cac: float) -> float:
    """LTV:CAC; infinite when acquisition was free, zero when nothing was acquired."""
    if not np.isfinite(cac):
        return 0.0
    if cac == 0.0:
        return float("inf")
    return ltv / cac


def blended(rows: list[ChannelEconomics], table: CohortTable) -> ChannelEconomics:
    """Whole-programme figures: total spend over total attributed conversions."""
    curve = average_curve(table, table.cumulative_revenue)
    spend = sum(r.spend for r in rows)
    conv = sum(r.conversions for r in rows)
    ltv = rows[0].ltv if rows else 0.0
    cac = spend / conv if conv > 0 else float("inf")
    return ChannelEconomics(
        "blended", spend, conv, cac, ltv, _ratio(ltv, cac), payback_month(curve, cac)
    )
