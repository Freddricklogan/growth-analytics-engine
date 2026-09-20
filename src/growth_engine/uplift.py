"""Uplift from a randomised treatment: average effect with a confidence interval, per-segment
effects, and a Qini curve for targeting by estimated uplift."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .events import Journey


@dataclass(frozen=True)
class Effect:
    n_treated: int
    n_control: int
    rate_treated: float
    rate_control: float
    uplift: float  # percentage points as a fraction
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def two_proportion(journeys: Sequence[Journey], z: float = 1.96) -> Effect:
    """Difference in conversion rate, treated minus control, with a Wald interval."""
    t = [j for j in journeys if j.treated]
    c = [j for j in journeys if not j.treated]
    if not t or not c:
        raise ValueError("both treated and control groups are required")
    pt = sum(j.converted for j in t) / len(t)
    pc = sum(j.converted for j in c) / len(c)
    se = math.sqrt(pt * (1 - pt) / len(t) + pc * (1 - pc) / len(c))
    d = pt - pc
    return Effect(len(t), len(c), pt, pc, d, d - z * se, d + z * se)


def by_segment(journeys: Sequence[Journey]) -> dict[str, Effect]:
    segments = sorted({j.segment for j in journeys})
    return {s: two_proportion([j for j in journeys if j.segment == s]) for s in segments}


def qini(journeys: Sequence[Journey]) -> list[tuple[float, float]]:
    """Cumulative incremental conversions when segments are targeted in order of estimated
    uplift, as (share of population targeted, incremental conversions). The last point is the
    whole population; a random policy is the straight line to it."""
    effects = by_segment(journeys)
    order = sorted(effects, key=lambda s: effects[s].uplift, reverse=True)
    n = len(journeys)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    covered, gain = 0, 0.0
    for s in order:
        e = effects[s]
        size = e.n_treated + e.n_control
        covered += size
        # Incremental conversions if everyone in the segment were treated.
        gain += e.uplift * size
        points.append((covered / n, gain))
    return points
