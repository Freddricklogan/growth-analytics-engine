"""Synthetic, privacy-safe event streams and a validating CSV loader.

The generator has a known causal structure (per-channel conversion effects, a treatment that
helps one segment only, a churn hazard on purchases) so that the analytics modules can be tested
against ground truth rather than against each other.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field

import numpy as np

CHANNELS: tuple[str, ...] = (
    "organic_search",
    "paid_search",
    "paid_social",
    "email",
    "referral",
    "webinar",
)
SEGMENTS: tuple[str, ...] = ("new", "lapsed", "alumni")

# Ground truth used by the generator. Logit contribution of each channel to conversion, the
# probability a journey starts on the channel, and the spend the channel cost over the period.
CHANNEL_EFFECT: dict[str, float] = {
    "organic_search": 0.35,
    "paid_search": 0.45,
    "paid_social": 0.10,
    "email": 0.30,
    "referral": 0.60,
    "webinar": 0.75,
}
CHANNEL_START: dict[str, float] = {
    "organic_search": 0.30,
    "paid_search": 0.22,
    "paid_social": 0.24,
    "email": 0.08,
    "referral": 0.10,
    "webinar": 0.06,
}
CHANNEL_SPEND: dict[str, float] = {
    "organic_search": 18_000.0,
    "paid_search": 64_000.0,
    "paid_social": 58_000.0,
    "email": 9_000.0,
    "referral": 12_000.0,
    "webinar": 21_000.0,
}
SEGMENT_EFFECT: dict[str, float] = {"new": 0.0, "lapsed": -0.6, "alumni": 0.4}
TREATMENT_EFFECT: dict[str, float] = {"new": 0.0, "lapsed": 0.9, "alumni": 0.0}
BASE_LOGIT = -2.6
MONTHLY_CHURN = 0.12
MEAN_REVENUE = 60.0


@dataclass(frozen=True)
class Journey:
    """One prospect's sequence of marketing touches and its outcome."""

    user_id: int
    channels: tuple[str, ...]
    converted: bool
    cohort_month: int
    segment: str
    treated: bool


@dataclass(frozen=True)
class Purchase:
    user_id: int
    month: int
    revenue: float


@dataclass(frozen=True)
class Dataset:
    journeys: list[Journey]
    purchases: list[Purchase]
    spend: dict[str, float]
    months: int
    source: str
    warnings: list[str] = field(default_factory=list)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def synthetic_dataset(n_users: int = 6000, months: int = 18, seed: int = 42) -> Dataset:
    """Generate journeys, conversions and purchases with the ground truth above."""
    if n_users < 1 or months < 2:
        raise ValueError("n_users must be >= 1 and months >= 2")
    rng = np.random.default_rng(seed)
    names = list(CHANNEL_START)
    start_p = np.array([CHANNEL_START[c] for c in names])
    start_p /= start_p.sum()
    # Next-touch preference: paid channels tend to hand off to organic, email or webinar.
    handoff = np.full((len(names), len(names)), 1.0)
    idx = {c: i for i, c in enumerate(names)}
    for src in ("paid_search", "paid_social"):
        handoff[idx[src], idx["organic_search"]] = 3.0
        handoff[idx[src], idx["email"]] = 2.0
    handoff[idx["email"], idx["webinar"]] = 3.0
    handoff[idx["webinar"], idx["referral"]] = 2.0
    handoff /= handoff.sum(axis=1, keepdims=True)

    journeys: list[Journey] = []
    purchases: list[Purchase] = []
    for uid in range(n_users):
        length = int(rng.integers(1, 6))
        first = int(rng.choice(len(names), p=start_p))
        path = [first]
        for _ in range(length - 1):
            path.append(int(rng.choice(len(names), p=handoff[path[-1]])))
        channels = tuple(names[i] for i in path)
        segment = str(rng.choice(SEGMENTS, p=[0.55, 0.25, 0.20]))
        treated = bool(rng.random() < 0.5)
        # Diminishing returns on repeated touches: count each channel once.
        logit = BASE_LOGIT + SEGMENT_EFFECT[segment]
        logit += sum(CHANNEL_EFFECT[c] for c in set(channels))
        if treated:
            logit += TREATMENT_EFFECT[segment]
        converted = bool(rng.random() < _sigmoid(logit))
        cohort = int(rng.integers(0, months - 1))
        journeys.append(Journey(uid, channels, converted, cohort, segment, treated))
        if converted:
            month = cohort
            while month < months:
                purchases.append(Purchase(uid, month, float(rng.gamma(4.0, MEAN_REVENUE / 4.0))))
                if rng.random() < MONTHLY_CHURN:
                    break
                month += 1
    return Dataset(
        journeys=journeys,
        purchases=purchases,
        spend=dict(CHANNEL_SPEND),
        months=months,
        source=f"synthetic event stream, seed {seed}",
    )


def load_journeys_csv(text: str, source: str = "upload") -> Dataset:
    """Parse a `user_id,path,converted` CSV where path is `a > b > c`. Rows that do not parse
    are dropped and reported in `warnings`; nothing is silently coerced."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    required = {"user_id", "path", "converted"}
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise ValueError(f"CSV must have columns {sorted(required)}")
    journeys: list[Journey] = []
    warnings: list[str] = []
    for n, row in enumerate(reader, start=2):
        try:
            uid = int(row["user_id"])
        except (TypeError, ValueError):
            warnings.append(f"row {n}: user_id is not an integer")
            continue
        channels = tuple(p.strip() for p in (row["path"] or "").split(">") if p.strip())
        conv_raw = (row["converted"] or "").strip().lower()
        if conv_raw not in {"0", "1", "true", "false"}:
            warnings.append(f"row {n}: converted must be 0/1/true/false")
            continue
        if not channels:
            warnings.append(f"row {n}: empty path")
            continue
        journeys.append(Journey(uid, channels, conv_raw in {"1", "true"}, 0, "new", False))
    if not journeys:
        raise ValueError("no valid rows")
    return Dataset(journeys, [], {}, 1, source, warnings)
