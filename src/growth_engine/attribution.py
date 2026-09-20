"""Multi-touch attribution: heuristic rules and Markov-chain removal effects.

Every method returns conversions credited per channel; the credits sum to the number of
converting journeys (up to floating point), so the models can be compared row by row.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from .events import Journey

START, CONVERSION, NULL = "(start)", "(conversion)", "(null)"


def channel_names(journeys: Sequence[Journey]) -> list[str]:
    seen: dict[str, None] = {}
    for j in journeys:
        for c in j.channels:
            seen.setdefault(c, None)
    return list(seen)


def last_touch(journeys: Sequence[Journey]) -> dict[str, float]:
    credit: defaultdict[str, float] = defaultdict(float)
    for j in journeys:
        if j.converted:
            credit[j.channels[-1]] += 1.0
    return _complete(credit, journeys)


def first_touch(journeys: Sequence[Journey]) -> dict[str, float]:
    credit: defaultdict[str, float] = defaultdict(float)
    for j in journeys:
        if j.converted:
            credit[j.channels[0]] += 1.0
    return _complete(credit, journeys)


def linear(journeys: Sequence[Journey]) -> dict[str, float]:
    credit: defaultdict[str, float] = defaultdict(float)
    for j in journeys:
        if j.converted:
            share = 1.0 / len(j.channels)
            for c in j.channels:
                credit[c] += share
    return _complete(credit, journeys)


def _complete(credit: defaultdict[str, float], journeys: Sequence[Journey]) -> dict[str, float]:
    return {c: float(credit.get(c, 0.0)) for c in channel_names(journeys)}


def transition_matrix(journeys: Sequence[Journey]) -> tuple[list[str], np.ndarray]:
    """Row-stochastic matrix over states [start, channels..., conversion, null], estimated by
    counting transitions in the journeys. Absorbing states loop to themselves."""
    names = channel_names(journeys)
    states = [START, *names, CONVERSION, NULL]
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)))
    for j in journeys:
        prev = START
        for c in j.channels:
            counts[idx[prev], idx[c]] += 1
            prev = c
        counts[idx[prev], idx[CONVERSION if j.converted else NULL]] += 1
    counts[idx[CONVERSION], idx[CONVERSION]] = 1
    counts[idx[NULL], idx[NULL]] = 1
    rows = counts.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return states, counts / rows


def conversion_probability(
    states: list[str], matrix: np.ndarray, removed: str | None = None
) -> float:
    """Probability of absorption into (conversion) from (start), optionally with one channel
    removed — its inbound transitions are redirected to (null)."""
    m = matrix.copy()
    if removed is not None:
        r = states.index(removed)
        m[:, states.index(NULL)] += m[:, r]
        m[:, r] = 0.0
        m[r, :] = 0.0
        m[r, states.index(NULL)] = 1.0
    transient = [i for i, s in enumerate(states) if s not in (CONVERSION, NULL)]
    q = m[np.ix_(transient, transient)]
    r_col = m[transient, states.index(CONVERSION)]
    absorb = np.linalg.solve(np.eye(len(transient)) - q, r_col)
    return float(absorb[transient.index(states.index(START))])


def removal_effects(journeys: Sequence[Journey]) -> dict[str, float]:
    """1 - P(conversion | channel removed) / P(conversion). In [0, 1] for each channel."""
    states, matrix = transition_matrix(journeys)
    base = conversion_probability(states, matrix)
    if base <= 0.0:
        return dict.fromkeys(channel_names(journeys), 0.0)
    out: dict[str, float] = {}
    for c in channel_names(journeys):
        p = conversion_probability(states, matrix, removed=c)
        out[c] = float(min(1.0, max(0.0, 1.0 - p / base)))
    return out


def markov(journeys: Sequence[Journey]) -> dict[str, float]:
    """Conversions credited in proportion to removal effect."""
    effects = removal_effects(journeys)
    total = sum(1 for j in journeys if j.converted)
    denom = sum(effects.values())
    if denom == 0.0:
        return dict.fromkeys(effects, 0.0)
    return {c: total * e / denom for c, e in effects.items()}


MODELS = {"last_touch": last_touch, "first_touch": first_touch, "linear": linear, "markov": markov}


def compare(journeys: Sequence[Journey]) -> dict[str, dict[str, float]]:
    """All models, keyed by model name then channel."""
    return {name: fn(journeys) for name, fn in MODELS.items()}


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    """Rank correlation between two channel scorings over their shared keys."""
    keys = [k for k in a if k in b]
    if len(keys) < 2:
        raise ValueError("need at least two shared channels")
    ra = np.argsort(np.argsort([a[k] for k in keys])).astype(float)
    rb = np.argsort(np.argsort([b[k] for k in keys])).astype(float)
    n = len(keys)
    return float(1.0 - 6.0 * np.sum((ra - rb) ** 2) / (n * (n**2 - 1)))
