import numpy as np
import pytest

from growth_engine.attribution import (
    CONVERSION,
    NULL,
    START,
    compare,
    conversion_probability,
    first_touch,
    last_touch,
    linear,
    markov,
    removal_effects,
    spearman,
    transition_matrix,
)
from growth_engine.events import CHANNEL_EFFECT, Journey, synthetic_dataset


def j(path: str, converted: bool, uid: int = 0) -> Journey:
    return Journey(uid, tuple(path.split()), converted, 0, "new", False)


JOURNEYS = [j("a b", True), j("a", False), j("b", True), j("a b", False)]


def test_heuristic_models_by_hand() -> None:
    assert last_touch(JOURNEYS) == {"a": 0.0, "b": 2.0}
    assert first_touch(JOURNEYS) == {"a": 1.0, "b": 1.0}
    assert linear(JOURNEYS) == {"a": 0.5, "b": 1.5}


def test_transition_matrix_is_row_stochastic_with_hand_counts() -> None:
    states, m = transition_matrix(JOURNEYS)
    assert states == [START, "a", "b", CONVERSION, NULL]
    assert np.allclose(m.sum(axis=1), 1.0)
    # From (start): 3 journeys begin with a, 1 with b.
    assert m[0, 1] == pytest.approx(0.75) and m[0, 2] == pytest.approx(0.25)
    # From a: two go to b, one to null.
    assert m[1, 2] == pytest.approx(2 / 3) and m[1, 4] == pytest.approx(1 / 3)


def test_conversion_probability_matches_hand_solution() -> None:
    states, m = transition_matrix(JOURNEYS)
    # P(conv) = 0.75 * (2/3 * P_b) + 0.25 * P_b, with P_b = 2/3 (b -> conv twice, null once).
    expected = 0.75 * (2 / 3) * (2 / 3) + 0.25 * (2 / 3)
    assert conversion_probability(states, m) == pytest.approx(expected)
    # Removing b makes conversion impossible: every conversion passed through b.
    assert conversion_probability(states, m, removed="b") == pytest.approx(0.0)


def test_removal_effects_and_markov_credit() -> None:
    eff = removal_effects(JOURNEYS)
    assert eff["b"] == pytest.approx(1.0)
    assert 0.0 < eff["a"] < 1.0
    credit = markov(JOURNEYS)
    assert sum(credit.values()) == pytest.approx(2.0)
    assert credit["b"] > credit["a"]


def test_every_model_conserves_conversions_on_synthetic_data() -> None:
    ds = synthetic_dataset(n_users=800, months=6, seed=7)
    total = sum(1 for x in ds.journeys if x.converted)
    for name, credit in compare(ds.journeys).items():
        assert sum(credit.values()) == pytest.approx(total), name
        assert all(v >= 0 for v in credit.values())


def test_no_conversions_gives_zero_credit() -> None:
    js = [j("a b", False), j("b", False)]
    assert markov(js) == {"a": 0.0, "b": 0.0}
    assert removal_effects(js) == {"a": 0.0, "b": 0.0}


def test_spearman_closed_forms() -> None:
    a = {"x": 1.0, "y": 2.0, "z": 3.0}
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, {"x": 3.0, "y": 2.0, "z": 1.0}) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="two shared"):
        spearman(a, {"x": 1.0})


def test_attribution_does_not_recover_causal_truth() -> None:
    """The documented finding: on a generator with additive channel effects and paid-to-organic
    hand-offs, no attribution model ranks channels like the true effects do."""
    ds = synthetic_dataset(n_users=6000, months=18, seed=42)
    rho = {m: spearman(c, CHANNEL_EFFECT) for m, c in compare(ds.journeys).items()}
    assert all(r < 0.8 for r in rho.values())
    assert rho["markov"] < rho["last_touch"]
