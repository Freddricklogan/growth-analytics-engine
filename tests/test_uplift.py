import math

import pytest

from growth_engine.events import Journey, synthetic_dataset
from growth_engine.uplift import by_segment, qini, two_proportion


def j(uid: int, treated: bool, converted: bool, segment: str = "new") -> Journey:
    return Journey(uid, ("a",), converted, 0, segment, treated)


def test_two_proportion_by_hand() -> None:
    js = [j(i, True, i < 3) for i in range(10)] + [j(10 + i, False, i < 1) for i in range(10)]
    e = two_proportion(js)
    assert (e.n_treated, e.n_control) == (10, 10)
    assert e.rate_treated == pytest.approx(0.3) and e.rate_control == pytest.approx(0.1)
    se = math.sqrt(0.3 * 0.7 / 10 + 0.1 * 0.9 / 10)
    assert e.uplift == pytest.approx(0.2)
    assert e.ci_low == pytest.approx(0.2 - 1.96 * se)
    assert e.ci_high == pytest.approx(0.2 + 1.96 * se)
    assert not e.significant


def test_requires_both_arms() -> None:
    with pytest.raises(ValueError, match="both"):
        two_proportion([j(1, True, True)])


def test_segments_and_qini_by_hand() -> None:
    js = (
        [j(i, True, True, "hot") for i in range(4)]
        + [j(10 + i, False, False, "hot") for i in range(4)]
        + [j(20 + i, True, i % 2 == 0, "cold") for i in range(4)]
        + [j(30 + i, False, i % 2 == 0, "cold") for i in range(4)]
    )
    seg = by_segment(js)
    assert seg["hot"].uplift == pytest.approx(1.0) and seg["hot"].significant
    assert seg["cold"].uplift == pytest.approx(0.0)
    q = qini(js)
    assert q == [(0.0, 0.0), (0.5, pytest.approx(8.0)), (1.0, pytest.approx(8.0))]


def test_synthetic_treatment_helps_lapsed_only() -> None:
    ds = synthetic_dataset(n_users=6000, months=18, seed=42)
    seg = by_segment(ds.journeys)
    assert seg["lapsed"].significant and seg["lapsed"].uplift > 0
    assert not seg["new"].significant
    assert two_proportion(ds.journeys).uplift > 0
