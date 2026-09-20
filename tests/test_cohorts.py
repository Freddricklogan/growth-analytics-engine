import numpy as np
import pytest

from growth_engine.cohorts import average_curve, cohort_table, ltv_at
from growth_engine.events import Journey, Purchase, synthetic_dataset


def conv(uid: int, month: int) -> Journey:
    return Journey(uid, ("a",), True, month, "new", False)


def test_cohort_table_by_hand() -> None:
    journeys = [conv(1, 0), conv(2, 0), conv(3, 1), Journey(4, ("a",), False, 0, "new", False)]
    purchases = [
        Purchase(1, 0, 10.0),
        Purchase(1, 1, 10.0),
        Purchase(1, 1, 5.0),  # second purchase in the same month counts once for retention
        Purchase(2, 0, 20.0),
        Purchase(3, 1, 8.0),
        Purchase(3, 2, 8.0),
        Purchase(9, 0, 99.0),  # unknown user is ignored
    ]
    t = cohort_table(journeys, purchases, months=3)
    assert t.cohorts == [0, 1] and t.sizes == [2, 1]
    assert t.retention[0].tolist() == [1.0, 0.5, 0.0]
    assert t.retention[1, :2].tolist() == [1.0, 1.0] and np.isnan(t.retention[1, 2])
    assert t.cumulative_revenue[0].tolist() == [15.0, 22.5, 22.5]
    assert t.cumulative_revenue[1, :2].tolist() == [8.0, 16.0]
    # Size-weighted average at month 1: (0.5*2 + 1.0*1) / 3.
    avg = average_curve(t, t.retention)
    assert avg[1] == pytest.approx(0.5 * 2 / 3 + 1.0 / 3)
    assert avg[2] == pytest.approx(0.0)  # only cohort 0 observed month 2
    assert ltv_at(t, 1) == pytest.approx((22.5 * 2 + 16.0) / 3)


def test_ltv_at_rejects_unobserved_horizon() -> None:
    t = cohort_table([conv(1, 1)], [Purchase(1, 1, 1.0)], months=3)
    with pytest.raises(ValueError, match="not observed"):
        ltv_at(t, 2)


def test_no_converted_users_raises() -> None:
    with pytest.raises(ValueError, match="no converted"):
        cohort_table([Journey(1, ("a",), False, 0, "new", False)], [], 3)


def test_synthetic_retention_is_monotone_on_average() -> None:
    ds = synthetic_dataset(n_users=3000, months=12, seed=11)
    t = cohort_table(ds.journeys, ds.purchases, ds.months)
    avg = average_curve(t, t.retention)
    assert avg[0] == pytest.approx(1.0)
    assert all(avg[k + 1] <= avg[k] + 1e-9 for k in range(6))
    assert np.isnan(t.retention[-1, -1])
