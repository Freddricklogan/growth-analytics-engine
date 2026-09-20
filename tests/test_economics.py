import math

import numpy as np
import pytest

from growth_engine.cohorts import CohortTable
from growth_engine.economics import blended, channel_economics, payback_month


def table() -> CohortTable:
    cum = np.array([[10.0, 25.0, 40.0]])
    return CohortTable([0], [4], np.ones((1, 3)), cum)


def test_channel_economics_by_hand() -> None:
    rows = channel_economics({"a": 100.0, "b": 0.0}, {"a": 5.0, "b": 2.0, "c": 0.0}, table(), 2)
    a, b, c = rows
    assert (a.cac, a.ltv, a.ltv_to_cac, a.payback_month) == (20.0, 40.0, 2.0, 1)
    assert b.cac == 0.0 and math.isinf(b.ltv_to_cac) and b.payback_month == 0
    assert math.isinf(c.cac) and c.ltv_to_cac == 0.0 and c.payback_month is None
    bl = blended(rows, table())
    assert bl.cac == pytest.approx(100.0 / 7.0)
    assert bl.ltv_to_cac == pytest.approx(40.0 * 7.0 / 100.0)
    assert bl.payback_month == 1


def test_payback_never_reached_and_unobserved_horizon() -> None:
    assert payback_month(np.array([1.0, 2.0, np.nan]), 5.0) is None
    with pytest.raises(ValueError, match="not observed"):
        channel_economics({"a": 1.0}, {"a": 1.0}, table(), 5)


def test_blended_with_no_rows() -> None:
    bl = blended([], table())
    assert math.isinf(bl.cac) and bl.ltv_to_cac == 0.0
