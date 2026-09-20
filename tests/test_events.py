import pytest

from growth_engine.events import CHANNELS, Journey, load_journeys_csv, synthetic_dataset


def test_synthetic_is_deterministic_and_well_formed() -> None:
    a = synthetic_dataset(n_users=300, months=6, seed=1)
    b = synthetic_dataset(n_users=300, months=6, seed=1)
    assert a.journeys == b.journeys
    assert a.purchases == b.purchases
    assert len(a.journeys) == 300
    converted = {j.user_id for j in a.journeys if j.converted}
    assert {p.user_id for p in a.purchases} == converted
    for j in a.journeys:
        assert 1 <= len(j.channels) <= 5
        assert set(j.channels) <= set(CHANNELS)
        assert 0 <= j.cohort_month < 5
    for p in a.purchases:
        assert 0 <= p.month < 6
        assert p.revenue > 0


def test_purchases_start_in_cohort_month_and_are_contiguous() -> None:
    ds = synthetic_dataset(n_users=200, months=8, seed=3)
    cohort = {j.user_id: j.cohort_month for j in ds.journeys}
    months: dict[int, list[int]] = {}
    for p in ds.purchases:
        months.setdefault(p.user_id, []).append(p.month)
    for uid, ms in months.items():
        assert ms[0] == cohort[uid]
        assert ms == list(range(ms[0], ms[0] + len(ms)))


def test_seed_changes_output() -> None:
    assert synthetic_dataset(50, 4, 1).journeys != synthetic_dataset(50, 4, 2).journeys


@pytest.mark.parametrize(("users", "months"), [(0, 6), (10, 1)])
def test_synthetic_rejects_bad_sizes(users: int, months: int) -> None:
    with pytest.raises(ValueError, match="n_users"):
        synthetic_dataset(users, months)


def test_csv_loader_parses_and_reports_bad_rows() -> None:
    text = (
        "﻿user_id,path,converted\n"
        "1,paid_search > organic_search,1\n"
        "2,email,false\n"
        "x,email,1\n"
        "3,,1\n"
        "4,email,maybe\n"
    )
    ds = load_journeys_csv(text)
    assert [j.user_id for j in ds.journeys] == [1, 2]
    assert ds.journeys[0] == Journey(1, ("paid_search", "organic_search"), True, 0, "new", False)
    assert ds.journeys[1].converted is False
    assert len(ds.warnings) == 3
    assert "row 4" in ds.warnings[0] and "row 5" in ds.warnings[1] and "row 6" in ds.warnings[2]


def test_csv_loader_rejects_missing_columns_and_empty() -> None:
    with pytest.raises(ValueError, match="columns"):
        load_journeys_csv("id,route\n1,a\n")
    with pytest.raises(ValueError, match="no valid rows"):
        load_journeys_csv("user_id,path,converted\nx,a,1\n")
