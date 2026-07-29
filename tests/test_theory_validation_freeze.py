import pytest

from src.distributed.theory_validation_freeze import select_monotone_schedule


def _row(batch: int, ratio: float) -> dict:
    return {"data_B_total": batch, "work_ratio_nog_over_me": ratio}


def test_select_monotone_schedule_matches_work_without_decreasing_batch() -> None:
    schedule = select_monotone_schedule(
        [
            [_row(8, 1.0), _row(16, 2.0)],
            [_row(8, 0.5), _row(16, 1.1)],
            [_row(16, 0.8), _row(24, 1.4)],
        ],
        switch_penalty=0.0,
    )
    assert [row["data_B_total"] for row in schedule] == [8, 16, 16]


def test_select_monotone_schedule_rejects_missing_epsilon() -> None:
    with pytest.raises(ValueError, match="Every primary epsilon"):
        select_monotone_schedule([[_row(8, 1.0)], []])
