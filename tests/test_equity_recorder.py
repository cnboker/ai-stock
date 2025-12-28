import pandas as pd
from datetime import datetime

from equity.equity_recorder import EquityRecorder


def test_to_series_basic():
    rec = EquityRecorder()

    rec.add(100_000, datetime(2024, 1, 1, 9, 30))
    rec.add(
        100_500,
        datetime(2024, 1, 1, 9, 31),
    )
    rec.add(
        100_200,
        datetime(2024, 1, 1, 9, 32),
    )

    s = rec.to_series()

    # --- 类型 ---
    assert isinstance(s, pd.Series)

    # --- index ---
    assert type(s.index) == pd.DatetimeIndex
    assert len(s) == 3

    # --- 值 ---
    assert s.iloc[0] == 100_000
    assert s.iloc[-1] == 100_200

    # --- index 单调递增（对 chronos / reindex 很关键） ---
    assert s.index.is_monotonic_increasing


def test_to_series_empty():
    rec = EquityRecorder()

    s = rec.to_series()

    assert isinstance(s, pd.Series)
    assert len(s) == 0
    assert isinstance(s.index, pd.DatetimeIndex)


def test_to_series_duplicate_timestamp_keep_last():
    rec = EquityRecorder()

    ts = datetime(2024, 1, 1, 9, 30)

    rec.add(
        100_000,
        ts,
    )
    rec.add(
        100_500,
        ts,
    )

    s = rec.to_series()

    assert len(s) == 1
    assert s.iloc[0] == 100_500


def test_to_series_no_fill_side_effect():
    rec = EquityRecorder()

    rec.add(
        100_000,
        datetime(2024, 1, 1, 9, 30),
    )
    rec.add(
        100_200,
        datetime(2024, 1, 1, 9, 32),
    )

    s = rec.to_series()

    # 中间缺失不该被自动补
    assert s.isna().sum() == 0
    assert len(s) == 2
