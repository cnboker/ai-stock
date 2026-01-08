from datetime import datetime
from equity.equity_features import equity_features
from equity.equity_recorder import EquityRecorder


def test_to_equity_features_ok():
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
    eq = rec.to_series()
    print('eq',eq)
    eq_feat = equity_features(eq)
    print(eq_feat)

def test_to_equity_features_get_metrics_ok():
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
    eq = rec.to_series()
    
    eq_feat = equity_features(eq)
    print('equity_features', eq_feat)

def test_eq_recorder():
    rec = EquityRecorder()
    eq = rec.to_series()
    print('eq---',eq)