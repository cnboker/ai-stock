from datetime import datetime
from equity.equity_features import equity_features, get_metrics
from equity.equity_recorder import EquityRecorder,eq_recorder


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
    
    eq_feat = get_metrics(eq)
    print('get_metrics', eq_feat)

def test_eq_recorder():
    eq = eq_recorder.to_series()
    print('eq---',eq)