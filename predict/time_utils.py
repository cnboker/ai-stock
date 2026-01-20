# utils/time_utils.py
import numpy as np
import pandas_market_calendars as mcal
import pandas as pd
from datetime import datetime, time
from config.settings import PREDICTION_LENGTH

def is_market_break():
    now = datetime.now().time()
    return now < time(9,30) or time(11, 30) <= now < time(13, 0) or now >= time(15,00)

def get_next_trading_times(start_dt, n, freq):
    freq_min = 5 if freq == "5" else 15
    cn = mcal.get_calendar("XSHG")

    start_dt = pd.to_datetime(start_dt).tz_localize(None)
    schedule = cn.schedule(
        start_date=start_dt.date(),
        end_date=(start_dt + pd.Timedelta(days=30)).date(),
    )
    trading_days = set(schedule.index.date)

    times = []
    current = start_dt

    while len(times) < n:
        current += pd.Timedelta(minutes=freq_min)

        if time(11, 30) <= current.time() < time(13, 0):
            current = current.replace(hour=13, minute=0)

        if current.time() >= time(15, 0):
            current += pd.Timedelta(days=1)
            current = current.replace(hour=9, minute=30)

        if current.date() not in trading_days:
            current += pd.Timedelta(days=1)
            current = current.replace(hour=9, minute=30)

        times.append(current)

    return pd.DatetimeIndex(times)




def build_future_index(df, period: str):
    """
    构建未来交易时间轴（解决午休 / 周末）
    """
    last_dt = df.index[-1].tz_localize(None)
    freq = "5min" if period == "5" else "15min"

    # future_index = pd.date_range(
    #             start=last_dt + pd.Timedelta(minutes=int(period)),
    #             periods=PREDICTION_LENGTH,
    #             freq=freq,
    #             tz=None,
    #         )
    # return future_index
    return get_next_trading_times(
        start_dt=last_dt,
        n=PREDICTION_LENGTH,
        freq=freq,
    )



def calc_atr(df: pd.DataFrame, period: int = 3) -> float:
    """
    鲁棒 ATR：
    - 不返回 0 / NaN
    - 支持涨停、一字板、小样本
    """
    if df is None or len(df) == 0:
        return 1e-6

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # rolling ATR，允许小样本
    atr = tr.rolling(period, min_periods=1).mean()

    val = float(atr.iloc[-1])

    # 🚑 兜底：防止 0 / NaN
    if not np.isfinite(val) or val <= 0:
        # 用价格比例兜底（千分之一）
        price = float(close.iloc[-1])
        return max(price * 0.001, 1e-6)

    return val
