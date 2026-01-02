# utils/time_utils.py
import pandas_market_calendars as mcal
import pandas as pd
from datetime import datetime, time
from config.settings import PREDICTION_LENGTH

def is_market_break():
    now = datetime.now().time()
    return time(11, 30) <= now < time(13, 0) or now >= time(15,00)

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
    df 必须包含: high, low, close
    返回最新一根 ATR
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean()

    return float(atr.iloc[-1])
