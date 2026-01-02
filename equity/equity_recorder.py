import numpy as np
import pandas as pd
import os
from datetime import datetime


class EquityRecorder:
    def __init__(self, path: str = "", max_len: int = 2000):
        self.path = path
        self.max_len = max_len
        self.df = pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "equity": pd.Series(dtype="float64"),
            }
        )
        self._load_disk()

    def _load_disk(self):
        if os.path.exists(self.path):
            try:
                self.df = pd.read_csv(self.path, parse_dates=["timestamp"])
                self.df = self.df.tail(self.max_len)  # 只保留最新 max_len 条
            except Exception as e:
                print(f"Warning: failed to load equity file: {e}")


    def add(self, equity: float, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()
        self.df = pd.concat(
            [self.df, pd.DataFrame([{"timestamp": timestamp, "equity": equity}])],
            ignore_index=True,
        )
        # 保留最近 max_len 条
        if len(self.df) > self.max_len:
            self.df = self.df.tail(self.max_len)
        self._save_disk()

    def _save_disk(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.df.to_csv(self.path, index=False)

        except Exception as e:
            print(f"Warning: failed to save equity file: {e}")

    def to_series(self) -> pd.Series:
        if self.df.empty:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]), name="equity")

        df = (
            self.df
            .dropna(subset=["timestamp", "equity"])
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
        )

        s = pd.Series(
            df["equity"].astype(float).values,
            index=pd.DatetimeIndex(df["timestamp"]),
            name="equity",
        )
       
        return s

    def latest(self) -> float:
        """返回最新 equity"""
        if len(self.df) == 0:
            return 0.0
        return float(self.df["equity"].iloc[-1])

