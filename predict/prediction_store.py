# predict/history.py
import pickle
import os
import pandas as pd
from config.settings import HISTORY_FILE

PREDICTION_HISTORY = {}

def load_history():
    global PREDICTION_HISTORY
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as f:
            PREDICTION_HISTORY = pickle.load(f)
    return PREDICTION_HISTORY


def append_prediction(ticker: str, new_pred: pd.DataFrame):
    new_pred = new_pred.copy()
    new_pred["ticker"] = ticker

    if ticker not in PREDICTION_HISTORY:
        PREDICTION_HISTORY[ticker] = new_pred
    else:
        old = PREDICTION_HISTORY[ticker]
        start = new_pred["datetime"].iloc[0]
        old = old[old["datetime"] < start]
        PREDICTION_HISTORY[ticker] = (
            pd.concat([old, new_pred]).sort_values("datetime").reset_index(drop=True)
        )

    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(PREDICTION_HISTORY, f)

    return PREDICTION_HISTORY[ticker]

def update_prediction_history(ticker, future_index, low, median, high):
    """
    合并 & 持久化预测历史
    """
    new_pred = pd.DataFrame({
        "datetime": future_index,
        "low": low,
        "median": median,
        "high": high,
    })
    return append_prediction(ticker, new_pred)
