# snapshot.py
from dataclasses import dataclass
import pandas as pd
import os
import matplotlib.pyplot as plt
from strategy.decision_context import DecisionContext
from strategy.gate import gater
import matplotlib.dates as mdates
@dataclass
class DecisionSnapshot:
    timestamp: str
    ticker: str
    price: float
    action: str  # BUY / SELL / HOLD
    position: float
    equity: float
    notes: str = ""

save_dir = "outputs"

def decision_to_csv(ticker,period, decision:dict):    
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(
        save_dir,
        f"{ticker}_{period}_dicision.csv"
    )
    #print(f'decision:{decision}')
    records = []
    records.append({
        "action": decision["action"],
        "strength":round(decision["strength"],3),
        "gate_mult":round(decision["gate_mult"],3),
        #"slope":decision["slope"],
        "raw_signal":decision["raw_action"],
        "raw_score":round(decision["raw_score"],3),
        "model_score":round(decision["model_score"],3)
    })
    df = pd.DataFrame(records)

    df.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False
    )


def prediction_to_csv(ticker,period, pre_result,close_df, decision:dict):
   
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(
        save_dir,
        f"{ticker}_{period}_prediction.csv"
    )

    low = pre_result.low
    median = pre_result.median
    high = pre_result.high
    model_score = pre_result.model_score
    atr = pre_result.atr
    price = pre_result.price

    gate_result = gater.evaluate(
            lower=low,
            mid=median,
            upper=high,
            close_df=close_df.values,
        )
    gate_allow = gate_result.allow
    gate_score = gate_result.score
    current_time = close_df.index[-1]

    records = []

    price_median_pct = f"{((median[-1]-price) / price * 100):.2f}%" if price else "0.00%"

    records.append({
        "datetime": current_time,   # ✅ 合并字段
        "low": round(low[-1],2),
        "median": round(median[-1],2),
        "high": round(high[-1],2),
        "model_score": round(model_score,2),
        "atr": round(atr,3),
        "price": price,
        "gate_allow": gate_allow,
        "gate_score": round(gate_score,3),
        "price_median_pct": price_median_pct,
        "action": decision["action"],
        "strength":round(decision["strength"],3),
        "gate_mult":round(decision["gate_mult"],3),
        #"slope":decision["slope"],
        "raw_signal":decision["raw_action"],
        "raw_score":round(decision["raw_score"],3),
        "final_model_score":round(decision["model_score"],3)    
    })

    df_pred = pd.DataFrame(records)

    df_pred.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False
    )

def plot_prediction(ticker, period, close_df):
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=3)

    close_df = close_df[close_df.index >= cutoff_date]

    csv_path = os.path.join(
        save_dir,
        f"{ticker}_{period}_prediction.csv"
    )

    df_history = pd.read_csv(csv_path, parse_dates=["datetime"])

    plt.figure(figsize=(10,5))

    # X轴
    x = range(len(close_df))

    # 1️⃣ 全部收盘价
    plt.plot(x, close_df.values, label="Close", color="black")

    # 2️⃣ 预测中位数
    pred_x = close_df.index.get_indexer(df_history["datetime"])
    plt.plot(pred_x, df_history["median"], label="Predicted Median")

    # 3️⃣ 预测区间
    plt.fill_between(
        pred_x,
        df_history["low"],
        df_history["high"],
        alpha=0.2,
        label="Prediction Range"
    )

    # 时间刻度
    step = 8
    ticks = list(x)[::step]
    labels = close_df.index.strftime("%m-%d %H:%M")[::step]

    plt.xticks(ticks, labels, rotation=45)

    plt.legend()
    plt.title("Chronos Prediction")
    plt.tight_layout()

    plt.show()

def plot_prediction_v1(ticker, period):
    csv_path = os.path.join(
        save_dir,
        f"{ticker}_{period}_prediction.csv"
    )
    df_history = pd.read_csv(csv_path, parse_dates=["datetime"])

    plt.figure(figsize=(10,5))

    x = range(len(df_history))

    plt.plot(x, df_history["price"], label="Price")
    plt.plot(x, df_history["median"], label="Predicted Median")

    plt.fill_between(
        x,
        df_history["low"],
        df_history["high"],
        alpha=0.2,
        label="Prediction Range"
    )

    step = 4

    ticks = list(x)[::step]
    labels = df_history["datetime"].dt.strftime("%m-%d %H:%M").iloc[::step]

    plt.xticks(ticks, labels, rotation=45)

    plt.legend()
    plt.title("Chronos Prediction")
    plt.tight_layout()
    plt.show()