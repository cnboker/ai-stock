import pandas as pd
from data.loader import load_stock_df
from main_live import on_bar
from plot.annotation import generate_tail_label
from plot.draw import draw_current_prediction, draw_prediction_band, draw_realtime_price, update_xaxes, update_yaxes
from predict.chronos_predict import run_prediction
from predict.prediction_store import update_prediction_history
from predict.time_utils import build_future_index
from datetime import datetime
from config.settings import ticker_name_map

def process_single_stock(fig, ticker, index, period, hs300_df,eq_feat):
    """
    单只股票：行情 → 预测 → 历史 → 绘图 → 标签
    """
    name = ticker_name_map.get(ticker,ticker)

    df = load_stock_df(ticker, period)
   
    low, median, high, model_score = run_prediction(
        df=df,
        hs300_df=hs300_df,
        ticker=ticker,
        period=period,
        eq_feat=eq_feat
    )
    last_price = df["close"].iloc[-1]
    print('last_price, model_score',type(last_price), last_price, model_score)
    atr = calc_atr(df)
  
    on_bar(ticker,name, df["close"],low, median, high, atr,model_score,eq_feat)

    future_index = build_future_index(df, period)

    history_pred = update_prediction_history(
        ticker, future_index, low, median, high
    )

    draw_prediction_band(fig, history_pred, index, name)
    draw_realtime_price(fig, df, index, name)
    draw_current_prediction(fig, future_index, low, median, high, index, name)
    update_yaxes(fig, last_price, index)
    update_xaxes(fig)
    return generate_tail_label(future_index, median, high, index, name)



def build_update_text(ticker="ALL"):
    now = datetime.now().strftime("%H:%M:%S")
    return f"更新: {now} | {ticker} | Chronos 实时预测"

def calc_atr(df: pd.DataFrame, period: int = 5) -> float:
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
