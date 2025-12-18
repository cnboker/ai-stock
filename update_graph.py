from data.loader import load_stock_df
from plot.annotation import generate_tail_label
from plot.draw import draw_current_prediction, draw_prediction_band, draw_realtime_price, update_yaxes
from predict.chronos_predict import run_prediction
from predict.prediction_store import update_prediction_history
from strategy.signal_engine import make_signal, print_signal
from time_utils import build_future_index
from datetime import datetime


def process_single_stock(fig, stock, index, period, hs300_df):
    """
    单只股票：行情 → 预测 → 历史 → 绘图 → 标签
    """
    ticker = stock["code"]
    name = stock["name"]

    df = load_stock_df(ticker, period)

    low, median, high = run_prediction(
        df=df,
        hs300_df=hs300_df,
        ticker=ticker,
        period=period,
    )
    last_price = df["close"].iloc[-1]
    signal = make_signal(
            low=low,
            median=median,
            high=high,
            last_price=last_price,
        )
    print_signal(ticker,signal)
    future_index = build_future_index(df, period)

    history_pred = update_prediction_history(
        ticker, future_index, low, median, high
    )

    draw_prediction_band(fig, history_pred, index, name)
    draw_realtime_price(fig, df, index, name)
    draw_current_prediction(fig, future_index, low, median, high, index, name)
    update_yaxes(fig, last_price, index)
    return generate_tail_label(future_index, median, high, index, name)



def build_update_text(ticker="ALL"):
    now = datetime.now().strftime("%H:%M:%S")
    return f"更新: {now} | {ticker} | Chronos 实时预测"
