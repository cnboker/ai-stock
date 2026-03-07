import pandas as pd
import numpy as np

from config.settings import TICKER_PERIOD
from data.loader import load_stock_df, load_index_df
from predict.chronos_predict import run_prediction
from trade.trade_engine import execute_stock_decision

from infra.core.context import TradingSession
from infra.core.runtime import RunMode

from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder
from equity.equity_features import equity_features

from global_state import equity_engine


class BacktestRunner:

    def __init__(self, ticker: str, days: int = 7, period="3"):

        self.ticker = ticker
        self.days = days
        self.period = period

        self.df_all = load_stock_df(
            ticker=ticker,
            period=period,
        ).sort_index()

        if self.df_all.empty:
            raise ValueError("无股票数据")

        self.hs300_df = load_index_df(period)

        # ===== session (只创建一次) =====

        self.eq_recorder = create_equity_recorder(RunMode.SIM, ticker)
        self.position_mgr = create_position_manager(10000, RunMode.SIM)

        eq_feat = equity_features(self.eq_recorder.to_series())

        eq_decision = equity_engine.decide(
            eq_feat,
            self.position_mgr.has_any_position()
        )

        self.session = TradingSession(
            run_mode=RunMode.SIM,
            position_mgr=self.position_mgr,
            eq_recorder=self.eq_recorder,
            period=period,
            hs300_df=self.hs300_df,
            eq_feat=eq_feat,
            tradeIntent=eq_decision,
        )

        self.equity_curve = []

    def run(self):

        trade_days = (
            pd.Series(self.df_all.index.date)
            .drop_duplicates()
            .tail(self.days)
            .tolist()
        )

        print("回测日期:", trade_days)

        start_price = None
        end_price = None

        for trade_day in trade_days:

            self._simulate_day(trade_day)

            price = self.df_all[self.df_all.index.date == trade_day]["close"].iloc[-1]

            if start_price is None:
                start_price = price

            end_price = price

            #equity = self.eq_recorder.current_equity
            equity = self.eq_recorder.to_series().iloc[-1]
            print(f'equity ->{equity}')
            self.equity_curve.append(equity)

        self._report(start_price, end_price)

    def _simulate_day(self, trade_day):

        trade_day = pd.to_datetime(trade_day)

        df_today = self.df_all[self.df_all.index.date == trade_day.date()]
        df_history = self.df_all[self.df_all.index.date < trade_day.date()]

        for i in range(len(df_today)):

            df_slice = df_today.iloc[: i + 1]
            df_combined = pd.concat([df_history, df_slice])

            pre_result = run_prediction(
                df=df_combined,
                hs300_df=self.hs300_df,
                ticker=self.ticker,
                period=self.period,
                eq_feat=self.session.eq_feat
            )

            execute_stock_decision(
                ticker=self.ticker,
                close_df=df_slice["close"],
                pre_result=pre_result,
                session=self.session
            )

    def _report(self, start_price, end_price):

        equity = np.array(self.equity_curve)

        strategy_return = (equity[-1] - equity[0]) / equity[0]
        buy_hold_return = (end_price - start_price) / start_price

        # max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        print()
        print("=" * 40)
        print("Ticker:", self.ticker)
        print("Days:", self.days)
        print()
        print("Strategy Return :", round(strategy_return * 100, 2), "%")
        print("BuyHold Return  :", round(buy_hold_return * 100, 2), "%")
        print("Alpha           :", round((strategy_return - buy_hold_return) * 100, 2), "%")
        print("Max Drawdown    :", round(max_dd * 100, 2), "%")
        print("=" * 40)