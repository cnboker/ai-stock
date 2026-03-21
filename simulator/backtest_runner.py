import os

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
from simulator.snapshot import prediction_to_csv, plot_prediction


class BacktestRunner:

    def __init__(self, ticker, days, period):
        self.ticker = ticker
        self.days = days
        self.period = period
        self.DrawChart = False
        self.SaveDecision = False
        self.df_all = load_stock_df(
            ticker=self.ticker,
            period=self.period,
        ).sort_index()

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=0)
        self.df_all = self.df_all[self.df_all.index < cutoff]

        if self.df_all.empty:
            raise ValueError("无股票数据")

        self.hs300_df = load_index_df(self.period)
        self.hs300_df = self.hs300_df[self.hs300_df.index < cutoff]

        self.eq_recorder = create_equity_recorder(RunMode.SIM, self.ticker)
        self.position_mgr = create_position_manager(100000, RunMode.SIM)

        eq_feat = equity_features(self.eq_recorder.to_series())
        self.eq_recorder.add(self.position_mgr.equity)
        eq_decision = equity_engine.decide(
            eq_feat, self.position_mgr.has_any_position()
        )

        self.session = TradingSession(
            run_mode=RunMode.SIM,
            position_mgr=self.position_mgr,
            eq_recorder=self.eq_recorder,
            period=self.period,
            hs300_df=self.hs300_df,
            eq_feat=eq_feat,
            tradeIntent=eq_decision,
        )

        self.equity_curve = []

    def show(self):
        plot_prediction(self.ticker, self.period, self.df_all["close"])

    def run(self):
     
        trade_days = (
            pd.Series(self.df_all.index.date).drop_duplicates().tail(self.days).tolist()
        )

        start_price = None
        end_price = None
        save_dir = "outputs"
        pre_csv_path = os.path.join(
            save_dir, f"{self.ticker}_{self.period}_prediction.csv"
        )
        decision_csv_path = os.path.join(
            save_dir, f"{self.ticker}_{self.period}_dicision.csv"
        )

        # 如果文件存在就删除
        if os.path.exists(pre_csv_path):
            os.remove(pre_csv_path)
        if os.path.exists(decision_csv_path):
            os.remove(decision_csv_path)

        for trade_day in trade_days:

            self._simulate_day(trade_day)

            price = self.df_all[self.df_all.index.date == trade_day]["close"].iloc[-1]

            if start_price is None:
                start_price = price

            end_price = price

            # equity = self.eq_recorder.current_equity
            equity = self.eq_recorder.to_series().iloc[-1]
            # print(f'equity ->{equity}')
            self.equity_curve.append(equity)

        result = self._report(start_price, end_price)

        if self.DrawChart:
            plot_prediction(self.ticker, self.period, self.df_all["close"])
            
        return result
    
    def _simulate_day(self, trade_day):

        trade_day = pd.to_datetime(trade_day)

        df_today = self.df_all[self.df_all.index.date == trade_day.date()]
        df_history = self.df_all[self.df_all.index.date < trade_day.date()]

        for i in range(len(df_today)):
            # update equity
            self.eq_recorder.add(self.position_mgr.equity)
            df_slice = df_today.iloc[: i + 1]
            df_combined = pd.concat([df_history, df_slice])

            pre_result = run_prediction(
                df=df_combined,
                hs300_df=self.hs300_df,
                ticker=self.ticker,
                period=self.period,
                eq_feat=self.session.eq_feat,
            )

            decision = execute_stock_decision(
                ticker=self.ticker,
                close_df=df_combined["close"],
                pre_result=pre_result,
                session=self.session,
            )
            if decision is not None and self.SaveDecision:
                prediction_to_csv(
                    self.ticker, self.period, pre_result, df_combined["close"], decision
                )
            # decision_to_csv(self.ticker,self.period, decision)

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
        print(
            "Alpha           :",
            round((strategy_return - buy_hold_return) * 100, 2),
            "%",
        )
        print("Max Drawdown    :", round(max_dd * 100, 2), "%")
        print("=" * 40)

        return {
            "Ticker:": self.ticker,
            "Days:": self.days,
            "Strategy_Return": round(strategy_return * 100, 2),
            "BuyHold_Return": round(buy_hold_return * 100, 2),
            "Max_Drawdown": round(max_dd * 100, 2),
            "Trade_Count": self.position_mgr.total_trade_count
        }