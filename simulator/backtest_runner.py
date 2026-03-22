import os
import pandas as pd
import numpy as np
from infra.core.runtime import RunMode
from infra.core.context import TradingSession
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder
from equity.equity_features import equity_features
from global_state import equity_engine
from data.loader import load_stock_df, load_index_df
from predict.chronos_predict import run_prediction
from trade.trade_engine import execute_stock_decision
from config.settings import LOOKBACK_WINDOW

class BacktestRunner:
    def __init__(self, ticker, period, total_limit=1000):
        self.ticker = ticker
        self.period = period
        # 多加载一点数据，为第一天提供 Chronos 的 LOOKBACK 背景
        self.total_limit = total_limit + LOOKBACK_WINDOW 
        
        # 1. 加载数据
        full_df = load_stock_df(ticker=self.ticker, period=self.period).sort_index()
        self.df_all = full_df.tail(self.total_limit)
        self.hs300_df = load_index_df(self.period).tail(self.total_limit)

        if len(self.df_all) < self.total_limit:
            print(f"警告: 数据量不足，仅有 {len(self.df_all)} 条")

        # 2. 计算 2/8 切分点（基于 K 线索引）
        # 训练集从 LOOKBACK_WINDOW 开始，确保第一天就有历史数据
        self.train_start_idx = LOOKBACK_WINDOW
        self.split_idx = self.train_start_idx + int((len(self.df_all) - LOOKBACK_WINDOW) * 0.8)

    def _reset_engine(self):
        """重置引擎，确保训练集和测试集资金互不干扰"""
        self.eq_recorder = create_equity_recorder(RunMode.SIM, self.ticker)
        self.position_mgr = create_position_manager(100000, RunMode.SIM)
        
        # 初始化 Session
        eq_feat = equity_features(self.eq_recorder.to_series())
        self.eq_recorder.add(self.position_mgr.equity)
        eq_decision = equity_engine.decide(eq_feat, self.position_mgr.has_any_position())
        
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

    def run_split_backtest(self):
        """执行 2/8 验证逻辑"""
        # 提取所有可交易的日期
        all_dates = pd.Series(self.df_all.index.date).drop_duplicates().tolist()
        
        # 根据 split_idx 找到切分日期
        split_date = self.df_all.index[self.split_idx].date()
        
        # 划分日期集合（跳过最开始的预留窗口日期）
        train_dates = [d for d in all_dates if d < split_date][5:] # 略微多跳几天确保 buffer
        test_dates = [d for d in all_dates if d >= split_date]

        # 分别运行
        print(f"--- 运行训练集 ({len(train_dates)} 天) ---")
        train_res = self._execute_loop(train_dates)
        
        print(f"--- 运行验证集 ({len(test_dates)} 天) ---")
        test_res = self._execute_loop(test_dates)
        
        return train_res, test_res

    def _execute_loop(self, target_dates):
        self._reset_engine()
        start_price = None
        end_price = None

        for trade_day in target_dates:
            self._simulate_day(trade_day)
            
            day_data = self.df_all[self.df_all.index.date == trade_day]
            if day_data.empty: continue
            
            price = day_data["close"].iloc[-1]
            if start_price is None: start_price = price
            end_price = price
            self.equity_curve.append(self.position_mgr.equity)
            
        return self._report(start_price, end_price)

    def _simulate_day(self, trade_day):
        # 这里的 get_history 会自动利用 df_all 中的历史数据，即使是测试集第一天也能向上回溯
        df_today = self.df_all[self.df_all.index.date == trade_day]
        df_history_fixed = self.get_history(trade_day, self.df_all)
        hs300_history = self.get_history(trade_day, self.hs300_df)

        for i in range(len(df_today)):
            self.eq_recorder.add(self.position_mgr.equity)
            df_slice = df_today.iloc[: i + 1]
            ticker_df = pd.concat([df_history_fixed, df_slice])

            pre_result = run_prediction(
                df=ticker_df,
                hs300_df=hs300_history,
                ticker=self.ticker,
                period=self.period,
                eq_feat=self.session.eq_feat,
            )

            execute_stock_decision(
                ticker=self.ticker,
                close_df=ticker_df["close"],
                pre_result=pre_result,
                session=self.session,
            )

    def get_history(self, trade_day, df):
        trade_day = pd.to_datetime(trade_day)
        today_start_ts = df[df.index.date == trade_day.date()].index[0]
        return df[df.index < today_start_ts].tail(LOOKBACK_WINDOW)

    def _report(self, start_price, end_price):
        equity = np.array(self.equity_curve)
        if len(equity) < 2: return {"Strategy_Return": -10, "Max_Drawdown": 100, "Trade_Count": 0}
        
        strat_ret = (equity[-1] - equity[0]) / equity[0]
        buy_hold_return = (end_price - start_price) / start_price
        peak = np.maximum.accumulate(equity)
        max_dd = ((equity - peak) / peak).min()
        
        print()
        print("=" * 40)
        print("Ticker:", self.ticker)
        print()
        print("Strategy Return :", round(strat_ret * 100, 2), "%")
        print("BuyHold Return  :", round(buy_hold_return * 100, 2), "%")
        print(
            "Alpha           :",
            round((strat_ret - buy_hold_return) * 100, 2),
            "%",
        )
        print("Max Drawdown    :", round(max_dd * 100, 2), "%")
        print("Trade Count  :", self.position_mgr.total_trade_count)
        print("=" * 40)

        return {
            "Strategy_Return": round(strat_ret * 100, 4),
            "Max_Drawdown": round(max_dd * 100, 4),
            "Trade_Count": self.position_mgr.total_trade_count
        }