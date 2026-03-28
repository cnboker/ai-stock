import os
import pandas as pd
import numpy as np
from infra.core.runtime import RunMode
from infra.core.trade_session import TradingSession
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder
from equity.equity_features import equity_features
from global_state import equity_engine
from data.loader import load_stock_df, load_index_df
from predict.chronos_predict import run_prediction
from trade.trade_engine import execute_stock_decision
from config.settings import LOOKBACK_WINDOW

class BacktestRunner:
    def __init__(self, ticker, period, total_limit=760):
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
        # 记录初始资金，用于后续计算
        self.initial_capital = self.position_mgr.equity
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

            execute_stock_decision(
                ticker=self.ticker,
                hs300_df = hs300_history,
                ticker_df=ticker_df,
                session=self.session,
            )

    def get_history(self, trade_day, df):
        trade_day = pd.to_datetime(trade_day)
        today_start_ts = df[df.index.date == trade_day.date()].index[0]
        return df[df.index < today_start_ts].tail(LOOKBACK_WINDOW)

    def _report(self, start_price, end_price):
        equity = np.array(self.equity_curve)
        
        # 💡 核心：定义你的作战基数
        # 既然 4000 是最大投资额度，我们用它作为分母
        ACTIVE_BUDGET = 5000.0 
        
        # 计算绝对盈亏额（比如赚了 80 元）
        absolute_profit = equity[-1] - equity[0]
        
        # 2. 计算策略收益 (基于作战额度)
        # 这样赚 80 元就是 2%，而不是 0.08%
        strat_ret = absolute_profit / ACTIVE_BUDGET
        
        # 3. 计算最大回撤 (同样基于作战额度)
        # 这样如果 4000 元亏了 400，回撤显示为 -10%，而不是 -0.4%
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / ACTIVE_BUDGET
        max_dd = drawdowns.min()
        
        # 4. Alpha 计算
        buy_hold_ret = (end_price - start_price) / start_price
        alpha = strat_ret - buy_hold_ret

        # --- 打印报告 ---
        print("\n" + "=" * 40)
        print(f"Ticker: {self.ticker}")
        print(f"Strategy Return : {round(strat_ret * 100, 2)} %")
        print(f"BuyHold Return  : {round(buy_hold_ret * 100, 2)} %")
        print(f"Alpha           : {round(alpha * 100, 2)} %")
        print(f"Max Drawdown    : {round(max_dd * 100, 2)} %")
        print(f"Trade Count     : {self.position_mgr.total_trade_count}")
        print("=" * 40 + "\n")

        # ... 返回给 Optuna 的数据保持百分比形式 ...
        return {
            "Strategy_Return": round(strat_ret * 100, 4), # 比如返回 2.0
            "Max_Drawdown": round(max_dd * 100, 4),
            "Trade_Count": self.position_mgr.total_trade_count,
            "Alpha": round(alpha * 100, 4)
        }

    def _report_1(self, start_price, end_price):
        # 转换净值曲线为 numpy 数组方便计算
        equity = np.array(self.equity_curve)
        
        # 💡 关键修改：定义“作战本金”
        # 假设你的单笔上限是 5000 5000 元的表现：
        ACTIVE_CAPITAL = 5000.0 
        
        # 计算绝对盈亏 (比如赚了 80 元)
        net_pnl = equity[-1] - self.initial_capital
        
        # 1. 真实作战收益率 (80 / 4000 = 2%)
        strat_ret = net_pnl / ACTIVE_CAPITAL
        
        # 2. 计算最大回撤 (也要基于 ACTIVE_CAPITAL 才有意义)
        # 否则 10万本金下，400元的跌幅看起来只有 0.4%，太迷惑人了
        peak = np.maximum.accumulate(equity)
        # 这里的 drawdown 计算要反映出 4000 元亏了多少
        max_dd = ((equity - peak) / ACTIVE_CAPITAL).min()

        # 3. Alpha 计算
        buy_hold_ret = (end_price - start_price) / start_price
        alpha = strat_ret - buy_hold_ret

        # --- 打印报告 ---
        print("\n" + "=" * 40)
        print(f"Ticker: {self.ticker}")
        print(f"Strategy Return : {round(strat_ret * 100, 2)} %")
        print(f"BuyHold Return  : {round(buy_hold_ret * 100, 2)} %")
        print(f"Alpha           : {round(alpha * 100, 2)} %")
        print(f"Max Drawdown    : {round(max_dd * 100, 2)} %")
        print(f"Trade Count     : {self.position_mgr.total_trade_count}")
        print("=" * 40 + "\n")

        # 返回给 Optuna 的字典，字段名要和 get_advanced_score 对应
        return {
            "Strategy_Return": round(strat_ret * 100, 4),
            "Max_Drawdown": round(max_dd * 100, 4),
            "Trade_Count": self.position_mgr.total_trade_count,
            "Alpha": round(alpha * 100, 4)
        }