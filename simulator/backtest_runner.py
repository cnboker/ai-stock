import os
import pandas as pd
import numpy as np
from infra.core.runtime import GlobalState, RunMode
from infra.core.trade_session import TradingSession
from infra.utils.time_profile import timer_decorator
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder
from equity.equity_features import equity_features
from global_state import equity_engine
from data.loader import load_stock_df, load_index_df
from trade.trade_engine import execute_stock_decision
from infra.core.runtime import GlobalState, RunMode
class BacktestRunner:
    def __init__(self, ticker, period):
        GlobalState.mode = RunMode.SIM
        self.ticker = ticker
        self.period = period

        # 1. 加载全量数据池 (假设有 1000 根线)
        self.full_pool_df = load_stock_df(ticker=self.ticker, period=self.period).sort_index()
        self.full_pool_hs300 = load_index_df(self.period).sort_index()

        # 2. 定义固定的“实战考试区”长度 (9天训练 + 4天验证 = 13天)
        # A股一天8根K线，13 * 8 = 104
        self.net_backtest_size = 104 
        
        # 3. 动态计算总需长度：实战区 + 策略窗口(预热区)
        # 这样无论 window 是 20 还是 160，都能保证最后 104 根线是用来考试的
        required_size = self.net_backtest_size + GlobalState.strategy_window
        
        # 4. 截取用于本次 Trial 的数据子集
        self.df_all = self.full_pool_df.tail(required_size)
        self.hs300_df = self.full_pool_hs300.tail(required_size)

        # 5. 确定 8/2 切分点 (基于 104 根净回测数据)
        # 我们从 df_all 的末尾向前数 104 根作为起点
        self.backtest_start_idx = len(self.df_all) - self.net_backtest_size
        # 训练集占净回测区的 80% (约 83 根，即 10.4 天，实际按日期切分更准)
        self.split_rel_idx = int(self.net_backtest_size * 0.8)
      

    def _reset_engine(self):
        """重置引擎，确保训练集和测试集资金互不干扰"""
        self.eq_recorder = create_equity_recorder()
        self.position_mgr = create_position_manager(1000000)
        
        # 初始化 Session
        eq_feat = equity_features(self.eq_recorder.to_series())
        self.eq_recorder.add(self.position_mgr.equity)
        eq_decision = equity_engine.decide(eq_feat, self.position_mgr.has_any_position())
        
        self.session = TradingSession(
            position_mgr=self.position_mgr,
            eq_recorder=self.eq_recorder,
            period=self.period,
            hs300_df=self.hs300_df,
            eq_feat=eq_feat,
            tradeIntent=eq_decision,
        )
        self.equity_curve = []

    def run_split_backtest(self):
        """执行 2/8 验证逻辑：9天训练 + 4天验证"""
        # 提取当前 df_all 中的所有日期
        all_dates = pd.Series(self.df_all.index.date).unique().tolist()
        battle_days = 21
        split_battle_days = 14
        # 确保我们只在最后 13 天进行回测，之前的日期全部留作 window 预热
        battle_dates = all_dates[-battle_days:] if len(all_dates) >= battle_days else all_dates
        
        # 严格执行 9+4 分类
        train_dates = battle_dates[:split_battle_days]
        test_dates = battle_dates[split_battle_days:]

        # 打印调试信息，确认数据对齐
        print(f"--- 策略窗口(Window): {GlobalState.strategy_window} ---")
        print(f"--- 训练集 ({len(train_dates)} 天): {train_dates[0]} ~ {train_dates[-1]} ---")
        train_res = self._execute_loop(train_dates)
        
        print(f"--- 验证集 ({len(test_dates)} 天): {test_dates[0]} ~ {test_dates[-1]} ---")
        test_res = self._execute_loop(test_dates)
        
        return train_res, test_res

    @timer_decorator
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

    # 模拟一天的交易，完全复用实盘逻辑
    #@timer_decorator 0.34s 左右，主要耗在模型推理上
    def _simulate_day(self, trade_day):
        # 这里的 get_history 会自动利用 df_all 中的历史数据，即使是测试集第一天也能向上回溯
        df_today = self.full_pool_df[self.full_pool_df.index.date == trade_day]
        df_hs300_today = self.full_pool_hs300[self.full_pool_hs300.index.date == trade_day]
        df_history = self.get_history(trade_day, self.full_pool_df)
        hs300_history = self.get_history(trade_day, self.full_pool_hs300)
        #print(f"模拟交易日 {trade_day}，历史数据点数: {len(df_history_fixed)}, 今日数据点数: {len(df_today)}")
        for i in range(len(df_today)):
            current_k = df_today.iloc[i]
            #提取行情时间撮
            self.position_mgr.current_market_time = current_k.name

            df_slice = df_today.iloc[: i + 1]
            hs300_slice = df_hs300_today.iloc[: i + 1]

            ticker_df = pd.concat([df_history, df_slice])
            hs300_df = pd.concat([hs300_history,hs300_slice])   
    
            execute_stock_decision(
                ticker=self.ticker,
                hs300_df = hs300_df,
                ticker_df=ticker_df,
                session=self.session,
            )
            self.eq_recorder.add(self.position_mgr.equity)

    #它就返回过去LOOKBACK_WINDOW个单位（天或分钟）的数据
    def get_history(self, trade_day, df_pool):
        trade_day = pd.to_datetime(trade_day)
        #找到当天的开盘第一分钟（如果是分钟级数据）或当天的时间点
        today_start_ts = df_pool[df_pool.index.date == trade_day.date()].index[0]
        return df_pool[df_pool.index < today_start_ts]

    def _report(self, start_price, end_price):
        equity = np.array(self.equity_curve)
        
        # 💡 核心：定义你的作战基数
        # 既然 4000 是最大投资额度，我们用它作为分母
        ACTIVE_BUDGET = self.position_mgr.max_occupied if self.position_mgr.max_occupied > 0 else 1
        print(f"作战基数 (ACTIVE_BUDGET): {ACTIVE_BUDGET:.2f}")
        
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

  