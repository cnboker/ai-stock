import pandas as pd

from infra.core.runtime import GlobalState
from data.loader import load_stock_df
from infra.core.dynamic_settings import settings
from trade.trade_engine import execute_final_order, execute_stock_decision
class StockSim:
    def __init__(self, ticker, session, position_mgr):
        self.ticker = ticker
        self.session = session
        # 每个股票实例拥有独立的管理器
        self.position_mgr = position_mgr
        self.net_backtest_size = 104
        required_size = self.net_backtest_size + settings.WINDOW
        full_df = load_stock_df(
            ticker=self.ticker, period=self.session.period
        ).sort_index()
        self.full_df = full_df.tail(required_size)
        self.hs300_df = session.hs300_df.tail(required_size)

    def get_history(self, trade_day, full_df):
        """获取交易日之前的历史数据"""
        return full_df[full_df.index.date < trade_day]

    def process_ticker_bar(
        self,
        ticker,
        i,
        trade_day, #pd.Timestamp
        callback=None,
        arbiter=None
    ):
        """
        核心函数：处理单根 K 线，产出信号并塞入仲裁器
        """
     
        df_history = self.get_history(trade_day, self.full_df)
        hs300_history = self.get_history(trade_day, self.hs300_df)
        df_today = self.full_df[self.full_df.index.date == trade_day]
        hs300_today = self.hs300_df[self.hs300_df.index.date == trade_day]
       
        current_k = df_today.iloc[i]
        timestamp = current_k.name

        # 1. 更新全局价格
        if callback:
            # 解决之前提到的 RuntimeError 问题
            GlobalState.tickers_price[ticker] = callback(ticker, timestamp)
        else:
            GlobalState.tickers_price[ticker] = current_k["close"]

        # 2. 同步市场时间给仓位管理器
        self.position_mgr.current_market_time = timestamp

        # 3. 构造决策所需的上下文数据集 (包含当前Bar)
        # 优化建议：如果频繁 concat 慢，可在外部拼接好全量数据后在此处 iloc 切片
        ticker_df = pd.concat([df_history, df_today.iloc[: i + 1]])
        hs300_df = pd.concat([hs300_history, hs300_today.iloc[: i + 1]])

        # 4. 执行决策算法逻辑
        # 假设 execute_stock_decision 返回 {'type': 'candidate', 'score': 0.8, ...}
        res = execute_stock_decision(
            ticker=ticker,
            hs300_df=hs300_df,
            ticker_df=ticker_df,
            session=self.session,
        )

        # 5. 如果产生有效信号，塞入当前时段的仲裁器
        if res and res.get("type") == "candidate":
            # 确保 res 中包含必要的决策信息
            res['ticker'] = ticker 
            arbiter.add_candidate(res)

