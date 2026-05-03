# simulate_cli.py
import asyncio
from datetime import datetime
import warnings

import pandas as pd
from rich.diagnose import report
from backtest.StockSim import StockSim
from backtest.backtest_runner import BacktestRunner
from data.loader import get_price_timestamp, load_index_df
from global_state import equity_engine
from equity.equity_factory import create_equity_recorder
from equity.equity_features import equity_features
from infra.core.dynamic_settings import use_config
from infra.core.runtime import GlobalState, RunMode
from infra.core.trade_session import TradingSession
from position.position_factory import create_position_manager
from trade.signal_arbiter import SignalArbiter
from trade.trade_engine import execute_final_order
from infra.core.config_manager import dynamic_config_manager

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.data.dataloader"
)

GlobalState.mode == RunMode.SIM


# python3 simulate_cli.py --date=2026-4-30
# 实时模拟跑成绩，对比实盘盈亏，验证回测的准确性
def main():

    eq_recorder = create_equity_recorder(RunMode.SIM)
    eq_recorder.reset()  # 每次模拟前重置权益记录器，确保数据干净

    position_mgr = create_position_manager(
        100000, RunMode.SIM
    )  # 每次模拟前创建独立的 PositionManager 实例，确保状态隔离

    tickers = position_mgr.get_tickers_from_positions_and_watchlist()

    dates = ["2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30"]
    #dates = ["2026-04-30"]
    report = {}
    for ticker in tickers:
        position_mgr.clear()
        eq_recorder.reset()
        position_mgr.cash = 100000  # 每次模拟前重置初始资金
        for date in dates:
            target_dt = datetime.strptime(date, "%Y-%m-%d")
            runner = BacktestRunner(ticker=ticker, period="30")
            runner.reset_engine(
                equity_recorder=eq_recorder, position_mgr=position_mgr
            )  # 确保每次模拟前都重置引擎状态
            runner.simulate_day(
                runner.full_pool_df,
                runner.full_pool_hs300,
                trade_day=target_dt.date(),
                callback=get_price_callback,
            )            
        report[ticker] = position_mgr.equity

    print(f"模拟完成！, equity: {report} ")


def get_price_callback(ticker, timestamp):
    async def get_price():
        price = await get_price_timestamp(ticker, timestamp)
        return price

    price = asyncio.run(get_price())
    print(f"回调函数：更新 {ticker} 的价格为 {price} at {timestamp}")
    return price


# 多只股票并行模拟跑成绩，对比实盘盈亏，验证回测的准确性
def run_simulation():
    period = "30"
    eq_recorder = create_equity_recorder(RunMode.SIM)
    eq_recorder.reset()  # 每次模拟前重置权益记录器，确保数据干净

    position_mgr = create_position_manager(
        100000, RunMode.SIM
    )  # 每次模拟前创建独立的 PositionManager 实例，确保状态隔离

    tickers = position_mgr.get_tickers_from_positions_and_watchlist()

    dates = ["2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30"]
    #dates = ["2026-04-30"]
    hs300_df = load_index_df("30").sort_index()
    eq_feat = equity_features(eq_recorder.to_series())
    eq_decision = equity_engine.decide(eq_feat, position_mgr.has_any_position())
    session = TradingSession(
        position_mgr=position_mgr,
        eq_recorder=eq_recorder,
        period=period,
        hs300_df=hs300_df,
        eq_feat=eq_feat,
        tradeIntent=eq_decision,
    )
    # 1. 构造实例字典
    ticker_instances = {
        tk: StockSim(tk, session, position_mgr=position_mgr) for tk in tickers
    }

    for trade_day_str in dates:
        # 每一时刻开启一个新仲裁器
        trade_day = pd.to_datetime(trade_day_str).date()
        df_today = session.hs300_df[session.hs300_df.index.date == trade_day]
        # print(f"开始模拟 {trade_day}，共 {len(df_today)} 根K线")
        
        for i in range(len(df_today)):
            arbiter = SignalArbiter(max_slots=1)
            for ticker, instance in ticker_instances.items():
                final_config = dynamic_config_manager.load_params(ticker)
                with use_config(final_config):
                    instance.process_ticker_bar(
                        ticker,
                        i,
                        trade_day,
                        arbiter=arbiter,
                    )
            # 时刻结束，执行最优决策
            best = arbiter.get_best_decisions()
            #print(f"模拟 {trade_day} 的第 {i+1}/{len(df_today)} 根K线，时间戳: {df_today.iloc[i].name}, 收盘价: {df_today.iloc[i]['close']}, 最优决策: {best}")
            if best:
                # 注意：这里需要决定把单子下到哪个实例的 manager 里
                target_instance = ticker_instances[best[0]["ticker"]]
                execute_final_order(best[0], position_mgr)

    print(f"模拟完成！, equity: {position_mgr.equity} ") 

if __name__ == "__main__":
    run_simulation()
