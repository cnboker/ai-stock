# simulate_cli.py
import argparse
from datetime import datetime
import warnings
from backtest.backtest_runner import BacktestRunner
from equity.equity_factory import create_equity_recorder
from infra.core.runtime import GlobalState, RunMode
from position import watchlist_loader
from position.position_factory import create_position_manager
from position.position_loader import LivePositionLoader

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.data.dataloader"
)

GlobalState.mode == RunMode.SIM
# python3 simulate_cli.py --date=2026-4-30
# 实时模拟跑成绩，对比实盘盈亏，验证回测的准确性
def main():
    parser = argparse.ArgumentParser(
        description="Simulate a trading day and compare with real trades"
    )
    eq_recorder = create_equity_recorder(RunMode.SIM)

    parser.add_argument("--date", required=True, help="Date to simulate YYYY-MM-DD")
    args = parser.parse_args()
    position_mgr = create_position_manager(0)
    pos_loader = LivePositionLoader("state/live_positions.yaml", position_mgr)
    pos_loader.sync()
    loader = watchlist_loader.live_watchlist_hot_load(position_mgr)
    tickers = position_mgr.get_tickers_from_positions_and_watchlist()
    # print(f"准备模拟 {tickers} 在 {args.date} 的交易情况...")
    dates = ['2026-04-27', '2026-04-28', '2026-04-29', '2026-04-30']
    for date in dates:
         print(f"准备模拟 {tickers} 在 {date} 的交易情况...")
         target_dt = datetime.strptime(date, "%Y-%m-%d")
         for ticker in tickers:
             runner = BacktestRunner(ticker=ticker, period="30")
             runner.reset_engine(equity_recorder=eq_recorder,position_mgr=position_mgr)  # 确保每次模拟前都重置引擎状态
             runner.simulate_day(trade_day=target_dt.date())
            

    print(f"模拟完成！, equity: {position_mgr.equity:.2f} ")

def load_config():
    """加载配置文件，支持动态更新"""
    # 这里可以扩展为从文件、环境变量或远程配置中心加载
    config = {
        "TICKER_PERIOD": "30",
        "UPDATE_INTERVAL_SEC": 60,
    }
    return config

if __name__ == "__main__":
    main()
