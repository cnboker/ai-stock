from simulator.backtest_runner import BacktestRunner
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

def run_backtest(ticker):
    runner = BacktestRunner(
        ticker=ticker,
        period="60" #60 minutes
    )
    # 执行 2/8 验证逻辑，返回 (train_stats, test_stats)
    return runner.run_split_backtest()

#run_backtest("sz000617")