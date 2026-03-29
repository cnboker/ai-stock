from simulator.backtest_runner import BacktestRunner
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

def run_backtest(ticker, period="60"):
    runner = BacktestRunner(
        ticker=ticker,
        period=period
    )
    # 执行 2/8 验证逻辑，返回 (train_stats, test_stats)
    return runner.run_split_backtest()

#run_backtest("sz300785")