from simulator.backtest_runner import BacktestRunner
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

def run_backtest(ticker):
    runner = BacktestRunner(
        ticker=ticker,
        days=15,
        period="10" #30 minutes
    )
    return runner.run()

