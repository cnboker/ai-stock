from simulator.backtest_runner import BacktestRunner
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

def run_backtest():
    runner = BacktestRunner(
        ticker="sz000617",
        days=10,
        period="10" #30 minutes
    )
    runner.run()

run_backtest()
