from simulator.backtest_runner import BacktestRunner
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

runner = BacktestRunner(
    ticker="sh600938",
    days=120,
    period="120" #30 minutes
)

runner.run()
#runner.show()