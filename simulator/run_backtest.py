from simulator.backtest_runner import BacktestRunner
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

runner = BacktestRunner(
    ticker="sz300697",
    days=30,
    period="30" #30 minutes
)

runner.run()
#runner.show()