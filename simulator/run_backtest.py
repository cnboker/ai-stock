from simulator.backtest_runner import BacktestRunner
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

runner = BacktestRunner(
    ticker="sz002137",
    days=15,
    period="15" #30 minutes
)

runner.run()
#runner.show()