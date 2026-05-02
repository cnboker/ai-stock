import asyncio
from backtest.backtest_runner import BacktestRunner
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.data.dataloader"
)


def run_backtest(ticker, period="30"):

    async def get_data():
        runner = BacktestRunner(ticker=ticker, period=period)
        # 执行 2/8 验证逻辑，返回 (train_stats, test_stats)
        return await runner.run_split_backtest()

        # 同步阻塞地等待异步结果
    return asyncio.run(get_data())


# run_backtest("sh512760")
