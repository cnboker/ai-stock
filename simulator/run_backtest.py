from simulator.backtest_runner import BacktestRunner

runner = BacktestRunner(
    ticker="sh600938",
    days=21,
    period="120"
)

runner.run()