from equity.equity_recorder import EquityRecorder
from infra.core.runtime import RunMode


def create_equity_recorder(
    run_mode: RunMode,
    ticker: str | None = None,
    trade_date: str | None = None
):
    if run_mode == RunMode.LIVE:
        path = "data/live/equity.csv"
    else:
        path = f"data/sim/{ticker}_{trade_date}_equity.csv"

    eq_recorder = EquityRecorder(path=path)
    eq_recorder._load_disk()
    return eq_recorder
