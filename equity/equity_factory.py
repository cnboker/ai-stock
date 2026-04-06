from equity.equity_recorder import EquityRecorder
from infra.core.runtime import GlobalState, RunMode
import os

def create_equity_recorder(
    
):
    if GlobalState.mode == RunMode.LIVE:
        path = "data/live/equity.csv"
    else:
        path = f"data/sim/equity.csv"
        if os.path.exists(path):
            os.remove(path)
            
    eq_recorder = EquityRecorder(path=path)
    eq_recorder._load_disk()
    return eq_recorder
