import threading
from equity.equity_engine import EquityEngine
from infra.core.runtime import RunMode


equity_engine = EquityEngine()
state_lock = threading.RLock()
