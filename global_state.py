import threading
from infra.core.runtime import RunMode
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder

# ===== 全局唯一状态 =====
position_mgr = create_position_manager(0, RunMode.LIVE)
eq_recorder = create_equity_recorder(RunMode.LIVE)

state_lock = threading.RLock()
