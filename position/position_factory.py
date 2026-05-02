# position/position_factory.py

from infra.core.runtime import RunMode
from position.position_loader import LivePositionLoader
from position.position_manager import PositionManager
from position import watchlist_loader

def create_position_manager(
    init_cash: float = 0.0,
    mode: RunMode = RunMode.LIVE,
) -> PositionManager:
    """
    创建独立的 PositionManager 实例
    - LIVE / SIM 完全隔离
    - 不复用任何全局状态
    """
    position_mgr = PositionManager(init_cash=init_cash)    
    position_mgr.clear()
    if mode == RunMode.LIVE:
        print("🚀 创建 LIVE 模式的 PositionManager")
        pos_loader = LivePositionLoader("state/live_positions.yaml", position_mgr)
        pos_loader.sync()
        watchlist_loader.live_watchlist_hot_load(position_mgr)
    elif mode == RunMode.SIM:
        print("🚀 创建 SIM 模式的 PositionManager")
        position_mgr.clear()  # 确保每次模拟前都清空仓位状态
        position_mgr.save(RunMode.SIM)
        pos_loader = LivePositionLoader("state/sim_positions.yaml", position_mgr)
        pos_loader.sync()
        watchlist_loader.live_watchlist_hot_load(position_mgr)

    return position_mgr
