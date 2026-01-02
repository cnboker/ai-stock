# position/position_factory.py

from infra.core.runtime import RunMode
from position.position_manager import PositionManager


def create_position_manager(
    init_cash:0,
    run_mode: RunMode
) -> PositionManager:
    """
    创建独立的 PositionManager 实例
    - LIVE / SIM 完全隔离
    - 不复用任何全局状态
    """

    pm = PositionManager(init_cash=init_cash,run_mode=run_mode.value)    
    pm.clear()

    return pm
