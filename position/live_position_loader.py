import os
import time
import yaml
from infra.core.runtime import RunMode
from typing import Callable, Optional
from global_state import state_lock
from position.position_manager import PositionManager


class LivePositionLoader:
    def __init__(
        self,
        path: str,
        position_mgr: PositionManager,
        on_reload: Optional[Callable[[dict], None]] = None,
    ):
        self.path = path
        self._last_mtime = 0
        self._cache = None
        self.position_mgr = position_mgr
        self._on_reload = on_reload

    def load(self, force=False):
        try:
            mtime = os.path.getmtime(self.path)
        except FileNotFoundError:
            raise RuntimeError(f"{self.path} not found")

        reloaded = False

        if force or self._cache is None or mtime != self._last_mtime:
            with open(self.path, "r", encoding="utf-8") as f:
                self._cache = yaml.safe_load(f) or {}

            # 🔐 唯一修改仓位的地方
            with state_lock:
                self.position_mgr.load_from_yaml(self._cache)

            self._last_mtime = mtime
            reloaded = True
            print(f"[LivePosition] reloaded @ {time.strftime('%H:%M:%S')}")

            if self._on_reload:
                self._on_reload(self._cache)

        return self._cache, reloaded


def reload_if_needed(data: dict):
    print(f"[Callback] positions updated: {list(data.keys())}")
    


live_position_loader = None
def live_positions_hot_load(position_mgr, stop_event=None, interval=1.0):
    global live_position_loader
    if live_position_loader is None:
        live_position_loader = LivePositionLoader(
            path="state/live_positions.yaml",
            position_mgr=position_mgr,
            on_reload=reload_if_needed,
        )

    while stop_event is None or not stop_event.is_set():
        live_position_loader.load()
        time.sleep(interval)
