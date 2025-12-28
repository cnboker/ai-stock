import os
import time
import yaml
from position.position_manager import position_mgr
from typing import Callable, Optional


class LivePositionLoader:
    def __init__(
        self,
        path: str,
        on_reload: Optional[Callable[[dict], None]] = None
    ):
        self.path = path
        self._last_mtime = 0
        self._cache = None
        self._on_reload = on_reload

    def load(self, force=False):
        try:
            mtime = os.path.getmtime(self.path)
        except FileNotFoundError:
            raise RuntimeError(f"{self.path} not found")

        reloaded = False

        if force or self._cache is None or mtime != self._last_mtime:
            with open(self.path, "r", encoding="utf-8") as f:
                self._cache = yaml.safe_load(f)
            self._last_mtime = mtime
            reloaded = True
            print(f"[LivePosition] reloaded @ {time.strftime('%H:%M:%S')}")

            if self._on_reload:
                self._on_reload(self._cache)

        return self._cache, reloaded

    

def on_positions_reload(data: dict):
    print(f"[Callback] positions updated: {list(data.keys())}")
    # ✅ 推荐：同步到 PositionManager
    position_mgr.load_from_yaml(data)


live_loader = LivePositionLoader(
    "state/live_positions.yaml",
    on_reload=on_positions_reload
)


def live_positions_hot_load():
    while True:
        data, reloaded = live_loader.load()
        time.sleep(1)
