import os
import threading
import time
from typing import Callable, Any, Optional

# 全系统唯一锁
state_lock = threading.Lock()

class HotReloader:
    def __init__(self, path: str, reload_func: Callable[[str], Any], interval: float = 1.0):
        """
        :param path: 要监控的文件路径
        :param reload_func: 文件变动时执行的具体函数（接收路径作为参数）
        :param interval: 检查间隔
        """
        self.path = path
        self.reload_func = reload_func
        self.interval = interval
        
        self._last_mtime = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None: return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # print(f"🔥 热加载已启动: {self.path}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None

    def _run(self):
        while not self._stop_event.is_set():
            try:
                if os.path.exists(self.path):
                    mtime = os.path.getmtime(self.path)
                    if mtime != self._last_mtime:
                        # 🔐 触发业务层定义的加载逻辑
                        # 注意：锁的粒度由业务逻辑控制，或者在这里统一加锁
                        with state_lock:
                            self.reload_func(self.path)
                        self._last_mtime = mtime
            except Exception as e:
                print(f"⚠️ [HotReloader] 监控 {self.path} 出错: {e}")
                
            time.sleep(self.interval)