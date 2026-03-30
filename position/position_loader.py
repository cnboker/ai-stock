import os
import yaml
import traceback
from global_state import state_lock

class LivePositionLoader:
    def __init__(self, path: str, position_mgr):
        self.path = path
        self.position_mgr = position_mgr
        self._last_mtime = 0

    def sync(self, force=False):
        """
        手动同步方法：在交易循环开始时调用。
        只有当文件时间戳改变时才执行读取，节省 IO。
        """
        try:
            if not os.path.exists(self.path):
                # 如果文件不存在，可能是初次运行，跳过
                return False

            mtime = os.path.getmtime(self.path)
            
            # 只有时间戳变了，才进入临界区执行昂贵的 YAML 解析
            if force or mtime != self._last_mtime:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                # 🔐 加锁更新内存，确保分析循环读取到的是完整数据
                with state_lock:
                    self.position_mgr.load_from_yaml(data)
                
                self._last_mtime = mtime
                # print(f"🔄 [Sync] 已从 {self.path} 同步最新仓位数据")
                return True
                
        except Exception as e:
            print(f"❌ [Sync Error] 同步持仓文件失败: {e}")
            # traceback.print_exc()
        
        return False
    @classmethod
    def load_tickers(cls, file_path):
        tickers = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
              data = yaml.safe_load(f) or {}
              tickers = list(data.get("positions", {}).keys())
              return tickers
        except Exception as e:
            print(f"❌ 加载观察池失败: {e}")
            traceback.print_exc()
        return tickers