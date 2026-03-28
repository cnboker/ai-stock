import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_dir: str = "./config/optimized_params"):
        self.config_dir = config_dir
        # 预加载全局默认参数（兜底用）
        self.default_config = {
            "STRENGTH_ALPHA": 1.35,
            "MODEL_LONG_THRESHOLD": 0.45,
            "ATR_STOP_MULT": 3.5,
            "RISK_PER_TRADE": 0.01
        }
        # 缓存结构: { file_path: {"mtime": 12345, "data": {...}} }
        self._config_cache = {}

    def _get_cached_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """内部工具：带 mtime 校验的缓存读取"""
        try:
            if not os.path.exists(file_path):
                return None
            
            current_mtime = os.path.getmtime(file_path)
            cached_item = self._config_cache.get(file_path)

            # 如果缓存不存在，或者文件被修改过，则重新加载
            if not cached_item or current_mtime != cached_item["mtime"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._config_cache[file_path] = {
                        "mtime": current_mtime,
                        "data": data
                    }
                # print(f"💾 [Disk -> Cache] {os.path.basename(file_path)}")
                return data
            
            # 命中缓存
            return cached_item["data"]
        except Exception as e:
            print(f"❌ Error loading config {file_path}: {e}")
            return None
        
    def _get_category(self, ticker: str) -> str:
        # 保持你原有的分类逻辑
        if ticker.startswith(('sz15', 'sh51')): return "ETF"
        return "STOCK"
    
    def load_params(self, ticker: str) -> Dict[str, Any]:
        """
        加载参数的优先级逻辑 (从内存缓存读取):
        """
        # 1. 尝试加载特定标的配置 (sz159908.json)
        ticker_path = os.path.join(self.config_dir, f"{ticker}.json")
        data = self._get_cached_json(ticker_path)
        if data:
            # print(f"🎯 Use Ticker-Specific: {ticker}")
            return data

        # 2. 尝试加载分类通用配置 (category_ETF.json)
        category = self._get_category(ticker)
        cat_path = os.path.join(self.config_dir, f"category_{category}.json")
        data = self._get_cached_json(cat_path)
        if data:
            # print(f"📂 Use Category Config: {category}")
            return data

        # 3. 兜底返回全局配置
        return self.default_config

# 1. 实例化加载器
dynamic_config_manager = ConfigManager()
