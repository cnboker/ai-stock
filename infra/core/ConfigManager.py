import json
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = config_dir
        # 预加载全局默认参数（兜底用）
        self.default_config = {
            "STRENGTH_ALPHA": 1.35,
            "MODEL_LONG_THRESHOLD": 0.45,
            "ATR_STOP_MULT": 3.5,
            "RISK_PER_TRADE": 0.01
        }

    def _get_category(self, ticker: str) -> str:
        """识别标的类型 (与之前的 Factory 逻辑一致)"""
        if ticker.startswith(('sz00', 'sz30', 'sh60', 'sh68')):
            return "STOCK"
        elif ticker.startswith(('sh51', 'sz15', 'sh58')):
            return "ETF"
        return "DEFAULT"

    def load_params(self, ticker: str) -> Dict[str, Any]:
        """
        加载参数的优先级逻辑 (从高到低):
        1. {ticker}.json (特定标的最优参数, 如 sz159908.json)
        2. category_{type}.json (分类通用参数, 如 category_ETF.json)
        3. global_default (全局兜底)
        """
        # 1. 尝试加载特定标的配置 (Best Trial from Optuna)
        ticker_path = os.path.join(self.config_dir, f"{ticker}.json")
        if os.path.exists(ticker_path):
            with open(ticker_path, 'r') as f:
                print(f"✅ Loaded ticker-specific config for {ticker}")
                return json.load(f)

        # 2. 尝试加载分类通用配置
        category = self._get_category(ticker)
        cat_path = os.path.join(self.config_dir, f"category_{category}.json")
        if os.path.exists(cat_path):
            with open(cat_path, 'r') as f:
                print(f"ℹ️ Loaded {category} category config for {ticker}")
                return json.load(f)

        # 3. 兜底返回全局配置
        print(f"⚠️ No config found for {ticker}, using global defaults.")
        return self.default_config

# --- 实战用法示例 ---

# 1. 实例化加载器
manager = ConfigManager(config_dir="./optimized_params")

# 2. 假设我们要为创业板 ETF 开启交易
# 它会寻找 sz159908.json -> category_ETF.json -> default
current_params = manager.load_params("sz159908")

# 3. 将参数注入你的 Strategy 类
# strategy = MyStrategy(ticker="sz159908", **current_params)