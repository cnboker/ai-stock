from dataclasses import dataclass
from typing import Tuple

# ===================== 配置表：score → action =====================
# 建议：Optuna 可以动态调整这些阈值
SCORE_ACTION_MAP = [
    {"min_score": 0.03, "action": "LONG"},
    {"min_score": -0.03, "action": "HOLD"},
    {"min_score": -0.2, "action": "REDUCE"},
    {"min_score": -1.0, "action": "SHORT"},
]

def score_to_action(score: float) -> str:
    for entry in SCORE_ACTION_MAP:
        if score >= entry["min_score"]:
            return entry["action"]
    return "SHORT"

@dataclass
class DebounceState:
    bucket: str
    count: int
    confirm_n: int
    confirmed: bool

class SignalDebouncer:
    def __init__(self, base_confirm_n=3):
        self.base_confirm_n = base_confirm_n
        self.last_bucket = None
        self.count = 0
        self.last_confirmed_bucket = None
        self.state = None

    def update(self, score: float, atr: float = 1.0) -> Tuple[str, float]:
        bucket = score_to_action(score)
        
        # 1. 状态重置检查：如果桶变了（比如从 LONG 跌入 HOLD），清空计数
        if bucket != self.last_bucket:
            self.count = 0
            self.last_bucket = bucket

        confirm_n = self.base_confirm_n
        confirmed = False
        action = "HOLD"
        confidence = 0.0

        # ================= SHORT 立即执行（最高优先级） =================
        if bucket == "SHORT":
            self.count += 1
            # SHORT 通常不防抖，直接切断
            self.last_confirmed_bucket = "SHORT"
            action = "SHORT"
            confidence = abs(score)
            confirmed = True

        # ================= REDUCE 只在切换时触发确认 =================
        elif bucket == "REDUCE":
            self.count += 1
            # 只要进入 REDUCE 区域就确认执行（风险控制优先级高）
            self.last_confirmed_bucket = "REDUCE"
            action = "REDUCE"
            confidence = abs(score)
            confirmed = True

        # ================= HOLD 状态：不确认，重置计数 =================
        elif bucket == "HOLD":
            self.count = 0 # 严格模式：HOLD 期间计数清零
            action = "HOLD"
            confidence = 0.0
            confirmed = False

        # ================= LONG 引入动态加速 =================
        elif bucket == "LONG":
            self.count += 1

            # 动态确认阈值 (Dynamic Confirmation)
            # 针对 Chronos 高分模型显著加速
            current_confirm_n = confirm_n
            if score >= 0.75:      # 极高分：1次确认（即刻）
                current_confirm_n = 1
            elif score >= 0.50:    # 高分：2次确认
                current_confirm_n = 2
            
            # 记录当前使用的阈值供日志显示
            confirm_n = current_confirm_n

            # 判定确认逻辑
            if self.count >= current_confirm_n:
                self.last_confirmed_bucket = "LONG"
                action = "LONG"
                confidence = abs(score)
                confirmed = True # 只要达到计数，持续为 True，驱动 Budget 计算

        # ================= 状态记录 =================
        self.state = DebounceState(
            bucket=bucket,
            count=self.count,
            confirm_n=confirm_n,
            confirmed=confirmed,
        )

        return action, confidence

class DebouncerManager:
    def __init__(self, confirm_n=3):
        self.confirm_n = confirm_n
        self.debouncers = {}

    def update(self, ticker: str, final_score: float, atr: float):
        if ticker not in self.debouncers:
            self.debouncers[ticker] = SignalDebouncer(self.confirm_n)

        action, confidence = self.debouncers[ticker].update(final_score, atr)
        state = self.debouncers[ticker].state
        
        return action, confidence, state

# 单例初始化
debouncer_manager = DebouncerManager(confirm_n=3)