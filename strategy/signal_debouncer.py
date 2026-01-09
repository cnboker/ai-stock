# ===================== 配置表：score → action =====================
# SCORE_ACTION_MAP = [
#     {"min_score": 0.3, "action": "LONG"},
#     {"min_score": 0.05, "action": "HOLD"},
#     {"min_score": -0.05, "action": "HOLD"},
#     {"min_score": -0.3, "action": "REDUCE"},
#     {"min_score": -1.0, "action": "SHORT"},
# ]
'''
    💡 实战参考：

    策略类型	model_score 阈值	说明
    高频日内	0.01~0.02	小波动可进场
    中短线跟随	0.03~0.05	过滤噪音
    趋势跟随 / 套利	0.05~0.1	确保信号强，滑点影响低
'''

from dataclasses import dataclass

@dataclass
class DebounceState:
    bucket: str
    count: int
    confirm_n: int
    confirmed: bool

SCORE_ACTION_MAP = [
    {"min_score": 0.01, "action": "LONG"},  # 原来 0.3 改成 0.01
    {"min_score": 0.0, "action": "HOLD"},
    {"min_score": -0.01, "action": "HOLD"},
    {"min_score": -0.3, "action": "REDUCE"},
    {"min_score": -1.0, "action": "SHORT"},
]


def score_to_action(score: float) -> str:
    for entry in SCORE_ACTION_MAP:
        if score >= entry["min_score"]:
            return entry["action"]
    return "SHORT"


class SignalDebouncer:
    def __init__(self, base_confirm_n=3):
        self.base_confirm_n = base_confirm_n
        self.last_bucket = None
        self.count = 0
        self.last_confirmed_bucket = None
        self.state = None   # ⭐ 新增

    def update(self, score: float, atr: float = 1.0):
        bucket = score_to_action(score)
        confirm_n = max(1, int(self.base_confirm_n * (atr / 1.0)))

        confirmed = False

        if bucket == "REDUCE":
            self.last_confirmed_bucket = "REDUCE"
            confirmed = True
            action = "REDUCE"
            confidence = abs(score)

        else:
            if bucket == self.last_bucket:
                self.count += 1
            else:
                self.last_bucket = bucket
                self.count = 1

            if self.count >= confirm_n and bucket != self.last_confirmed_bucket:
                self.last_confirmed_bucket = bucket
                confirmed = True
                action = bucket
                confidence = abs(score)
            else:
                action = "HOLD"
                confidence = 0.0

        # ⭐ 关键：记录状态
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

    def update(self, ticker, final_score, atr: float):
        if ticker not in self.debouncers:
            self.debouncers[ticker] = SignalDebouncer(self.confirm_n)

        action, confidence = self.debouncers[ticker].update(final_score, atr)
        state = self.debouncers[ticker].state
        return action, confidence, state



# 模块级单例
debouncer_manager = DebouncerManager(confirm_n=3)
