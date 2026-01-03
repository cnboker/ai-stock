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

    def update(self, score: float, atr: float = 1.0):
        """
        动态 confirm_n 随波动率调整
        atr: 当前波动率指标
        """
        bucket = score_to_action(score)

        # 动态确认次数：波动越大，确认越慢
        confirm_n = max(1, int(self.base_confirm_n * (atr / 1.0)))

        # REDUCE / SHORT / LONG 处理
        if bucket == "REDUCE":
            self.last_confirmed_bucket = "REDUCE"
            return "REDUCE", abs(score)  # confidence 用 score 强度

        if bucket == self.last_bucket:
            self.count += 1
        else:
            self.last_bucket = bucket
            self.count = 1

        if self.count >= confirm_n:
            if bucket != self.last_confirmed_bucket:
                self.last_confirmed_bucket = bucket
                return bucket, abs(score)

        return "HOLD", 0.0


class DebouncerManager:
    def __init__(self, confirm_n=3):
        self.confirm_n = confirm_n
        self.debouncers = {}

    def update(self, ticker, final_score, atr: float):
        if ticker not in self.debouncers:
            self.debouncers[ticker] = SignalDebouncer(self.confirm_n)
        final_action, confidence = self.debouncers[ticker].update(final_score, atr=atr)
        return final_action, confidence


# 模块级单例
debouncer_manager = DebouncerManager(confirm_n=3)
