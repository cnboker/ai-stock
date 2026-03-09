from dataclasses import dataclass


# ===================== 配置表：score → action =====================

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


# ===================== 状态数据结构 =====================

@dataclass
class DebounceState:
    bucket: str
    count: int
    confirm_n: int
    confirmed: bool


# ===================== 单个标的 Debouncer =====================

class SignalDebouncer:

    def __init__(self, base_confirm_n=3):

        self.base_confirm_n = base_confirm_n

        self.last_bucket = None
        self.count = 0

        self.last_confirmed_bucket = None
        self.state = None

    def update(self, score: float, atr: float = 1.0):

        bucket = score_to_action(score)
        confirm_n = self.base_confirm_n

        confirmed = False
        action = "HOLD"
        confidence = 0.0

        # ================= SHORT 立即执行 =================
        if bucket == "SHORT":

            if self.last_confirmed_bucket != "SHORT":

                self.last_confirmed_bucket = "SHORT"
                self.last_bucket = "SHORT"
                self.count = 0

                action = "SHORT"
                confidence = abs(score)
                confirmed = True

        # ================= REDUCE 只触发一次 =================
        elif bucket == "REDUCE":

            if self.last_confirmed_bucket != "REDUCE":

                self.last_confirmed_bucket = "REDUCE"

                action = "REDUCE"
                confidence = abs(score)
                confirmed = True

        # ================= HOLD 冻结状态 =================
        elif bucket == "HOLD":

            action = "HOLD"
            confidence = 0.0

        # ================= LONG 需要 debounce =================
        elif bucket == "LONG":

            if bucket == self.last_bucket:
                self.count += 1
            else:
                self.last_bucket = bucket
                self.count = 1

            if (
                self.count >= confirm_n
                #and bucket != self.last_confirmed_bucket
            ):

                self.last_confirmed_bucket = bucket

                action = "LONG"
                confidence = abs(score)
                confirmed = True

        # ================= 状态记录 =================

        self.state = DebounceState(
            bucket=bucket,
            count=self.count,
            confirm_n=confirm_n,
            confirmed=confirmed,
        )

        return action, confidence


# ===================== 多标的 Manager =====================

class DebouncerManager:

    def __init__(self, confirm_n=3):

        self.confirm_n = confirm_n
        self.debouncers = {}

    def update(self, ticker, final_score, atr: float):

        if ticker not in self.debouncers:
            self.debouncers[ticker] = SignalDebouncer(self.confirm_n)

        action, confidence = self.debouncers[ticker].update(final_score, atr)
        state = self.debouncers[ticker].state

        last_confirmed = self.debouncers[ticker].last_confirmed_bucket

        # print(
        #     f"[DEBOUNCE] "
        #     f"score={final_score:.4f} "
        #     f"bucket={state.bucket} "
        #     f"count={state.count}/{state.confirm_n} "
        #     f"confirmed={state.confirmed} "
        #     f"last_confirmed={last_confirmed} "
        #     f"action={action}"
        # )

        return action, confidence, state


# ===================== 模块级单例 =====================

debouncer_manager = DebouncerManager(confirm_n=3)