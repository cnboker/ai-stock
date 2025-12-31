from collections import defaultdict
from datetime import datetime, timedelta

class EquityRegimeCooldown:
    def __init__(
        self,
        bad_hold_sec=300,     # bad 至少维持 5 分钟
        good_confirm=3,       # 连续 good N 次才升级
    ):
        self.state = defaultdict(lambda: {
            "regime": "neutral",
            "last_change": datetime.min,
            "good_count": 0,
        })
        self.bad_hold = timedelta(seconds=bad_hold_sec)
        self.good_confirm = good_confirm

    def update(self, ticker: str, new_regime: str) -> str:
        s = self.state[ticker]
        now = datetime.now()

        # ===== bad：立即进入，延迟退出 =====
        if new_regime == "bad":
            s["regime"] = "bad"
            s["last_change"] = now
            s["good_count"] = 0
            return "bad"

        # bad 冷却中，不允许跳出
        if s["regime"] == "bad":
            if now - s["last_change"] < self.bad_hold:
                return "bad"
            # 冷却结束，允许进入 neutral
            s["regime"] = "neutral"

        # ===== good：需要连续确认 =====
        if new_regime == "good":
            s["good_count"] += 1
            if s["good_count"] >= self.good_confirm:
                s["regime"] = "good"
        else:
            s["good_count"] = 0
            s["regime"] = "neutral"

        return s["regime"]


regime_cooldown = EquityRegimeCooldown()