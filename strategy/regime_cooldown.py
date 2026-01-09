from collections import defaultdict
from datetime import datetime, timedelta

class EquityRegimeCooldown:
    def __init__(self, bad_hold_sec=300, good_confirm=3):
        self.state = {
            "regime": "neutral",
            "last_change": datetime.min,
            "good_count": 0,
        }
        self.bad_hold = timedelta(seconds=bad_hold_sec)
        self.good_confirm = good_confirm

    # =========================
    # 核心更新（原封不动）
    # =========================
    def update(self, new_regime: str) -> str:
        s = self.state
        now = datetime.now()

        if new_regime == "bad":
            s["regime"] = "bad"
            s["last_change"] = now
            s["good_count"] = 0
            return "bad"

        if s["regime"] == "bad":
            if now - s["last_change"] < self.bad_hold:
                return "bad"
            s["regime"] = "neutral"
            s["good_count"] = 0
            return "neutral"

        if new_regime == "good":
            s["good_count"] += 1
            if s["good_count"] >= self.good_confirm:
                s["regime"] = "good"
        else:
            s["good_count"] = 0
            if s["regime"] != "good":
                s["regime"] = "neutral"

        return s["regime"]

    # =========================
    # 新增：显性接口
    # =========================
    @property
    def regime(self) -> str:
        return self.state["regime"]

    @property
    def good_count(self) -> int:
        return self.state["good_count"]

    def bad_left_sec(self, now: datetime | None = None) -> float:
        if self.state["regime"] != "bad":
            return 0.0
        now = now or datetime.now()
        elapsed = now - self.state["last_change"]
        return max(0.0, self.bad_hold.total_seconds() - elapsed.total_seconds())


regime_cooldown = EquityRegimeCooldown()