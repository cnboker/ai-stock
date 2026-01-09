class EquityRegimeManager:
    """
    负责：
    - regime 判断（good / neutral / bad）
    - good 连续确认计数
    """

    def __init__(self, *, good_confirm_need: int = 3):
        self.current: str = "neutral"
        self.good_count: int = 0
        self.good_confirm_need: int = good_confirm_need

    def update(self, *, gate: float, score: float, dd: float) -> str:
        """
        返回当前 regime
        """

        # ===== 1️⃣ bad 判定（最高优先级）=====
        if dd < -0.05:   # 示例：回撤 >5%
            self._set_bad()
            return self.current

        # ===== 2️⃣ good 判定 =====
        if score >= gate:
            self.good_count += 1
            if self.good_count >= self.good_confirm_need:
                self.current = "good"
            else:
                self.current = "neutral"
            return self.current

        # ===== 3️⃣ neutral =====
        self.good_count = 0
        self.current = "neutral"
        return self.current

    # -------------------------
    # 内部工具
    # -------------------------
    def _set_bad(self):
        self.current = "bad"
        self.good_count = 0
