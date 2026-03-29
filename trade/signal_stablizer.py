
class SignalStablizer:
    def __init__(self, window: int = 10):
        """
        :param window: 需要连续出现多少次相同信号才放行
        """
        self.window = window
        self.counters = {}  # 格式: {ticker: {"action": str, "count": int}}

    def check(self, ticker: str, current_action: str) -> bool:
        """
        检查信号是否稳定
        :return: True (已稳定，可执行), False (待确认，需拦截)
        """
        # 如果是空信号或重置信号，清空计数并放行（比如强制平仓不应被拦截）
        if current_action in ["HOLD", "LIQUIDATE"]:
            self.counters[ticker] = {"action": current_action, "count": 0}
            return True

        # 获取历史状态
        state = self.counters.get(ticker, {"action": None, "count": 0})

        # 如果信号与上一次一致，计数加1
        if state["action"] == current_action:
            state["count"] += 1
        else:
            # 信号发生切换，重新开始计数
            state = {"action": current_action, "count": 1}

        self.counters[ticker] = state

        # 只有计数达到窗口值时才返回 True
        return state["count"] >= self.window

    def reset(self, ticker: str = None):
        """
        重置计数器
        :param ticker: 如果传 ticker，只重置该标的；如果不传，重置全部。
        """
        if ticker:
            if ticker in self.counters:
                self.counters[ticker] = {"action": None, "count": 0}
        else:
            self.counters = {}

    def get_progress(self, ticker: str) -> str:
        """调试辅助：获取当前确认进度"""
        state = self.counters.get(ticker, {"count": 0})
        return f"{state['count']}/{self.window}"
    
