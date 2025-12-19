class SignalDebouncer:
    def __init__(self, confirm_n=3):
        self.confirm_n = confirm_n
        self.last_signal = None
        self.count = 0
        self.last_confirmed_signal = "HOLD"

    def update(self, raw_signal: str) -> str:
        if raw_signal == self.last_signal:
            self.count += 1
        else:
            self.last_signal = raw_signal
            self.count = 1

        if self.count >= self.confirm_n:
            if raw_signal != self.last_confirmed_signal:
                self.last_confirmed_signal = raw_signal
                return raw_signal

        return "HOLD"

class DebouncerManager:
    def __init__(self, confirm_n=3):
        self.confirm_n = confirm_n
        self.debouncers = {}

    def update(self, ticker, raw_signal):
        if ticker not in self.debouncers:
            self.debouncers[ticker] = SignalDebouncer(self.confirm_n)
        return self.debouncers[ticker].update(raw_signal)


# 模块级单例
debouncer_manager = DebouncerManager(confirm_n=3)



