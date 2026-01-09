from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from collections import deque
from datetime import datetime


class DecisionDebugger:
    def __init__(
        self,
        max_rows: int = 300,
        refresh_hz: int = 4,
        watch_tickers: set[str] | None = None,
    ):
        self.console = Console()
        self.rows = deque(maxlen=max_rows)
        self.live = Live(console=self.console, refresh_per_second=refresh_hz)
        self.started = False
        self.watch_tickers = watch_tickers

    # ===== public =====
    def start(self):
        if not self.started:
            self.live.start()
            self.started = True

    def stop(self):
        if self.started:
            self.live.stop()
            self.started = False

    def update(self, ctx):
        if self.watch_tickers and ctx.ticker not in self.watch_tickers:
            return  # 🚫 直接忽略
        if not self.started:
            self.start()

        self.rows.append(self._ctx_to_row(ctx))
        self.live.update(self._build_table())

    # ===== styling =====
    def _style_regime(self, v):
        return {
            "good": "green",
            "neutral": "yellow",
            "bad": "red",
        }.get(v, "white")

    def _style_gate(self, allow):
        return "green" if allow else "red"

    # ===== transform =====
    def _ctx_to_row(self, ctx):
        return {
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": ctx.ticker,
            "price": f"{ctx.latest_price:.2f}",
            "atr": f"{ctx.atr:.4f}",
            "model": f"{ctx.model_score:+.4f}",
            "pred_up": f"{ctx.predicted_up:+.4f}",
            "gate": Text(
                "Y" if ctx.gate_allow else "N", self._style_gate(ctx.gate_allow)
            ),
            "mult": f"{ctx.gate_mult:.2f}",
            "slope": f"{ctx.slope:.1f}",
            "str": f"{ctx.strength:.3f}",
            "regime": Text(ctx.regime, self._style_regime(ctx.regime)),
            "good": str(ctx.good_count),
            "need": str(ctx.good_confirm_need),
            "cd": f"{ctx.regime_cooldown_left}",
            "dd": f"{ctx.dd:.2%}",
            "pos": "Y" if ctx.has_position else "-",
            "size": f"{ctx.position_size:.2f}" if ctx.has_position else "-",
            "add": "Y" if ctx.allow_add else "-",
            "raw_sig": ctx.raw_signal,
            "raw_score": f"{ctx.raw_score:+.4f}",
        }

    # ===== render =====
    def _build_table(self) -> Table:
        table = Table(title="DecisionContext Timeline (Full)", expand=True)

        cols = [
            "Time",
            "Ticker",
            "Price",
            "ATR",
            "Model",
            "PredUp",
            "Gate",
            "Mult",
            "Slope",
            "Strength",
            "Regime",
            "Good",
            "Need",
            "CD",
            "DD",
            "Pos",
            "Size",
            "Add",
            "RawSig",
            "RawScore",
        ]

        for c in cols:
            table.add_column(c, no_wrap=True)

        for r in self.rows:
            table.add_row(
                r["time"],
                r["ticker"],
                r["price"],
                r["atr"],
                r["model"],
                r["pred_up"],
                r["gate"],
                r["mult"],
                r["slope"],
                r["str"],
                r["regime"],
                r["good"],
                r["need"],
                r["cd"],
                r["dd"],
                r["pos"],
                r["size"],
                r["add"],
                r["raw_sig"],
                r["raw_score"],
            )

        return table
