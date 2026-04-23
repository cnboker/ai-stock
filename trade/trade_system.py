from infra.core.runtime import GlobalState
from infra.persistence.live_positions import persist_live_positions
from log import signal_log
from strategy.decision_context import DecisionContext
from trade.signal_stablizer import SignalStablizer
from trade.equity_executor import execute_equity_action
from infra.core.dynamic_settings import settings


class TradingSystem:
    def __init__(self, signal_mgr, budget_mgr, risk_mgr, position_mgr):
        self.signal_mgr = signal_mgr
        self.budget_mgr = budget_mgr
        self.risk_mgr = risk_mgr
        self.position_mgr = position_mgr
        self.stablizer = None

    def run_tick(self, ticker, ctx: DecisionContext,low, high):
        """每只股票、每根K线的核心处理流程"""

        # 2. 评估意图 (判断：进场/离场/强制减仓/观望)
        # 这一步会自动处理 Debounce 和 账户风险(REDUCE) 的合并
        intent = self.signal_mgr.evaluate(ctx)

        # 3. 拦截未确认信号 (快捷路径)
        if not intent.confirmed and not intent.force_reduce:
            persist_live_positions(self.position_mgr)
            return {"ticker": ticker, "action": "HOLD", "reason": "Unconfirmed"}

        # 4. 针对买入信号(LONG)进行资金规划
        plan = None
        if self.stablizer is None:
            self.stablizer = SignalStablizer(window=settings.CONFIRM_N)
        if intent.action == "LONG":
            # print(f"intent={intent} settings.CONFIRM_N={settings.CONFIRM_N}")
            # 调用稳定器校验
            is_stable = self.stablizer.check(ticker, "LONG")

            if not is_stable:
                # 记录日志，但不触发下单
                progress = self.stablizer.get_progress(ticker)
                signal_log(f"⏳ [{ticker}] Signal unstable, confirming: {progress}")
                return {
                    "ticker": ticker,
                    "action": "HOLD",
                    "reason": f"Stablizing {progress}",
                }

            self.stablizer.reset(ticker=ticker)
            # 0. 增加【单一标的持仓上限】硬约束 (例如单标的不得超过总资产 30%)
            price = GlobalState.tickers_price[ticker]
            MAX_TICKER_WEIGHT = 0.15
            current_weight = (
                self.position_mgr.get_ticker_value(ticker, price)
                / self.position_mgr.equity
            )

            if current_weight >= MAX_TICKER_WEIGHT:
                # 已经买够了，不再加仓，改为 HOLD
                # signal_log(f"⚠️ [{ticker}] Weight limit reached: {current_weight:.2%}, holding position.")
                return {
                    "ticker": ticker,
                    "action": "HOLD",
                    "reason": f"Weight Limit Reached ({current_weight:.2%})",
                }

            # A. 算钱 (这里需要把剩余额度传进去)
            remaining_budget = max(
                0, (MAX_TICKER_WEIGHT - current_weight) * self.position_mgr.equity
            )

            budget = self.budget_mgr.get_budget(
                ticker=ticker,
                gate_score=intent.gate_mult,
                available_cash=min(
                    self.position_mgr.available_cash, remaining_budget
                ),  # 取交集
                equity=self.position_mgr.equity,
                positions_value=self.position_mgr.position_value(),
            )

            # B. 算股数 (考虑了止损距离、盈亏比和 A股一手限制)
            plan = self.risk_mgr.evaluate(
                ticker=ticker,
                chronos_low=float(low[-1]),
                chronos_high=float(high[-1]),
                atr=ctx.atr,
                capital=budget,
                position_mgr=self.position_mgr,
            )
            print(f"plan={plan}")

        # 5. 最终物理执行 (修改仓位)
        pos_dict = self.position_mgr.pos_to_dict(ticker=ticker)
        print(f"交易前:{pos_dict},intent={intent},plan={plan}")
        result = execute_equity_action(
            decision=intent,
            position_mgr=self.position_mgr,
            ticker=ticker,
            plan=plan,
        )
        print(f"交易后:{result}")
        persist_live_positions(self.position_mgr)
        return result
