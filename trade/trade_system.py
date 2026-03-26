from infra.core.context import TradingSession
from infra.persistence.live_positions import persist_live_positions
from trade.equity_executor import execute_equity_action


class TradingSystem:
    def __init__(self, ctx_builder, signal_mgr, budget_mgr, risk_mgr, position_mgr):
        self.ctx_builder = ctx_builder
        self.signal_mgr = signal_mgr
        self.budget_mgr = budget_mgr
        self.risk_mgr = risk_mgr
        self.position_mgr = position_mgr

    def run_tick(self, ticker, pre_result, close_df, session: TradingSession):
        """每只股票、每根K线的核心处理流程"""
        # 1. 构建标准上下文 (聚合所有原始数据与账户状态)
        ctx = self.ctx_builder.build(
            ticker=ticker,
            low=pre_result.low,
            median=pre_result.median,
            high=pre_result.high,
            latest_price=pre_result.price,
            atr=pre_result.atr,
            model_score=pre_result.model_score,
            close_df=close_df,
            eq_decision=session.tradeIntent,  # 账户层风险意图
        )

        # 2. 评估意图 (判断：进场/离场/强制减仓/观望)
        # 这一步会自动处理 Debounce 和 账户风险(REDUCE) 的合并
        intent = self.signal_mgr.evaluate(ctx)

        # 3. 拦截未确认信号 (快捷路径)
        if not intent.confirmed and not intent.force_reduce:
            return {"ticker": ticker, "action": "HOLD", "reason": "Unconfirmed"}

        # 4. 针对买入信号(LONG)进行资金规划
        plan = None
        if intent.action == "LONG" and not ctx.has_position:
            # A. 算钱 (考虑了回撤保护、起步价和现金限制)
            budget = self.budget_mgr.get_budget(
                ticker=ticker,
                gate_score=intent.gate_mult,
                available_cash=self.position_mgr.available_cash,
                equity=self.position_mgr.equity,
                positions_value=self.position_mgr.position_value(),
            )
            # B. 算股数 (考虑了止损距离、盈亏比和 A股一手限制)
            plan = self.risk_mgr.evaluate(
                ticker=ticker,
                last_price=ctx.latest_price,
                chronos_low=float(pre_result.low[-1]),
                chronos_high=float(pre_result.high[-1]),
                atr=ctx.atr,
                capital=budget,
                position_mgr=self.position_mgr,
            )

        # 5. 最终物理执行 (修改仓位)
        result = execute_equity_action(
            decision=intent,
            position_mgr=self.position_mgr,
            ticker=ticker,
            last_price=ctx.latest_price,
            plan=plan,
        )

        persist_live_positions(self.position_mgr)
        return result
