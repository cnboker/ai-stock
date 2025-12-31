def log_equity_decision(ticker, eq_feat, decision):
    if eq_feat is None or eq_feat.empty:
        return

    dd = eq_feat["eq_drawdown"].iloc[-1]
    slope = eq_feat["eq_slope"].iloc[-1]

    print(
        f"[EQUITY] {ticker} "
        f"regime={decision.regime} "
        f"dd={dd:.2%} "
        f"slope={slope:.4f} "
        f"gate={decision.gate_mult:.2f} "
        f"action={decision.action} "
        f"strength={decision.reduce_strength:.2f}"
    )
