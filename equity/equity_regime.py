#Equity Regime（好 / 中 / 坏状态）
'''
Docstring for equity.equity_regime
📌 用途

bad → 熔断 / 禁开新仓

neutral → 正常

good → 放大仓位 / 放宽止盈

EquityRecorder 是一个 账户级别的回测/实盘 equity 曲线记录器

dd = 当前总资产回撤

slope = 总资产曲线的趋势斜率

'''

#判断状态,Equity Regime（资金状态）

def equity_regime(eq_feat):
    if eq_feat is None or eq_feat.empty:
        return "neutral"
    """
    返回: good / neutral / bad
    """
    dd = eq_feat["eq_drawdown"].iloc[-1]
    slope = eq_feat["eq_slope"].iloc[-1]

    if dd < -0.06:
        return "bad"
    if slope > 0 and dd > -0.02:
        return "good"
    return "neutral"
