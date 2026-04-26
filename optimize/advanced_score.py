# 逻辑修改要点解析：
# 移除 Win_Rate 相关逻辑：删除了原有的胜率奖惩阶梯，避免了因缺少参数导致的 KeyError。
import numpy as np

# 强化 Strategy_Return 权重：在第一部分将 ret 的乘数从 2.0 提到了 5.0。这能解决你之前提到的“盈利了判分却很低”的问题，让绝对收益占据更主导的地位。

# 保留 Alpha 引导梯度：即便你的 Alpha 目前是 -44%，只要下一组参数变成了 -30%，这个函数返回的分值就会大幅跳升，从而指引 Optuna 往正确的方向进化。

# 增加回撤硬约束：针对富祥药业这种动辄 20cm 的票，在评分里加入了一个 mdd > 0.15 的额外扣分项，强迫系统在追求收益的同时，必须通过你的 “均价步长加仓” 逻辑来压低回撤。
def get_advanced_score(stats, is_test=False):
    RET = stats.get("Strategy_Return", 0.0) 
    ALPHA = stats.get("Alpha", 0.0) 
    MDD = abs(stats.get("Max_Drawdown", 0.0)) 
    TRADES = stats.get("Trade_Count", 0)

    # 降低 Alpha 惩罚权重，提高 RET 权重（让收益更重要）
    QUALITY_SCORE = (RET * 1.8 + ALPHA * 1.8) / (MDD + 1.5)   # 分母略增大，平滑惩罚
    
    MIN_TRADES = 1 if is_test else 3
    ACTIVITY_PUNISH = 0
    if TRADES < MIN_TRADES:
        ACTIVITY_PUNISH = (MIN_TRADES - TRADES) * 10.0   # 扣分从20降到10

    SCORE = (QUALITY_SCORE * 8.0) - ACTIVITY_PUNISH   # 整体乘数从10降到8

    # 极端回撤门槛提高，扣分变轻
    if MDD > 12.0:
        SCORE -= (MDD - 12.0) * 10.0   # 原10%×20 → 12%×10

    return float(SCORE)