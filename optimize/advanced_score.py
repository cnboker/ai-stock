# 逻辑修改要点解析：
# 移除 Win_Rate 相关逻辑：删除了原有的胜率奖惩阶梯，避免了因缺少参数导致的 KeyError。
import numpy as np

# 强化 Strategy_Return 权重：在第一部分将 ret 的乘数从 2.0 提到了 5.0。这能解决你之前提到的“盈利了判分却很低”的问题，让绝对收益占据更主导的地位。

# 保留 Alpha 引导梯度：即便你的 Alpha 目前是 -44%，只要下一组参数变成了 -30%，这个函数返回的分值就会大幅跳升，从而指引 Optuna 往正确的方向进化。

# 增加回撤硬约束：针对富祥药业这种动辄 20cm 的票，在评分里加入了一个 mdd > 0.15 的额外扣分项，强迫系统在追求收益的同时，必须通过你的 “均价步长加仓” 逻辑来压低回撤。
def get_advanced_score(stats, is_test=False):
    trades = stats.get("Trade_Count", 0)
    ret = stats.get("Strategy_Return", 0.0)
    mdd = abs(stats.get("Max_Drawdown", 1e-6))
    alpha = stats.get("Alpha", 0.0)

    # 计算基础表现分 (核心质量)
    # 哪怕只有1次交易，这个分值也能告诉优化器：这个方向是对的
    quality_score = (ret * 15.0) / (mdd + 0.05) + (alpha * 15.0)

    # 活跃度分
    activity_score = np.log1p(trades) * 10.0

    score = quality_score + activity_score

    # 软性惩罚：与其直接打死，不如根据缺少的交易次数扣分
    threshold = 1 if is_test else 3
    if trades < threshold:
        # 每次交易缺失扣 50 分，但保留 quality_score 的正向引导
        score -= (threshold - trades) * 50.0

    # 极致回撤惩罚 (>8% 视为失控)
    if mdd > 0.08:
        score -= (mdd - 0.08) * 200.0

    return float(score)