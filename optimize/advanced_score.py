# 逻辑修改要点解析：
# 移除 Win_Rate 相关逻辑：删除了原有的胜率奖惩阶梯，避免了因缺少参数导致的 KeyError。

# 强化 Strategy_Return 权重：在第一部分将 ret 的乘数从 2.0 提到了 5.0。这能解决你之前提到的“盈利了判分却很低”的问题，让绝对收益占据更主导的地位。

# 保留 Alpha 引导梯度：即便你的 Alpha 目前是 -44%，只要下一组参数变成了 -30%，这个函数返回的分值就会大幅跳升，从而指引 Optuna 往正确的方向进化。

# 增加回撤硬约束：针对富祥药业这种动辄 20cm 的票，在评分里加入了一个 mdd > 0.15 的额外扣分项，强迫系统在追求收益的同时，必须通过你的 “均价步长加仓” 逻辑来压低回撤。

def get_advanced_score(stats, is_test=False):
    trades = stats.get("Trade_Count", 0)
    ret = stats.get("Strategy_Return", -5.0)  
    mdd = abs(stats.get("Max_Drawdown", 0.01)) 
    alpha = stats.get("Alpha", -5.0)           

    # --- 1. 基础收益分 (强化版卡玛比率) ---
    # 增加 ret 的权重，即便 MDD 较大，只要收益覆盖得住，评分依然为正
    # 加入 0.1 防止除以 0
    score = (ret * 5.0) / (mdd + 0.1) 

    # --- 2. 核心竞争力奖励 (Alpha) ---
    # Alpha 是你相对于买入持有的核心指标，维持 15.0 的高权重
    # 这会引导 Optuna 寻找能跑赢 53% 涨幅的参数
    score += alpha * 15.0 

    # --- 3. 活跃度与质量阶梯 ---
    # 活跃度检查：如果交易次数太少，不给予上面的完整评分逻辑
    threshold = 2 if is_test else 5  
    
    if trades < threshold: 
        # 惩罚项：-50 是底分，通过 trades 和 alpha 提供引导梯度
        # 让 Optuna 知道：多成交一次或者 Alpha 稍微变好一点，分值都会涨
        return -50.0 + (trades * 5.0) + (alpha * 2.0)

    # --- 4. 稳定性奖励 (回撤惩罚) ---
    # 如果最大回撤超过 15%，额外扣分，防止通过极高杠杆换取收益
    if mdd > 0.15:
        score -= (mdd - 0.15) * 100.0

    return float(score)