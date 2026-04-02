    # 在 objective 函数内部改进评分：
def get_advanced_score(stats,is_test=False):
    trades = stats.get("Trade_Count", 0)
    ret = stats.get("Strategy_Return", -5.0)  
    mdd = abs(stats.get("Max_Drawdown", 0.01)) 
    alpha = stats.get("Alpha", -5.0)           
    win_rate = stats.get("Win_Rate", 0.0)

    # --- 1. 基础收益分 (卡玛比率) ---
    # 无论正负都计算，给 Optuna 提供连续的梯度
    score = (ret * 2.0) / (mdd + 0.1) 

    # --- 2. 核心竞争力奖励 (Alpha) ---
    # 只要 Alpha 在变大，分值就应该涨，即便是从 -5 变成 -2
    score += alpha * 15.0 

    # --- 3. 质量奖励 (胜率) ---
    if trades > 5:
        if win_rate > 0.45:
            score += (win_rate - 0.45) * 100.0 # 优秀的胜率给重赏
        elif win_rate < 0.30:
            score -= 30.0 # 低胜率给严惩

    # --- 4. 活跃度阶梯 ---
    # 只有交易次数够了，才执行上面的最终评分
    threshold = 2 if is_test else 5  # 验证集只要有 2 次交易就不惩罚
    if trades < threshold: 
        # 这个梯度的存在是为了引
        # 导它通过阈值线
        # 加上 alpha * 2 是为了让它在同样没成交够时，优先选那个预测更准的
        return -50.0 + (trades * 5.0) + (alpha * 2.0)

    return float(score)