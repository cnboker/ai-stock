# 逻辑修改要点解析：
# 移除 Win_Rate 相关逻辑：删除了原有的胜率奖惩阶梯，避免了因缺少参数导致的 KeyError。

# 强化 Strategy_Return 权重：在第一部分将 ret 的乘数从 2.0 提到了 5.0。这能解决你之前提到的“盈利了判分却很低”的问题，让绝对收益占据更主导的地位。

# 保留 Alpha 引导梯度：即便你的 Alpha 目前是 -44%，只要下一组参数变成了 -30%，这个函数返回的分值就会大幅跳升，从而指引 Optuna 往正确的方向进化。

# 增加回撤硬约束：针对富祥药业这种动辄 20cm 的票，在评分里加入了一个 mdd > 0.15 的额外扣分项，强迫系统在追求收益的同时，必须通过你的 “均价步长加仓” 逻辑来压低回撤。
def get_advanced_score(stats, is_test=False):
    trades = stats.get("Trade_Count", 0)
    ret = stats.get("Strategy_Return", 0.0)  
    mdd = abs(stats.get("Max_Drawdown", 0.001)) # ETF回撤小，调低默认值
    alpha = stats.get("Alpha", 0.0)           

    # --- 1. 基础收益分 (强化版卡玛比率) ---
    # 针对 ETF 收益率低（如 0.3%）的特点，将系数从 5.0 提至 10.0
    # 这样即使是 0.5% 的小利，在低回撤下也能贡献显著正分
    score = (ret * 10.0) / (mdd + 0.05) 

    # --- 2. 核心竞争力奖励 (Alpha) ---
    # 在 ETF 震荡市中，Alpha 为正（如你之前的 4.73%）说明避险极佳
    # 维持 15.0 权重，这是区分“平庸”和“优秀”参数的关键
    score += alpha * 15.0 

    # --- 3. 活跃度与质量阶梯 ---
    # 验证集只要 1 次交易，训练集只要 5 次交易
    threshold = 1 if is_test else 5  
    
    if trades < threshold: 
        # 🚨 引导梯度优化：增加 alpha 的引导权重
        # 即使没成交，如果模型预判的方向（Alpha）在变好，评分也该上升
        return -50.0 + (trades * 10.0) + (alpha * 5.0)

    # --- 4. 稳定性奖励 (ETF 回撤惩罚) ---
    # 对于 ETF，15% 的回撤太宽松了，建议改为 8%
    # 如果 MDD 超过 8%，说明杠杆（Kelly）加得太离谱了，必须重罚
    if mdd > 0.08:
        score -= (mdd - 0.08) * 150.0

    # --- 5. 绝对收益奖励 (额外保护) ---
    # 如果 Strategy_Return 为正，额外给一个“生存奖”
    if ret > 0:
        score += 5.0

    return float(score)