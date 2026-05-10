import numpy as np

"""
risk_adj_score分数情况含义
高正分预测会上涨，且下方支撑位很近（亏损空间小），且模型非常有把握（区间窄）。—— 优质信号
中低分预测会上涨，但下方跌幅空间很大，或者模型自己也看不太准。—— 需谨慎
负分预测会下跌，或者下行风险远大于预期收益。—— 排除屏蔽
"""
def calc(low, median, high, latest_price):
    if latest_price <= 0 or len(low) == 0:
        return 0.0

    pred = median[-1]
    low_pred = low[-1]
    high_pred = high[-1]

    # 1. 计算预期收益（基于中位数）
    up = (pred - latest_price) / latest_price

    # 2. 计算下行空间（风险）与上行空间（机会）
    # 使用 Kronos 提供的非对称边界
    downside_dist = max(0.0, latest_price - low_pred)
    upside_dist = max(0.0, high_pred - latest_price)
    
    # 3. 计算“不确定性宽度” (替代 std 的功能)
    # 宽度越大，说明模型越没信心，增加惩罚
    uncertainty = (high_pred - low_pred) / latest_price

    # 4. 综合评分逻辑
    # 逻辑：偏好 (预期收益 > 0) 且 (预期收益 > 下行空间) 且 (整体不确定性较小)
    risk_adj_score = up - 0.5 * (downside_dist / latest_price) - 0.1 * uncertainty

    return float(risk_adj_score)

def calc_live_signal(low, median, high, latest_price, 
                     pos_weight=0.40, 
                     unc_weight=0.25, 
                     risk_weight=0.35):
    """
    计算实时交易信号得分（越高越倾向买入）
    
    返回值范围建议控制在 [-1, 1] 左右，便于后续决策
    """
    if latest_price <= 0:
        return 0.0
    
    # 统一转为 numpy array 并检查是否为空
    low = np.asarray(low)
    median = np.asarray(median)
    high = np.asarray(high)
    
    if len(low) == 0 or len(median) == 0 or len(high) == 0:
        return 0.0
    
    p_low = low[-1]
    p_med = median[-1]
    p_high = high[-1]
    
    # 防止除零
    if p_high <= p_low or latest_price <= 0:
        return 0.0
    
    # 1. 相对位置 (0=最便宜, 1=最贵)
    position = (latest_price - p_low) / (p_high - p_low)
    position = max(0.0, min(1.0, position))          # 截断到 [0,1]
    
    # 2. 预期收益
    expected_return = (p_med - latest_price) / latest_price
    
    # 3. 不确定性（区间宽度相对价格）
    uncertainty = (p_high - p_low) / latest_price
    
    # 4. 风险惩罚（重点加强破位惩罚）
    if latest_price < p_low:
        # 破支撑：非线性强惩罚
        downside = (p_low - latest_price) / latest_price
        risk_penalty = downside * 1.5 + (downside ** 1.5) * 2.0
    else:
        # 在区间内时，离支撑越远风险越低（可选）
        risk_penalty = max(0.0, (latest_price - p_low) / latest_price) * 0.3
    
    # 综合得分
    score = (expected_return 
             - pos_weight * position 
             - unc_weight * uncertainty 
             - risk_weight * risk_penalty)
    
    # 建议做平滑/归一化（可选，但强烈推荐）
    # score = max(-1.0, min(1.0, score * 0.8))   # 根据实际回测调整缩放
    
    return round(float(score), 6)    