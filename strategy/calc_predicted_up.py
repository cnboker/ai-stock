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