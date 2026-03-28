import traceback

import numpy as np
import optuna
import pandas as pd
from simulator.run_backtest import BacktestRunner, run_backtest  # 确保 BacktestRunner 已按之前的逻辑改造
from infra.core.dynamic_settings import settings
from optimize.persist import write_best_config

def objective(trial):
    # --- 1. 让 Optuna 建议参数 (保持你现有的配置) ---
    config = {
        "MODEL_LONG_THRESHOLD": trial.suggest_float("model_th", 0.42, 0.48),
        "TREND_SLOPE_THRESHOLD": trial.suggest_float("slope_th", -0.001, 0.005),
        "PREDICTED_UP": trial.suggest_float("predict_up_th", 0.0, 0.005),
        "INIT_PROFIT_TRIGGER": trial.suggest_float("init_pt", 0.02, 0.04),
        "TREND_STAGE_TRIGGER": trial.suggest_float("trend_pt", 0.05, 0.25),
        "ATR_MULTIPLIER": trial.suggest_float("atr_mult", 2.5, 4.5),
        "TP1_RATIO": trial.suggest_float("tp1", 1.03, 1.06),
        "TP2_RATIO": trial.suggest_float("tp2", 1.1, 1.25),
        "KELLY_FRACTION": trial.suggest_float("kelly", 0.3, 0.5),
        "RISK_PER_TRADE": trial.suggest_float("max_lost_pct", 0.01, 0.015),
        "ATR_STOP_MULT": trial.suggest_float("atr_stop_mult", 2.5, 3.5),
        "MAX_STOP_PCT": trial.suggest_float("max_stop", 0.03, 0.10),
        "MIN_STOP_PCT": trial.suggest_float("min_stop", 0.01, 0.02),
        "MIN_RR": trial.suggest_float("min_rr", 0.0, 0.3),
        "STRENGTH_ALPHA": trial.suggest_float("strength_alpha", 1.2, 1.5),
        "CONFIRM_WINDOW": trial.suggest_float("confirm_n", 2, 5),
    }
    
    # 动态赋值给全局单例 settings
    for key, value in config.items():
        setattr(settings, key, value)

    # --- 2. 这里的多只股票是“泛化”的基础 ---
    # 建议至少包含一只宽基指数（如 sz159908）和 1-2 只个股
    tickers = ["sz159908", "sh510300"] # 建议双标的，更具普适性
    scores = []

    for ticker in tickers:
        try:
            # 2/8 验证运行
            train_stats, test_stats = run_backtest(ticker)

            # 2. 【核心调用】针对训练集和测试集分别评分
            # 训练集占 40%，测试集占 60% (侧重未来表现)
            train_val = max(-200, get_advanced_score(train_stats)) 
            test_val = max(-200, get_advanced_score(test_stats))
            
            
          # 方案 A：线性惩罚，给算法留点改进的余地
            test_return = test_stats.get("Strategy_Return", 0)
            if test_return < -0.05:
                # 比如每多亏 1%，额外扣 5 分，而不是直接定死 -15
                penalty = (abs(test_return) - 0.05) * 100 
                combined_ticker_score = (train_val * 0.7) + (test_val * 0.3) - penalty
            else:
                combined_ticker_score = (train_val * 0.7) + (test_val * 0.3)
            scores.append(combined_ticker_score)

        except Exception as e:
            print(f"Error on {ticker}: {e}")
            traceback.print_exc()
            scores.append(-30.0)

    # 最终汇总
    return np.mean(scores) - 0.5 * np.std(scores)

# 在 objective 函数内部改进评分：
def get_advanced_score(stats):
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
    if trades < 5: 
        # 这个梯度的存在是为了引导它通过阈值线
        # 加上 alpha * 2 是为了让它在同样没成交够时，优先选那个预测更准的
        return -50.0 + (trades * 5.0) + (alpha * 2.0)

    return float(score)

if __name__ == "__main__":
    # 1. 使用全新的 study_name，确保不读取旧的 1.2 止损参数
    study = optuna.create_study(
        study_name="quant_experiment_v1", # 换个名字
        storage="sqlite:///db.sqlite3", 
        load_if_exists=False, # 建议先设为 False，跑一轮干净的
        direction="maximize" 
    )

    # 2. 注入一个符合我们“新逻辑”的初始经验
    # 如果你想引导 Optuna，请给它真正正确的方向：
    study.enqueue_trial({
        "model_th": 0.45,
        "slope_th": -0.001,
        "predict_up_th": 0.0,
        "init_pt": 0.05,
        "trend_pt": 0.15,
        "atr_mult": 3.0,
        "tp1": 1.05,
        "tp2": 1.20, 
        "kelly": 0.5,
        "max_lost_pct": 0.015,
        "atr_stop_mult": 3.5, # <--- 这里的初始值必须设大！
        "max_stop": 0.08,
        "min_stop": 0.01, 
        "min_rr": 0.1,         # 降低盈亏比门槛方便入场
        "strength_alpha": 1.3,
        "confirm_n": 3
    })

    # 3. 开始优化
    study.optimize(objective, n_trials=200)