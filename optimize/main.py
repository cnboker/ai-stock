import numpy as np
import optuna
import pandas as pd
from simulator.run_backtest import BacktestRunner, run_backtest  # 确保 BacktestRunner 已按之前的逻辑改造
from infra.core.config import settings
from optimize.persist import write_best_config

def objective(trial):
    # --- 1. 让 Optuna 建议参数 (保持你现有的配置) ---
    config = {
        "MODEL_LONG_THRESHOLD": trial.suggest_float("model_th", 0.4, 0.5),
        "TREND_SLOPE_THRESHOLD": trial.suggest_float("slope_th", -0.001, 0.003),
        "PREDICTED_UP": trial.suggest_float("predict_up_th", -0.01, 0.0),
        "INIT_PROFIT_TRIGGER": trial.suggest_float("init_pt", 0.03, 0.08),
        "TREND_STAGE_TRIGGER": trial.suggest_float("trend_pt", 0.05, 0.25),
        "ATR_MULTIPLIER": trial.suggest_float("atr_mult", 1.0, 4.5),
        "TP1_RATIO": trial.suggest_float("tp1", 1.02, 1.05),
        "TP2_RATIO": trial.suggest_float("tp2", 1.1, 1.25),
        "KELLY_FRACTION": trial.suggest_float("kelly", 0.1, 0.7),
        "RISK_PER_TRADE": trial.suggest_float("max_lost_pct", 0.05, 0.1),
        "ATR_STOP_MULT": trial.suggest_float("atr_stop_mult", 2, 2.5),
        "MAX_STOP_PCT": trial.suggest_float("max_stop", 0.03, 0.10),
        "MIN_STOP_PCT": trial.suggest_float("min_stop", 0.01, 0.02),
        "MIN_RR": trial.suggest_float("min_rr", 0.5, 1.0),
        "STRENGTH_ALPHA": trial.suggest_float("strength_alpha", 1.2, 1.5),
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
            train_val = get_advanced_score(train_stats)
            test_val = get_advanced_score(test_stats)
            
            
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
            scores.append(-30.0)

    # 最终汇总
    return np.mean(scores) - 0.5 * np.std(scores)

# 在 objective 函数内部改进评分：
def get_advanced_score(stats):
    """
    改进版评分：提供平滑的梯度引导，防止 Optuna 陷入死区
    """
    trades = stats.get("Trade_Count", 0)
    ret = stats.get("Strategy_Return", -5.0)  
    mdd = abs(stats.get("Max_Drawdown", 0.01)) 
    alpha = stats.get("Alpha", -5.0)           
    win_rate = stats.get("Win_Rate", 0.0)      # 确保获取胜率字段

    # --- 1. 活跃度平滑引导 (代替硬惩罚) ---
    # 不要直接 return -100，而是给一个随交易量增加而改善的惩罚项
    # 目标 20 笔，每少一笔扣 5 分，但允许它继续计算后面的收益分
    trade_penalty = max(0, 20 - trades) * 5.0

    # --- 2. 核心效益：卡玛比率 (Calmar) ---
    # 加上平滑项，防止 MDD 过小时分值爆炸
    score = ret / (mdd + 0.1) 

    # --- 3. 质量奖励 ---
    # 奖励 Alpha（超额收益）
    if alpha > 0:
        score += alpha * 10.0
    
    # 奖励胜率 (如果胜率 > 40%, 线性增加分值)
    if win_rate > 0.40:
        score += (win_rate - 0.40) * 50.0
    elif win_rate < 0.30 and trades > 5:
        # 只有在有一定样本量时才惩罚低胜率
        score -= 20.0

    # --- 4. 最终结算 ---
    final_score = score - trade_penalty
    # 临时修改，先让它找出能成交 5-10 笔的参数
    if trades < 5: 
        return -100 - (5 - trades)
    # 极端保底：如果连一笔交易都没有，必须重罚

    return float(final_score)

if __name__ == "__main__":
    # 创建研究
    study = optuna.create_study(
        study_name="quant_experiment_v3", 
        storage="sqlite:///db.sqlite3", 
        load_if_exists=True,
        direction="maximize" 
    )

    # 注入初始“经验”
    study.enqueue_trial({
        "model_th": 0.42,         # 降低到日志显示的平均水平
        "slope_th": 0.0001,       # 极微弱趋势即可
        "predict_up_th": 0.0,     # 不求大涨，只要不跌
        "init_pt": 0.03,          # 提早开始利润保护
        "trend_pt": 0.08,         # 8%进入趋势跟踪
        "atr_mult": 2.5,          # 稍微收紧止损，保护本金
        "tp1": 1.03,              # 3%就先落袋一部分，提升胜率
        "tp2": 1.10, 
        "kelly": 0.3,             # 降低杠杆，由于模型信心一般，仓位要保守
        "max_pct": 0.1, 
        "max_lost_pct": 0.01, 
        "atr_stop_mult": 1.2,     # 移动止损更贴身
        "max_stop": 0.04,         # 4%强制离场
        "min_stop": 0.01, 
        "min_rr": 0.8,
        "strength_alpha":1.2
    })

    # 开始优化
    study.optimize(objective, n_trials=100)
    
    # 持久化最优配置
    write_best_config(study)
    print("最优参数已保存。")