import numpy as np
import optuna
import pandas as pd
from simulator.run_backtest import BacktestRunner  # 确保 BacktestRunner 已按之前的逻辑改造
from infra.core.config import settings
from optimize.persist import write_best_config

def objective(trial):
    # --- 1. 让 Optuna 建议参数 (保持你现有的配置) ---
    config = {
        "MODEL_LONG_THRESHOLD": trial.suggest_float("model_th", 0.4, 0.65),
        "TREND_SLOPE_THRESHOLD": trial.suggest_float("slope_th", -0.01, 0.01),
        "PREDICTED_UP": trial.suggest_float("predict_up_th", -0.02, 0.01),
        "INIT_PROFIT_TRIGGER": trial.suggest_float("init_pt", 0.03, 0.08),
        "TREND_STAGE_TRIGGER": trial.suggest_float("trend_pt", 0.10, 0.25),
        "ATR_MULTIPLIER": trial.suggest_float("atr_mult", 1.0, 3.5),
        "TP1_RATIO": trial.suggest_float("tp1", 1.05, 1.15),
        "TP2_RATIO": trial.suggest_float("tp2", 1.15, 1.30),
        "KELLY_FRACTION": trial.suggest_float("kelly", 0.1, 0.7),
        "MAX_SIGNAL_PCT": trial.suggest_float("max_pct", 0.05, 0.2),
        "RISK_PER_TRADE": trial.suggest_float("max_lost_pct", 0.01, 0.05),
        "ATR_STOP_MULT": trial.suggest_float("atr_stop_mult", 0.5, 2.0),
        "MAX_STOP_PCT": trial.suggest_float("max_stop", 0.03, 0.10),
        "MIN_STOP_PCT": trial.suggest_float("min_stop", 0.01, 0.02),
        "MIN_RR": trial.suggest_float("min_rr", 0.5, 2.0),
    }

    # 动态赋值给全局单例 settings
    for key, value in config.items():
        setattr(settings, key, value)

    # --- 2. 这里的多只股票是“泛化”的基础 ---
    # 建议至少包含一只宽基指数（如 sz159908）和 1-2 只个股
    tickers = ["sz159908"] 
    all_ticker_scores = []

    for ticker in tickers:
        try:
            # 初始化回测器，设置总数据量为 1000 条
            runner = BacktestRunner(ticker=ticker, period="60", total_limit=1000)
            
            # 执行 2/8 验证
            train_stats, test_stats = runner.run_split_backtest()

            # --- 3. 核心评价逻辑 (上帝视角) ---
            def get_calmar(stats):
                ret = stats.get("Strategy_Return", -5.0)
                mdd = abs(stats.get("Max_Drawdown", 10.0))
                trades = stats.get("Trade_Count", 0)
                if trades < 2 or mdd < 0.01: return -5.0
                return ret / mdd

            train_calmar = get_calmar(train_stats)
            test_calmar = get_calmar(test_stats)

            # 评分公式：
            # 1. 如果训练集都不赚钱，直接淘汰
            if train_stats.get("Strategy_Return", 0) <= 0:
                ticker_score = -20.0
            # 2. 如果测试集发生严重亏损（说明严重过拟合）
            elif test_stats.get("Strategy_Return", 0) < -2.0:
                ticker_score = -15.0
            else:
                # 综合分 = 40% 训练集表现 + 60% 测试集表现
                # 加重测试集权重，逼迫 Optuna 寻找能应对“未来”的参数
                ticker_score = (train_calmar * 0.4) + (test_calmar * 0.6)

            all_ticker_scores.append(ticker_score)

        except Exception as e:
            print(f"回测 {ticker} 出错: {e}")
            all_ticker_scores.append(-30.0)

    # 最终得分：平均分 - 0.5 * 标准差 (惩罚那些在某些票上极好但在另一些票上极差的参数)
    if not all_ticker_scores: return -100.0
    final_score = np.mean(all_ticker_scores) - 0.5 * np.std(all_ticker_scores)
    return final_score

if __name__ == "__main__":
    # 创建研究
    study = optuna.create_study(
        study_name="quant_experiment_v2", 
        storage="sqlite:///db.sqlite3", 
        load_if_exists=True,
        direction="maximize" 
    )

    # 注入初始“经验”
    study.enqueue_trial({
        "model_th": 0.5, "slope_th": 0.0, "init_pt": 0.05, "trend_pt": 0.12,
        "atr_mult": 3, "tp1": 1.08, "tp2": 1.15, "kelly": 0.5,"predict_up_th":-0.01,
        "max_pct": 0.1, "max_lost_pct": 0.01, "atr_stop_mult": 1.5,
        "max_stop": 0.05, "min_stop": 0.01, "min_rr": 0.9,
    })

    # 开始优化
    study.optimize(objective, n_trials=100)
    
    # 持久化最优配置
    write_best_config(study)
    print("最优参数已保存。")