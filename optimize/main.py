import numpy as np
import optuna
from simulator.run_backtest import run_backtest  # 假设这是你的回测入口
from infra.core.config import settings
from persist import write_best_config


def objective(trial):
    # 1. 让 Optuna 建议一组参数
    config = {
        # 入场门槛：如果太高就没单子，太低就是炮灰
        "MODEL_LONG_THRESHOLD": trial.suggest_float("model_th", 0.3,0.6),
        "TREND_SLOPE_THRESHOLD": trial.suggest_float("slope_th", -0.02, 0.02),
        "PREDICTED_UP": trial.suggest_float("predict_up_th", -0.05, 0.02),
        # 止盈阶段触发点：优化什么时候进入“利润锁定”和“趋势跟踪”
        "INIT_PROFIT_TRIGGER": trial.suggest_float("init_pt", 0.03, 0.08),
        "TREND_STAGE_TRIGGER": trial.suggest_float("trend_pt", 0.10, 0.25),
        # 移动止损 ATR 倍数：核心参数！太紧被甩下车，太松利润回撤大
        "ATR_MULTIPLIER": trial.suggest_float("atr_mult", 1.0, 3.5),
        # 分批止盈点
        "TP1_RATIO": trial.suggest_float("tp1", 1.05, 1.15),
        "TP2_RATIO": trial.suggest_float("tp2", 1.15, 1.30),
        # 凯利公式缩放：直接影响仓位大小
        "KELLY_FRACTION": trial.suggest_float("kelly", 0.1, 0.7),
        "MAX_SIGNAL_PCT": trial.suggest_float("max_pct", 0.01, 0.05),
        # --- 风控
        # 趋势过滤：ATR 止损倍数（这是你拿住利润的核心）
        "ATR_STOP_MULT": trial.suggest_float("atr_stop_mult", 0.5, 2.0),
        # 止盈野心：ATR 止盈倍数
        "ATR_TAKE_MULT": trial.suggest_float("atr_take_mult", 2.5, 8.0),
        # 止损底线：最大允许的价格回撤百分比 (对应原 0.95)
        # 建议设为 0.03 到 0.10 (即 3% 到 10%)
        "MAX_STOP_PCT_BASE": trial.suggest_float("max_stop_base", 0.03, 0.10),
        # 止损上限：最小的价格保护空间 (对应原 0.99)
        "MIN_STOP_PCT_BASE": trial.suggest_float("min_stop_base", 0.005, 0.02),
        "MIN_RR": trial.suggest_float("min_rr", 0.5, 2.0)
    }

    # 2. 【关键步】动态赋值给全局单例 settings
    for key, value in config.items():
        setattr(settings, key, value)
    # --- 2. 定义测试集 (你的多只股票列表) ---
    tickers = ["sh603123", "sz000617", "sz002137", "sz301380", "sz300785"] 
    
    all_returns = []
    trade_counts = []

    # --- 3. 循环回测每只股票 ---
    for ticker in tickers:
        try:
            # 运行单只股票回测 (确保内部使用了全局 settings)
            stats = run_backtest(ticker) 
            
            if stats and stats.get("trade_count", 0) > 0:
                all_returns.append(stats["Strategy_Return"])
                trade_counts.append(stats["Trade_Count"])
            else:
                # 惩罚：如果没有交易，给一个负收益
                all_returns.append(-2.0) 
                trade_counts.append(0)
        except Exception:
            all_returns.append(-5.0) # 报错惩罚

    # --- 4. 【核心】计算综合评价分数 (泛化评分) ---
    if not all_returns:
        return -100.0

    avg_return = np.mean(all_returns)      # 平均收益
    std_return = np.std(all_returns)       # 收益的标准差（衡量稳定性）
    total_trades = sum(trade_counts)       # 总交易次数
    
    # 评价指标公式：平均收益 - 0.5 * 标准差
    # 逻辑：收益越高越好，但股票间差异越大（不稳定）扣分越多
    # 同时要求总交易次数不能太少，否则是运气
    if total_trades < len(tickers) * 1: # 平均每只票不到 1 笔交易
        return -10.0
        
    final_score = avg_return - (0.5 * std_return)
    
    # 将此轮全市场的表现记录到 CSV (可选)
    # log_to_csv(trial.number, config, avg_return, std_return, total_trades)
    return final_score



if __name__ == "__main__":
   
    # 启动调参
    study = optuna.create_study(direction="maximize")
    # 建议先把你的现有写死参数作为第一个 Trial 运行
    study.enqueue_trial(
        {
            "model_th": 0.5,
            "slope_th": 0.005,
            "init_pt": 0.05,
            "trend_pt": 0.12,
            "atr_mult": 1.5,
            "tp1": 1.08,
            "tp2": 1.15,
            "kelly": 0.5,
            "max_pct": 0.02,
            "atr_stop_mult": 1.5,
            "atr_take_mult": 4.0,
            "max_stop_base": 0.05,
            "min_stop_base": 0.01,
            "min_rr": 0.9
        }
    )
    study.optimize(objective, n_trials=100)
    write_best_config(study)

    