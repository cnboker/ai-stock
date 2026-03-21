import optuna
from simulator.run_backtest import run_backtest  # 假设这是你的回测入口
from infra.core.config import settings
import csv
import os

# 定义 CSV 文件名
CSV_FILE = "outputs/optuna_results.log.csv"

# 初始化 CSV 表头（包含所有参数名 + 回测结果字段）
HEADERS = [
    "trial_num", 
    "MODEL_LONG_THRESHOLD", "TREND_SLOPE_THRESHOLD", "PREDICTED_UP", 
    "INIT_PROFIT_TRIGGER", "TREND_STAGE_TRIGGER", "ATR_MULTIPLIER", 
    "TP1_RATIO", "TP2_RATIO", "KELLY_FRACTION", "MAX_SIGNAL_PCT",
    "ATR_STOP_MULT", "ATR_TAKE_MULT", "MAX_STOP_PCT", "MIN_STOP_PCT",
    "MIN_RR",
    "Strategy_Return", "BuyHold_Return", "Max_Drawdown", "Score"
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

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
    # 2. 运行回测
    # 注意：run_backtest 内部需要接收这组 config 并应用到 position_mgr 和 budget_mgr
    stats = run_backtest()

    # 3. 定义评价指标（盈利能力指标）
    # 既然你风控不错，目标可以设为：总收益率 * (1 - 最大回撤) 或直接用夏普比率
    # 如果回测期间没交易，返回一个极小值
    # if stats["trade_count"] < 5:
    #     return -1.0

    # 3. 处理回测失败的情况 (防御性逻辑)
    if stats is None:
        return -99.0

    # 4. 【核心步骤】将参数和结果合并，写入 CSV
    try:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # 按照 HEADERS 的顺序组织数据
            row = [
                trial.number,
                config.get("MODEL_LONG_THRESHOLD"),
                config.get("TREND_SLOPE_THRESHOLD"),
                config.get("PREDICTED_UP"),
                config.get("INIT_PROFIT_TRIGGER"),
                config.get("TREND_STAGE_TRIGGER"),
                config.get("ATR_MULTIPLIER"),
                config.get("TP1_RATIO"),
                config.get("TP2_RATIO"),
                config.get("KELLY_FRACTION"),
                config.get("MAX_SIGNAL_PCT"),
                config.get("ATR_STOP_MULT"),
                config.get("ATR_TAKE_MULT"),
                config.get("MAX_STOP_PCT"),
                config.get("MIN_STOP_PCT"),
                config["MIN_RR"],
                stats.get("Strategy_Return"),
                stats.get("BuyHold_Return"),
                stats.get("Max_Drawdown"),
                stats.get("Strategy_Return%") # 这里假设用收益率作为打分标准
            ]
            writer.writerow(row)
    except Exception as e:
        print(f"写入 CSV 失败: {e}")

    # 5. 返回给 Optuna 的最终分值
    return stats.get("Strategy_Return", -99.0)



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

    print("最佳参数:", study.best_params)