import numpy as np
import optuna
from simulator.run_backtest import run_backtest  # 假设这是你的回测入口
from infra.core.config import settings
from optimize.persist import write_best_config


def objective(trial):
    # 1. 让 Optuna 建议一组参数
    config = {
        # 入场门槛：如果太高就没单子，太低就是炮灰
        "MODEL_LONG_THRESHOLD": trial.suggest_float("model_th", 0.4, 0.9),
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
        #单笔最大投资
        "MAX_SIGNAL_PCT": trial.suggest_float("max_pct", 0.05, 0.2),
        # 单笔最多亏
        "RISK_PER_TRADE": trial.suggest_float("max_lost_pct", 0.01, 0.05),
        # --- 风控
        # 趋势过滤：ATR 止损倍数（这是你拿住利润的核心）
        "ATR_STOP_MULT": trial.suggest_float("atr_stop_mult", 0.5, 2.0),
        # 止损底线：最大允许的价格回撤百分比 (对应原 0.95)
        # 建议设为 0.03 到 0.10 (即 3% 到 10%)
        "MAX_STOP_PCT": trial.suggest_float("max_stop", 0.03, 0.10),
        # 止损上限：最小的价格保护空间 (对应原 1%)
        "MIN_STOP_PCT": trial.suggest_float("min_stop", 0.01, 0.02),
        "MIN_RR": trial.suggest_float("min_rr", 0.5, 2.0),
    }

    # 2. 【关键步】动态赋值给全局单例 settings
    for key, value in config.items():
        setattr(settings, key, value)
    # --- 2. 定义测试集 (你的多只股票列表) ---
    tickers = ["sz159908"]
    #tickers = ["sh603123", "sz000617", "sz002137", "sz301380", "sz300785"]
    scores = []

    # --- 3. 循环回测每只股票 ---
    for ticker in tickers:
        stats = run_backtest(ticker)
        
        # 提取核心数据
        ret = stats.get("Strategy_Return", -5.0)
        mdd = abs(stats.get("Max Drawdown", 10.0)) # 取绝对值
        trades = stats.get("Trade_Count", 0)

        # 惩罚项：如果没有交易，或者回撤是0（通常意味着没开仓）
        if trades < 2 or mdd < 0.0001:
            scores.append(-5.0)
            continue

        # 计算上帝视角分数：卡玛比率 (收益/回撤)
        # 这里的 ret 是百分比，例如 0.33，mdd 是 0.17
        calmar = ret / mdd if mdd > 0 else 0
        
        # 如果输给了基准 (Alpha < 0)，给予轻微惩罚
        if stats.get("Alpha", 0) < 0:
            calmar *= 0.8
            
        scores.append(calmar)

    # 最终得分：平均分 - 标准差 (追求所有票都表现稳健)
    final_score = np.mean(scores) - 0.5 * np.std(scores)
    return final_score



if __name__ == "__main__":

    # 启动调参
    #study = optuna.create_study(direction="maximize")
    # 创建或加载一个名为 "my_optimization" 的研究
    # storage 参数就是它的“记忆库”
    # 经验学习： 采样器会读取数据库中所有已完成的 $x$（超参数）和 $y$（目标值）的关系。它通过历史数据构建概率模型，预测哪些区域更有可能产生最优解，从而避免重复失败的路径。
    study = optuna.create_study(
        study_name="my_experiment", 
        storage="sqlite:///db.sqlite3", 
        load_if_exists=True,
        direction="maximize"  # 必须是最大化分数
    )

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
            "max_pct": 0.1,
            "max_lost_pct": 0.01,
            "atr_stop_mult": 1.5,
            "max_stop": 0.05,
            "min_stop": 0.01,
            "min_rr": 0.9,
        }
    )
    study.optimize(objective, n_trials=50)
    write_best_config(study)



# 周期 (Timeframe),每天 K 线数量,1000 条数据可覆盖时长,是否满足 6 个月？
# 日线 (Daily),1 条,~1000 个交易日 (约 4 年),是 (过度覆盖)
# 1 小时 (1H),4 条,~250 个交易日 (约 12 个月),是 (完美契合)
# 30 分钟 (30M),8 条,~125 个交易日 (约 6 个月),是 (刚好满足)
# 15 分钟 (15M),16 条,~62.5 个交易日 (约 3 个月),否
# 5 分钟 (5M),48 条,~21 个交易日 (约 1 个月),否