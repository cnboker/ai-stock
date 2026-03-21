
# 定义 CSV 文件名
import csv
import json
import os


CSV_FILE = "outputs/optuna_results.log.csv"

def write_head():
    # 初始化 CSV 表头（包含所有参数名 + 回测结果字段）
    HEADERS = [
        "trial_num", 
        "MODEL_LONG_THRESHOLD", "TREND_SLOPE_THRESHOLD", "PREDICTED_UP", 
        "INIT_PROFIT_TRIGGER", "TREND_STAGE_TRIGGER", "ATR_MULTIPLIER", 
        "TP1_RATIO", "TP2_RATIO", "KELLY_FRACTION", "MAX_SIGNAL_PCT",
        "ATR_STOP_MULT", "ATR_TAKE_MULT", "MAX_STOP_PCT", "MIN_STOP_PCT",
        "MIN_RR",
        "Strategy_Return", "BuyHold_Return", "Max_Drawdown"
    ]

    
    # 1. 无论文件是否存在，直接以 'w' 模式打开
    # 这会自动清空文件内容，并写入全新的表头
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
    
# 将所有生成参数数据写入csv
def write_data(trial,config,stats):
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
                stats.get("Max_Drawdown")
            ]
            writer.writerow(row)
    except Exception as e:
        print(f"写入 CSV 失败: {e}")

# 写最佳参数到参数配置文件
def write_best_config(study):

    # 1. 获取最佳参数字典
    best_params = study.best_params

    # 2. 回写到 settings 对象中 (注意：这里的 key 要对应 settings 里的实际变量名)
    # 因为你 suggest 时用了简写（如 "model_th"），而 settings 里是 "MODEL_LONG_THRESHOLD"
    # 我们需要一个映射表
    param_mapping = {
        "model_th": "MODEL_LONG_THRESHOLD",
        "slope_th": "TREND_SLOPE_THRESHOLD",
        "predict_up_th": "PREDICTED_UP",
        "init_pt": "INIT_PROFIT_TRIGGER",
        "trend_pt": "TREND_STAGE_TRIGGER",
        "atr_mult": "ATR_MULTIPLIER",
        "tp1": "TP1_RATIO",
        "tp2": "TP2_RATIO",
        "kelly": "KELLY_FRACTION",
        "max_pct": "MAX_SIGNAL_PCT",
        "atr_stop_mult": "ATR_STOP_MULT",
        "atr_take_mult": "ATR_TAKE_MULT",
        "max_stop_base": "MAX_STOP_PCT_BASE", # 注意你代码里定义的 key
        "min_stop_base": "MIN_STOP_PCT_BASE",
        "min_rr": "MIN_RR"
    }

    print("最佳参数:", study.best_params)

    best_config = {}
    for optuna_key, value in study.best_params.items():
        settings_key = param_mapping.get(optuna_key, optuna_key)
        best_config[settings_key] = value

    # 保存到文件
    config_path = "outputs/best_strategy_config.json"
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=4)

    print(f"最佳配置已导出至: {config_path}")