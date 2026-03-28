
# 定义 CSV 文件名
import csv
import json
import os


CSV_FILE = "outputs/optuna_results.log.csv"

def write_head():
    # 初始化 CSV 表头（包含所有参数名 + 回测结果字段）
    HEADERS = [
        "trial_num", 
        "MODEL_TH", "SLOPE", "PREDICT_UP", 
        "INIT_PT", "TREND_STAGE", "ATR_MULT", 
        "TP1", "TP2", "KELLY",
        "ATR_STOP", "MIN_STOP",
        
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
                config.get("MODEL_TH"),
                config.get("SLOPE"),
                config.get("PREDICT_UP"),
                config.get("INIT_PT"),
                config.get("TREND_STAGE"),
                config.get("ATR_MULT"),
                config.get("TP1"),
                config.get("TP2"),
                config.get("KELLY"),
                config.get("ATR_STOP"),
                config.get("MIN_STOP"),
               
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
    # 因为你 suggest 时用了简写（如 "model_th"），而 settings 里是 "MODEL_TH"
    # 我们需要一个映射表
    param_mapping = {
        "model_th": "MODEL_TH",
        "slope_th": "SLOPE",
        "predict_up_th": "PREDICT_UP",
        "init_pt": "INIT_PT",
        "trend_pt": "TREND_STAGE",
        "atr_mult": "ATR_MULT",
        "tp1": "TP1",
        "tp2": "TP2",
        "kelly": "KELLY",
        "atr_stop_mult": "ATR_STOP",
        "max_lost_pct":"RISK",
        "max_stop": "MAX_STOP", # 注意你代码里定义的 key
        "min_stop": "MIN_STOP",
        
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