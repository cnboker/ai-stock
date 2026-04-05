import json
import os
from collections import OrderedDict
from datetime import datetime


class PersistManager:

    # 2. 默认值兜底：如果 Optuna 结果里真的没有，就用这些凑数，保证 JSON 结构完整
    DEFAULTS = {
        "TREND_STAGE": 0.18,
        "MIN_STOP": 0.01,
        "SLOPE_MIN": 0.02,
        "ATR_MULT": 3.87,
        "PREDICT_UP": 0.0,
        "SLOPE": 0.0001,
    }

    # 3. 严格按照你要求的物理顺序排队
    ORDERED_TEMPLATE = [
        "MODEL_TH",
        "SLOPE",
        "PREDICT_UP",
        "INIT_PT",
        "TREND_STAGE",
        "ATR_MULT",
        "TP1",
        "TP2",
        "KELLY",
        "RISK",
        "ATR_STOP",
        "MAX_STOP",
        "MIN_STOP",
        "STRENGTH_ALPHA",
        "SLOPE_MIN",
        "CONFIRM_N",
    ]

    @staticmethod
    def save_ticker_config(ticker, advice_object,config_path:str = "./config/optimized_params"):
        """将 Gemini 建议的参数和空间保存到本地"""
        config_file = f"{config_path}/{ticker}.json"
        
        # 构造要保存的完整字典
        data_to_save = {
            "ticker": ticker,
            "last_optimized": datetime.now(), # 记录时间
            "search_space": advice_object.suggest_search_space,
            "initial_trial": advice_object.recommended_initial_trial,
            "analysis": advice_object.analysis
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f"💾 配置文件已更新: {config_file}")

    @staticmethod
    def save_best_config(
        study, ticker: str, config_root: str = "./config/optimized_params"
    ):
        if not os.path.exists(config_root):
            os.makedirs(config_root)

        raw_best = study.best_params
        print(f"raw_best={raw_best}")
        # --- 步骤 A: 预填充默认值 ---
        cleaned_pool = PersistManager.DEFAULTS.copy()

        # --- 步骤 B: 用 Optuna 结果覆盖默认值 ---
        for k, v in raw_best.items():
            final_key = k.lower().replace("stock_", "").replace("etf_", "").upper()
            # 精度控制
            val = round(v, 4) if isinstance(v, float) else v
            cleaned_pool[final_key] = val
            print(f"{final_key}={val}")
        # --- 步骤 C: 按照 ORDERED_TEMPLATE 抽取数据 ---
        final_ordered_data = OrderedDict()

        # 加上元数据
        final_ordered_data["_META"] = {
            "ticker": ticker,
            "best_value": round(study.best_value, 4),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        for key in PersistManager.ORDERED_TEMPLATE:
            # 只要模板里要求的，池子里一定会通过 DEFAULT 或 Optuna 存在
            if key in cleaned_pool:
                final_ordered_data[key] = cleaned_pool.pop(key)

        # 兜底：处理池子里剩下没在模板里的杂项
        for key, val in cleaned_pool.items():
            final_ordered_data[key] = val

        # --- 步骤 D: 保存 ---
        file_path = os.path.join(config_root, f"{ticker}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    final_ordered_data, f, indent=4, ensure_ascii=False, sort_keys=False
                )
            print(f"✅ 配置文件已生成！已自动填充缺少的参数并按顺序排列: {file_path}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
