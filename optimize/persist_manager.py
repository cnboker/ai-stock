import json
import os
import logging
from optimize.config_factory import ConfigFactory

logger = logging.getLogger("PersistManager")

class PersistManager:
    @staticmethod
    def save_best_config(study, ticker: str, config_root: str = "./config/optimized_params"):
        """
        提取 Optuna 结果并保存为分类通用配置
        """
        if not os.path.exists(config_root):
            os.makedirs(config_root)

        # 1. 获取当前分类 (如 ETF 或 STOCK)
        category = ConfigFactory.get_ticker_category(ticker)
        prefix = category.lower() + "_"
        
        # 2. 清洗参数：去除前缀并转为大写
        best_params = study.best_params
        cleaned_config = {}
        
        for k, v in best_params.items():
            # 处理带前缀的参数 (如 etf_model_th -> MODEL_LONG_THRESHOLD)
            if k.startswith(prefix):
                clean_key = k.replace(prefix, "").upper()
                cleaned_config[clean_key] = v
            else:
                # 处理不带前缀的通用参数 (如 strength_alpha -> STRENGTH_ALPHA)
                cleaned_config[k.upper()] = v

        # 3. 写入文件 (例如 category_ETF.json)
        file_name = f"category_{category}.json"
        file_path = os.path.join(config_root, file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_config, f, indent=4, ensure_ascii=False)
            print(f"\n✨ [SUCCESS] 最优参数已同步至: {file_path}")
            print(f"📊 最终得分 (Mean-0.5Std): {study.best_value:.4f}")
        except Exception as e:
            logger.error(f"❌ 自动保存参数失败: {e}")