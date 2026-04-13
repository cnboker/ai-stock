import os
import sys
from functools import lru_cache
import pandas as pd
from .base import BaseTSFMAdapter
from model_torch import is_torch_available, optional_inference_mode

def get_device():
    if not is_torch_available():
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

class ChronosAdapter(BaseTSFMAdapter):
    def __init__(self, model_name: str, base_path: str = "."):
        self.model_name = model_name
        self.base_path = base_path
        # 预加载模型到 pipeline
        self.pipeline = self._get_tsfm_pipeline(model_name, base_path)

    @lru_cache(maxsize=1)
    def _get_tsfm_pipeline(self, model_name, base_path):
        """原 chronos_model.py 的逻辑整合在此"""
        from chronos import Chronos2Pipeline, ChronosPipeline
        
        device = get_device()
        model_path = os.path.join(base_path, "models", model_name)

        if model_name == "chronos-2":
            return Chronos2Pipeline.from_pretrained(
                model_path,
                device_map=device,
                low_cpu_mem_usage=False, # ⚡ 避免生成 Meta Tensor
            )
        elif model_name == "chronos-t5-large":
            return ChronosPipeline.from_pretrained(model_path, device_map=device)
        else:
            raise ValueError(f"ChronosAdapter 不支持模型: {model_name}")

    def predict(self, input_df: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
        """执行推理并返回标准格式"""
        # 调用原始 Chronos Pipeline 的 predict_df 方法
        return self.pipeline.predict_df(
            df=input_df,
            prediction_length=prediction_length
        )