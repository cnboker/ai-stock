# predict/chronos_model.py
from functools import lru_cache
from model_torch import is_torch_available
from chronos import Chronos2Pipeline, ChronosPipeline
import os
import sys
# 1. 定位到 Kronos_Source 目录
# 假设你的项目结构是：stock-model/predict/chronos_model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上跳一级到项目根目录 stock-model，再进入 Kronos_Source
kronos_source_dir = os.path.abspath(os.path.join(current_dir, "../Kronos_Source"))

# 2. 将源码目录加入系统路径，这样 import model 才能生效
if kronos_source_dir not in sys.path:
    sys.path.insert(0, kronos_source_dir)

def get_device():
    if not is_torch_available():
        return "cpu"

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# cache
@lru_cache(maxsize=2)
def get_tsfm_pipeline(model_name: str = "chronos-2", base_path: str = "."):
    """
    model_name:
        - "chronos-2"
        - "chronos-t5-large"
    """

    device = get_device()
    model_path = f"{base_path}/models/{model_name}"

    # print(f"[Chronos] loading {model_name} on {device}")
    if model_name.startswith("kronos"):
        from model.kronos import KronosPredictor

        pipeline = KronosPredictor.from_pretrained(model_path, device_map=device)
    # Chronos-2
    if model_name == "chronos-2":
        pipeline = Chronos2Pipeline.from_pretrained(
            model_path,
            device_map=device,
            low_cpu_mem_usage=False,  # ⚡ 关键：设为 False，避免生成 Meta Tensor
        )

    # Chronos T5 Large
    elif model_name == "chronos-t5-large":
        pipeline = ChronosPipeline.from_pretrained(model_path, device_map=device)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return pipeline
