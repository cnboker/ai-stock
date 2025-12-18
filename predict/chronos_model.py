# predict/chronos_model.py
import torch
from chronos import ChronosPipeline

def load_chronos_model(model_path="./chronos-t5-large"):
    return ChronosPipeline.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
