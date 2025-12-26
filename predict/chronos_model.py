# predict/chronos_model.py
import torch
from chronos import Chronos2Pipeline, ChronosPipeline


def get_device():
    """
    自动选择设备
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_chronos_model(
    model_name: str = "chronos-2",
    base_path: str = "."
):
    """
    model_name:
        - "chronos-2"
        - "chronos-t5-large"
    """

    device = get_device()
    model_path = f"{base_path}/{model_name}"

    print(f"[Chronos] loading {model_name} on {device}")

    # Chronos-2
    if model_name == "chronos-2":
        pipeline = Chronos2Pipeline.from_pretrained(
            model_path,
            device_map=device
        )

    # Chronos T5 Large
    elif model_name == "chronos-t5-large":
        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return pipeline

