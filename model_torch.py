try:
    import torch
except ImportError:
    torch = None

def is_torch_available():
    return torch is not None


def optional_inference_mode():
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        torch = None
        TORCH_AVAILABLE = False
    if is_torch_available():
        return torch.inference_mode()
    else:
        def identity(fn):
            return fn
        return identity