import time
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"🚀 函数 {func.__name__} 运行耗时: {end - start:.4f}s")
        return result
    return wrapper

