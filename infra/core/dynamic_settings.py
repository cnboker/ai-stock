from contextvars import ContextVar
from typing import Any

# 定义一个存储动态配置的容器
_dynamic_config_ctx: ContextVar[dict] = ContextVar("dynamic_config", default={})

class DynamicSettings:
    """声明式代理：优先读取动态上下文，没有则回退到静态 settings"""
    def __getattr__(self, name: str) -> Any:
        # 1. 尝试从当前回测的上下文中取值
        ctx = _dynamic_config_ctx.get()
        if name in ctx:
            return ctx[name]
        
        # 2. 如果没有动态值，回退到原始静态配置 (假设原配置叫 BaseSettings)
        # 这里你可以引用你原来的静态 settings 逻辑
        from .base_settings import BaseSettings 
        return getattr(BaseSettings, name)

# 导出这个代理对象，替换掉原来的 settings
settings = DynamicSettings()

# 提供一个上下文管理器，用于注入 Optuna 参数
from contextlib import contextmanager

@contextmanager
def use_config(config: dict):
    """声明式注入入口"""
    token = _dynamic_config_ctx.set(config)
    try:
        yield
    finally:
        _dynamic_config_ctx.reset(token)