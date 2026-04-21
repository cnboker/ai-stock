import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from config.settings import ENABLE_LIVE_PERSIST
from infra.api.client import api_save_order
from infra.core.runtime import GlobalState, RunMode


@dataclass
class OrderEvent:
    ts: datetime
    symbol: str
    action: str  # OPEN / ADD / REDUCE / CLOSE
    side: str  # LONG / SHORT
    size: int  # 手
    price: float
    value: float  # 成交金额
    reason: str  # WHY（信号 / 止损 / 加仓）
    extra: Optional[Dict[str, Any]] = None
    status: str = "FILLED"  # 模拟成交
    prediction_id: Optional[int] = None  # 关联的预测 ID（如果有的话）


async def play_sound():
    if GlobalState.mode != RunMode.LIVE:
        return
    # This starts the process without stopping your whole program
    process = await asyncio.create_subprocess_exec("xdg-open", "data/tick.mp3")

    # Optional: wait for the command to finish in the background
    await process.wait()

    # 配置你的 API 地址


def notify_order(event: OrderEvent, position_mgr):
    # 1. ===== 原有的控制台打印（保留）=====
    headline = f"🚨 ORDER {event.action} {event.side} 🚨"
    detail = (
        f"Symbol: {event.symbol}\n"
        f"Size:   {event.size}\n"
        f"Price:  {event.price:.2f}\n"
        f"Value:  {event.value:.2f}\n"
        f"Reason: {event.reason}\n"
        f"Time:   {event.ts:%Y-%m-%d %H:%M:%S}"
        f"Prediction_id: {event.prediction_id}\n"
    )
    print("\n" + "=" * 50)
    print(headline)
    print(detail)
    print("=" * 50 + "\n")

    # 2. ===== 逻辑判断：只在真实成交后执行 =====
    if (
        ENABLE_LIVE_PERSIST
        and event.status == "FILLED"
        and event.action in {"ADD", "OPEN", "CLOSE", "REDUCE"}
    ):
        ts = position_mgr.current_market_time or datetime.now()
        # 获取当前持仓信息
        pos = position_mgr.positions.get(event.symbol)

        actual_return = 0.0
        pnl_amount = 0.0
        entry_p: float = event.price
        action = event.action
        # --- 关键修复： entry_p 的取值 ---
        if action in ["REDUCE", "CLOSE"]:
            # 减仓或清仓时，必须使用【成交前】的持仓均价
            # 如果 pos 还在，取 pos.entry_price
            # 如果 pos 刚被 del 了（close 逻辑），你需要确保调用 _record 时 pos 对象还没销毁或者传了快照
            entry_p = pos.entry_price if pos else event.price

            # 计算该笔成交的盈亏
            pnl_amount = (
                (event.price - entry_p) * pos.size * (pos.contract_size if pos else 100)
            )
            actual_return = (event.price - entry_p) / entry_p if entry_p > 0 else 0

        elif action == "ADD":
            # 加仓时，我们关心的是“摊薄前”的成本与当前买入价的偏离，
            # 或者简单记录 entry_p 为当前买入价（因为 side 是 buy，不计算 PnL）
            entry_p = event.price
        else:
            # OPEN 阶段
            entry_p = event.price

        order_payload = {
            "symbol": event.symbol,
            "side": event.side,
            "entry_price": event.price,
            "exit_price": event.price if event.side == "sell" else None,
            "entry_time": (
                pos.open_time if (event.side == "sell" and pos) else event.ts
            ).isoformat(),
            "exit_time": event.ts.isoformat() if event.side == "sell" else None,
            "actual_return": actual_return,
            "pnl_amount": pnl_amount,
            "prediction_id": event.prediction_id,
            "status": action,
        }
        print(f"📤 Prepared order payload for API: {order_payload}")
        # 调用之前写的 API 函数
        api_save_order(order_payload)
        # 4. 异步处理：写入数据库和播放声音
        try:
            loop = asyncio.get_running_loop()

            # 定义一个内部协程，处理 HTTP 请求
            async def persist_and_notify():
                # 播放声音
                await play_sound()

            # 将任务加入事件循环
            loop.create_task(persist_and_notify())

        except RuntimeError:
            # 如果当前没有 running loop，说明可能在脚本直接运行模式下
            pass
