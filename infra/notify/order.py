from asyncio import subprocess
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from config.settings import ENABLE_LIVE_PERSIST


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


def notify_order(event: OrderEvent, position_mgr):
    # ===== 控制台（高可读）=====
    headline = f"🚨 ORDER {event.action} {event.side} 🚨"
    detail = (
        f"Symbol: {event.symbol}\n"
        f"Size:   {event.size}\n"
        f"Price:  {event.price:.2f}\n"
        f"Value:  {event.value:.2f}\n"
        f"Reason: {event.reason}\n"
        f"Time:   {event.ts:%Y-%m-%d %H:%M:%S}"
    )
    print("\n" + "=" * 50)
    print(headline)
    print(detail)
    print("=" * 50 + "\n")

    # ===== 只在真实成交后写 =====
    if (
        ENABLE_LIVE_PERSIST
        and event.status == "FILLED"
        and event.action in {"ADD", "OPEN", "CLOSE", "REDUCE"}
    ):
       asyncio.run(play_sound())

from infra.core.runtime import GlobalState,RunMode
async def play_sound():
    if GlobalState.mode != RunMode.LIVE:
        return
    # This starts the process without stopping your whole program
    process = await asyncio.create_subprocess_exec("xdg-open", "data/tick.mp3")
    
    # Optional: wait for the command to finish in the background
    await process.wait()