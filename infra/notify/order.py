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


def notify_order_v1(event: OrderEvent, position_mgr):
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
       # 修正点：不要用 asyncio.run，而是获取当前的 loop 并创建任务
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(play_sound())
        except RuntimeError:
            # 如果当前没有运行的 loop（比如在某些测试环境），则回退处理
            pass

from infra.core.runtime import GlobalState,RunMode
async def play_sound():
    if GlobalState.mode != RunMode.LIVE:
        return
    # This starts the process without stopping your whole program
    process = await asyncio.create_subprocess_exec("xdg-open", "data/tick.mp3")
    
    # Optional: wait for the command to finish in the background
    await process.wait()


    # 配置你的 API 地址
API_URL = "http://localhost:8000/api/v1/analytics/orders"

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
        # 3. 构造要写入数据库的数据（对应 SQLModel 的 Order 模型）
        order_data = {
            "symbol": event.symbol,
            "side": event.side,  # "buy" or "sell"
            "entry_price": event.price,
            "entry_time": event.ts.isoformat(),
            "actual_return": 0.0, # 初始值为0，收盘后由 analytics 接口更新
            "status": "closed" if event.action in {"CLOSE", "REDUCE"} else "open",
            # 如果你有 trade_id 或关联的 prediction_id，也可以加上
        }

        # 4. 异步处理：写入数据库和播放声音
        try:
            loop = asyncio.get_running_loop()
            
            # 定义一个内部协程，处理 HTTP 请求
            async def persist_and_notify():
                # 写入数据库
                async with httpx.AsyncClient() as client:
                    try:
                        r = await client.post(API_URL, json=order_data, timeout=5.0)
                        if r.status_code == 200:
                            print(f"✅ Order persisted to DB: {event.symbol}")
                        else:
                            print(f"⚠️ DB Persist Failed: {r.text}")
                    except Exception as e:
                        print(f"❌ Network Error during DB persist: {e}")
                
                # 播放声音
                await play_sound()

            # 将任务加入事件循环
            loop.create_task(persist_and_notify())
            
        except RuntimeError:
            # 如果当前没有 running loop，说明可能在脚本直接运行模式下
            pass