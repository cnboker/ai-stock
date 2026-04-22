import asyncio
from datetime import datetime
import os
import time
import warnings
import traceback

# ========================== 项目内模块 ==========================
from global_state import equity_engine, state_lock
from infra.core.dynamic_settings import use_config
from infra.core.trade_session import TradingSession
from infra.core.runtime import RunMode
from infra.utils.sync_watchlist import sync_account_and_watchlist
from position import watchlist_loader
from position.position_loader import LivePositionLoader
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder
from equity.equity_features import equity_features
from config.settings import TICKER_PERIOD, UPDATE_INTERVAL_SEC
from predict.time_utils import is_market_break
from data.loader import GlobalState, load_index_df, load_stock_df
from trade.trade_engine import execute_stock_decision
from typing import List, Tuple
from infra.core.config_manager import dynamic_config_manager
from data.tencent_api import StockProvider


# 环境变量与警告忽略
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.data.dataloader"
)

# ========================== 初始化全局状态 ==========================
sync_account_and_watchlist()
position_mgr = create_position_manager(0)
eq_recorder = create_equity_recorder()
stock_provider = StockProvider()

async def run_trade_cycle():
    if is_market_break():
        print("⏸️ 市场休息中，跳过本周期")
        return
    # 打印周期开始时间（建议使用更简洁的格式）
    now_str = time.strftime('%H:%M:%S')
    print(f"\n🚀 [Cycle Start] {now_str}")

    try:
        # 1. 环境准备
        period = str(TICKER_PERIOD)
        hs300_df = load_index_df(period)
        position_mgr.current_market_time = datetime.now()  # 初始化当前市场时间，后续会在循环中更新 
        # 2. ⚡ 内存快照提取 (一次性锁定，减少竞争)
        with state_lock:
            # 获取所有需要处理的标的：(代码, 持仓对象或None)
            # 这样我们就把持仓和观察池统一成了处理队列
            active_positions = list(position_mgr.positions.items()) # [(ticker, pos_obj), ...]
            watch_tickers = [(t, None) for t in position_mgr.watchlist.keys()]
            
            # 合并队列：先处理持仓（为了止损优先级），再处理观察池
            task_queue: List[Tuple[str, any]] = active_positions + watch_tickers
            print(f"task_queue={task_queue}")
            # 获取当前资金状态
            has_pos = position_mgr.has_any_position()

        tickers = [item[0] for item in task_queue]
        symbols_price = await stock_provider.get_price_map(tickers)
        print(f"symbols_prices={symbols_price}")


        GlobalState.tickers_price = symbols_price
        # 3. 更新 Equity 决策引擎
        eq_feat = equity_features(eq_recorder.to_series())
        eq_decision = equity_engine.decide(eq_feat, has_pos)

        # 4. 初始化交易会话上下文
        session = TradingSession(
            period=period,
            hs300_df=hs300_df,
            eq_feat=eq_feat,
            tradeIntent=eq_decision,
            position_mgr=position_mgr,
            eq_recorder=eq_recorder,
        )

        # 5. 遍历任务队列 (统一逻辑)
        for ticker, pos_obj in task_queue:
            try:
                print(f"ticker={ticker},price={symbols_price[ticker]}")
                # 2. 假设我们要为创业板 ETF 开启交易
                # 它会寻找 sz159908.json -> category_ETF.json -> default
                final_config = dynamic_config_manager.load_params(ticker)
                GlobalState.chronos_context_length = final_config.get("WINDOW",128)
                #print(f'current_params={final_config}')
                best_value = final_config.get("_META", {}).get("best_value", 0) 
                print(f"🔍 [Config] {ticker} best_value={best_value}")
                # if best_value < 0:
                #     print(f"⚠️ [Skipped] {ticker} best_value={best_value} < 0")
                #     continue
                # 加载数据 (如果这里慢，考虑增加多线程读取)
                df = load_stock_df(ticker, session.period)
                if df is None or df.empty:
                    continue
                with use_config(final_config):
                    # 核心决策逻辑执行
                    # 注意：execute_stock_decision 内部应根据 pos_obj 是否为 None 自动判断是 卖出/止损 还是 买入
                    result = execute_stock_decision(
                        ticker=ticker,
                        hs300_df=session.hs300_df,
                        ticker_df=df,
                        session=session,
                        #pos_obj=pos_obj # 传入持仓对象，方便内部逻辑判断
                    )

            except Exception as e:
                print(f"❌ [Error] {ticker} 分析失败: {e}")
                traceback.print_exc() # 实盘时建议只记日志，不刷屏

        print(f"🏁 [Cycle End] 处理标的总数: {len(task_queue)}")

    except Exception as e:
        print(f"🚨 [CRITICAL] 循环崩溃: {e}")
        traceback.print_exc()


async def main_loop():
    """主程序循环"""
    print("🚀 Chronos 后台引擎启动中...")
    from infra.core.runtime import GlobalState
    GlobalState.mode = RunMode.LIVE
    pos_loader = LivePositionLoader("state/live_positions.yaml", position_mgr)
    pos_loader.sync()
    watchlist_loader.live_watchlist_hot_load(position_mgr)
    
    print(f"定时器已启动，每 {UPDATE_INTERVAL_SEC} 秒执行一次。")
    try:
        while True:
            now = time.time()
            # 计算距离下一个整周期还剩多少秒
            # 例如现在 13:05，下一个 13:30 触发
            wait_time = UPDATE_INTERVAL_SEC - (now % UPDATE_INTERVAL_SEC)
            await asyncio.sleep(wait_time)
           # 记录开始执行时间，用于性能监控
            cycle_start = time.time()
            try:
                await run_trade_cycle()
            except Exception as e:
                print(f"❌ 周期执行异常: {e}")
                
            elapsed = time.time() - cycle_start
            if elapsed > 10:  # 如果执行耗时过长，打印警告
                print(f"⚠️ 预警：run_trade_cycle 耗时过长 ({elapsed:.2f}s)，请检查模型推断速度")

    except KeyboardInterrupt:
        print("\n用户中断，程序退出...")
        


if __name__ == "__main__":
    asyncio.run(main_loop())