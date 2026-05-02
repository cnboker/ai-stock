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
from log import signal_log
from position import watchlist_loader
from position.position_loader import LivePositionLoader
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder
from equity.equity_features import equity_features
from config.settings import TICKER_PERIOD, UPDATE_INTERVAL_SEC
from data.loader import GlobalState, load_index_df, load_stock_df
from predict.time_utils import is_market_break
from trade.signal_arbiter import SignalArbiter
from trade.trade_engine import execute_final_order, execute_final_order, execute_stock_decision
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
        has_pos = position_mgr.has_any_position()

        tickers = position_mgr.get_tickers_from_positions_and_watchlist()
        print(f"🔍 [Tickers] 活跃标的: {tickers } (持仓: {position_mgr.positions.keys()}, 关注: {position_mgr.watchlist.keys()})")  
        symbols_price = await stock_provider.get_price_map(tickers)
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
      
        arbiter = SignalArbiter(max_slots=1)
        # 5. 遍历任务队列 (统一逻辑)
        for ticker in tickers:
            try:
                final_config = dynamic_config_manager.load_params(ticker)
                CHRONOS_CONTEXT_LENGTH = final_config.get("WINDOW",128)
                final_config.update({"CHRONOS_CONTEXT_LENGTH": CHRONOS_CONTEXT_LENGTH})  # 注入到全局配置中 
                #print(f'current_params={final_config}')
                best_value = final_config.get("_META", {}).get("best_value", 0) 
                print(f"🔍 [Config] {ticker} best_value={best_value}")
                if best_value < 0:
                    print(f"⚠️ [Skipped] {ticker} best_value={best_value} < 0")
                    continue
               
                df = load_stock_df(ticker, session.period)
                if df is None or df.empty:
                    continue

                with use_config(final_config):
                   
                    res = execute_stock_decision(
                        ticker=ticker,
                        hs300_df=session.hs300_df,
                        ticker_df=df,
                        session=session
                    )
                
                    # 如果是买入候选人，加入漏斗
                    if res["type"] == "candidate":
                        arbiter.add_candidate(res)  
                    eq_recorder.add(position_mgr.equity)  # 每处理一个标的就记录一次权益，保持数据的完整性
            except Exception as e:
                print(f"❌ [Error] {ticker} 分析失败: {e}")
                traceback.print_exc() # 实盘时建议只记日志，不刷屏

        candidates = arbiter.get_best_decisions()
        # --- 2. 截面优选 ---
        if candidates:
            execute_final_order(candidates[0],position_mgr)
        print(f"🏁 [Cycle End] 处理标的总数: {len(tickers)}")

    except Exception as e:
        print(f"🚨 [CRITICAL] 循环崩溃: {e}")
        traceback.print_exc()


async def main_loop(interval):
    """主程序循环"""
    print("🚀 Chronos 后台引擎启动中...")
    from infra.core.runtime import GlobalState
    GlobalState.mode = RunMode.LIVE
    pos_loader = LivePositionLoader("state/live_positions.yaml", position_mgr)
    pos_loader.sync()
    watchlist_loader.live_watchlist_hot_load(position_mgr)
    
    print(f"定时器已启动，每 {interval} 秒执行一次。")
    first_run = True
    try:
        while True:
            
            now = time.time()
            # 计算距离下一个整周期还剩多少秒
            # 例如现在 13:05，下一个 13:30 触发
            wait_time = interval - (now % interval)
            if not first_run:
                print(f"⏳ 等待 {wait_time:.2f} 秒，直到下一个周期...") 
                await asyncio.sleep(wait_time)
            first_run = False
           # 记录开始执行时间，用于性能监控
            cycle_start = time.time()
            try:
                await run_trade_cycle()
            except Exception as e:
                print(f"❌ 周期执行异常: {e}")
                
            elapsed = time.time() - cycle_start
            if elapsed > 10:  # 如果执行耗时过长，打印警告
                print(f"⚠️ 预警：run_trade_cycle 耗时过长 ({elapsed:.2f}s)，请检查模型推断速度")
            #break   # 目前先跑一次，后续改成持续循环
    except KeyboardInterrupt:
        print("\n用户中断，程序退出...")
        


if __name__ == "__main__":
    interval = UPDATE_INTERVAL_SEC
    #interval = 1
     # 计算距离下一个整周期还剩多少秒
    asyncio.run(main_loop(interval))