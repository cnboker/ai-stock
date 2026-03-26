import os
import threading
import time
import pandas as pd
import warnings
import traceback

# ========================== 项目内模块 ==========================
from global_state import equity_engine, state_lock
from infra.core.context import TradingSession
from infra.core.runtime import RunMode
from position.live_position_loader import live_positions_hot_load
from position.position_factory import create_position_manager
from equity.equity_factory import create_equity_recorder
from predict.prediction_store import load_history
from equity.equity_features import equity_features
from config.settings import TICKER_PERIOD, UPDATE_INTERVAL_SEC
from predict.time_utils import is_market_break
from data.loader import load_index_df, load_stock_df
from trade.trade_engine import execute_stock_decision

# 环境变量与警告忽略
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.data.dataloader"
)

# ========================== 初始化全局状态 ==========================
position_mgr = create_position_manager(0, RunMode.LIVE)
eq_recorder = create_equity_recorder(RunMode.LIVE)


def run_analysis_cycle():
    """单次分析循环核心逻辑"""

    # 1. 检查市场状态
    # if is_market_break():
    #     print(f"[{time.strftime('%H:%M:%S')}] 市场休市中...")
    #     return

    print(f"\n--- 周期开始: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    try:
        period = TICKER_PERIOD
        hs300_df = load_index_df(str(period))

        # 2. 获取当前持仓（加锁保证线程安全）
        with state_lock:
            positions = list(position_mgr.positions.items())

        if not positions:
            print("当前无活跃持仓。")
            return

        # 3. 更新 Equity 决策引擎
        eq_feat = equity_features(eq_recorder.to_series())
        eq_decision = equity_engine.decide(eq_feat, position_mgr.has_any_position())

        session = TradingSession(
            run_mode=RunMode.LIVE,
            period=str(period),
            hs300_df=hs300_df,
            eq_feat=eq_feat,
            tradeIntent=eq_decision,
            position_mgr=position_mgr,
            eq_recorder=eq_recorder,
        )

        results_summary = []

        # 4. 遍历分析所有标的
        for ticker, p in positions:
            try:
                df = load_stock_df(ticker, session.period)
                # 执行核心分析（包含预测逻辑）
                result = execute_stock_decision(
                    ticker=ticker,
                    hs300_df=session.hs300_df,
                    ticker_df=df,
                    session=session,
                )

            except Exception as e:
                print(f"[ERROR] 分析 {ticker} 失败: {e}")
                traceback.print_exc()

        # 5. 这里可以添加结果持久化或报警逻辑
        # df = pd.DataFrame(results_summary)
        # df.to_csv("latest_signals.csv", index=False)

    except Exception as e:
        print(f"[CRITICAL] 循环执行异常: {e}")
        traceback.print_exc()


def main_loop():
    """主程序循环"""
    print("🚀 Chronos 后台引擎启动中...")

    # 加载历史预测
    load_history()

    # 启动持仓热加载线程
    stop_event = threading.Event()
    hotload_thread = threading.Thread(
        target=live_positions_hot_load,
        args=(position_mgr, stop_event),
        daemon=True,
    )
    hotload_thread.start()

    print(f"定时器已启动，每 {UPDATE_INTERVAL_SEC} 秒执行一次。")

    try:
        while True:
            start_time = time.time()

            run_analysis_cycle()

            # 计算补偿时间，确保循环间隔准确
            elapsed = time.time() - start_time
            sleep_time = max(0.1, UPDATE_INTERVAL_SEC - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n用户中断，程序退出...")
        stop_event.set()


if __name__ == "__main__":
    main_loop()
