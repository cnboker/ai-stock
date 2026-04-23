import traceback
import logging
from optimize.Orchestrator import start_optimization_cycle
from position.watchlist_loader import load_watchlist
from position.position_loader import LivePositionLoader
from dotenv import load_dotenv
from infra.core.runtime import GlobalState, RunMode

load_dotenv()


if __name__ == "__main__":
    GlobalState.mode = RunMode.SIM  # 明确设置为 LIVE 模式
    watchlist = load_watchlist("state/watchlist.csv")
    live_tickers = LivePositionLoader.load_tickers("state/live_positions.yaml")

    task_queue = list(dict.fromkeys(watchlist + live_tickers))  # 保持顺序去重

    # task_queue = ['sz300383','sz300603','sh515050','sh588200', 'sh561330','sh512760', 'sz159515']
    # task_queue = ['sh588200','sh515260','sh513130','sh515880']

    # task_queue 2= ['sz300383','sz300603','sh515050','sh588200', 'sh561330','sh512760', 'sz159515']
    # task_queue = ['sh588200','sh515260','sh513130','sh515880']
    task_queue = ["sh603002"]
    print(f"当前观察池: {task_queue}")
    for ticker in task_queue:
        try:

            print(f"🚀 [Start] 正在调优: {ticker}...")
            # 你的 Optuna 调优主逻辑
            start_optimization_cycle(
                ticker, "30", reset_study=False
            )  # 30 分钟级别的调优
            print(f"✅ [Success] {ticker} 调优完成")

        except KeyboardInterrupt:
            # 如果你手动 Ctrl+C，应该让程序停下来，而不是跳到下一个
            print("\n⚠️ 检测到用户中断，正在安全退出...")
            break

        except Exception as e:
            # 1. 记录详细错误到日志文件
            error_msg = f"❌ [Error] {ticker} 失败: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())

            # 2. 控制台只打印关键简讯，避免刷屏
            print(f"{error_msg} (详情请查看 tuning_error.log)")

            # 3. 重要：清理动作（如果有的话）
            # 比如：Study 没关掉，或者数据库 session 没 reset，在这里处理
            continue  # 显式跳过当前，进入下一个 ticker
