import sys
import traceback
from optimize.config_factory_v2 import ConfigFactory
from optimize.opt_study import run_optuna_study
from optimize.smart_optimizer import SmartOptimizer

# 这个脚本是整个自动化调优流程的入口
# 如何stregth总是0,调整strength_alpha=30.0试试
def tune(ticker,ticker_interval="60"):
    # 你可以从命令行读取，也可以写死
    #ticker = "sz300142" 
    
    print(f"\n🚀 >>> 启动标的自动化管线: {ticker} <<<")
    
    # 1. 初始化“破冰器”
    # SmartOptimizer 的 start_auto_loop 内部会：
    #    a. 调用 DiagnosticScanner 检查 initial_trial
    #    b. 如果 success_count == 0，自动扩张 search_space 并更新 JSON
    #    c. 循环直到有信号或达到 max_retries
    optimizer = SmartOptimizer(ticker,ticker_interval)
    
    # 2. 执行自动破冰循环
    # 我们修改 start_auto_loop，让它在成功后返回 True
    is_ready = optimizer.start_auto_loop()

    # 3. 最终决策
    if is_ready:
        print(f"✅ {ticker} 参数空间已就绪，进入正式 Study 阶段。")
        
        # 再次加载最新配置（可能已经被 SmartOptimizer 修改过）
        cfg = ConfigFactory.load_ticker_config(ticker)
        
        # 这里执行最终的深度调优
        # 传入当前的体检报告（可选），用于 save_results 归档
        # 注意：这里的 report 可以在 start_auto_loop 结束前通过 self 存下来
        run_optuna_study(ticker,ticker_interval, 100) 
        
        print(f"🏁 {ticker} 全流程优化完成。")
    else:
        print(f"❌ {ticker} 无法通过自适应扩张触发信号，已跳过深度调优。")

if __name__ == "__main__":
    from position.watchlist_loader import load_watchlist
    from position.position_loader import LivePositionLoader
    watchlist = load_watchlist("state/watchlist.csv")
    live_tickers = LivePositionLoader.load_tickers("state/live_positions.yaml")

    task_queue = list(dict.fromkeys(watchlist + live_tickers)) # 保持顺序去重
    task_queue = ['sz159908']
    print(f"当前观察池: {task_queue}")
    for ticker in task_queue:
        try:
            tune(ticker)
        except Exception as e:
            print(f"❌ [Error] {ticker} 调优失败: {e}")
            traceback.print_exc() # 可选：调优阶段的错误可以选择只记录日志，不刷屏
 