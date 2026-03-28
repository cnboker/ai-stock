import pandas as pd
from simulator.run_backtest import BacktestRunner
from infra.core.dynamic_settings import settings

def diagnose_run(ticker, test_config):
    """
    专门用于诊断为什么不出单的运行模式
    """
    print(f"\n{'='*20} 策略体检开始: {ticker} {'='*20}")
    
    # 1. 注入测试参数
    for key, value in test_config.items():
        setattr(settings, key, value)
        print(f"设定参数 {key} = {value}")

    # 2. 初始化 Runner (只跑最近 100 条，快一点)
    runner = BacktestRunner(ticker=ticker, period="60", total_limit=100)
    
    # 3. 拦截决策函数（核心诊断逻辑）
    # 我们临时“劫持” execute_stock_decision 的输入，观察入场条件
    print("\n[开始扫描入场信号...]")
    
    # 获取可交易日期
    all_dates = pd.Series(runner.df_all.index.date).drop_duplicates().tolist()
    trade_dates = all_dates[-20:] # 只看最近 20 天
    
    match_count = 0
    reject_reasons = {
        "model_confidence": 0,
        "slope_filter": 0,
        "predict_up_filter": 0,
        "position_limit": 0
    }

    for day in trade_dates:
        # 这里模拟 _simulate_day 的部分逻辑，但加入 print 诊断
        df_today = runner.df_all[runner.df_all.index.date == day]
        df_hist = runner.get_history(day, runner.df_all)
        
        # 抽取当天最后一条数据做测试
        ticker_df = pd.concat([df_hist, df_today.head(1)])
        
        # 运行预测
        from predict.chronos_predict import run_prediction
        pre_result = run_prediction(
            df=ticker_df,
            hs300_df=runner.get_history(day, runner.hs300_df),
            ticker=ticker,
            period="60",
            eq_feat=None # 临时传入
        )
        
        # --- 模拟你的 execute_stock_decision 逻辑 ---
        # 1. 从对象中提取属性
        conf = pre_result.model_score  
        median_prices = pre_result.median
        current_price = pre_result.price

        # 2. 计算预测涨幅 (up_pct)
        # 取预测序列的第一个点对比当前价
        up_pct = (median_prices[0] - current_price) / current_price

        # 3. 计算斜率 (slope) 
        # 使用预测序列的终点和起点差值，除以步长，并对当前价归一化
        # 这样 slope 才能和 settings.SLOPE (如 0.005) 在一个量级上
        history_len = len(median_prices)
        slope = (median_prices[-1] - median_prices[0]) / history_len / current_price

        # 4. 打印诊断信息 (确保变量名全部对应)
        print(f"日期: {day} | 置信度: {conf:.4f} | 斜率: {slope:.6f} | 预涨: {up_pct:.4f}")

        # 5. 诊断拦截点
        if conf < settings.MODEL_TH:
            reject_reasons["model_confidence"] += 1
        elif slope < settings.SLOPE:
            reject_reasons["slope_filter"] += 1
        elif up_pct < settings.PREDICT_UP:
            reject_reasons["predict_up_filter"] += 1
        else:
            print(f"✅ >>> 发现潜在入场点！")
            match_count += 1

    print(f"\n{'='*20} 诊断报告 {'='*20}")
    print(f"总扫描次数: {len(trade_dates)}")
    print(f"成功入场次数: {match_count}")
    print(f"被拦截原因分布: {reject_reasons}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # 使用你之前 Trial 45 失败的参数看看原因
    fail_config = {
        "MODEL_TH": 0.5,
        "SLOPE": 0.005,
        "PREDICT_UP": 0.005,
    }
    diagnose_run("sz159908", fail_config)