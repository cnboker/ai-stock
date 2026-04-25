import traceback

import numpy as np
from data.loader import GlobalState
from simulator.run_backtest import BacktestRunner
from infra.core.dynamic_settings import settings, use_config
from predict.inference import run_prediction
from optimize.config_factory import ConfigFactory
from strategy.slope import compute_hybrid_slope
from strategy.strength import compute_strength


class DiagnosticScanner:
    @staticmethod
    def run_body_check(ticker, ticker_period, test_config):
        """
        核心诊断逻辑：通过上下文注入确保参数生效，并提供深度调参指南
        """
        print(f"\n{'='*20} 🏥 策略深度体检: {ticker} {'='*20}")

        # 1. 参数平铺与注入准备
        category = ConfigFactory.get_ticker_category(ticker).lower()
        final_inject_config = {
            k.replace(f"{category}_", "").upper(): v for k, v in test_config.items()
        }
        print(f"🔍 体检使用参数: {final_inject_config}")
        # 2. 统计容器
        match_count = 0
        reject_reasons = {
            "model_confidence": 0,
            "slope_filter": 0,
            "predict_up_filter": 0,
            "zero_strength_ghost": 0,  # 核心：有斜率但强度消失
            "other_error": 0,
        }

        sample_details = []  # 用于存储入场时的详细数据
        # 1. 确保扫描范围与回测区间对齐 (13天 = 104根)
        scan_length = 104
        # 3. 使用上下文注入执行体检
        all_observed_slopes = []  # 新增：记录所有计算出的原始斜率
        with use_config(final_inject_config):
            try:
                runner = BacktestRunner(ticker=ticker, period=ticker_period)
                trade_bars = runner.df_all.index[-scan_length:]
            except Exception as e:
                print(f"❌ 初始化数据失败: {e}")
                return {"success_count": 0, "intercept_report": {"init_error": 1}}
            
            for current_time in trade_bars:
                try:
                    # 1. 严格切分历史：只取当前 Bar 之前的数据
                    # 确保预测模型看不到 current_time 之后的价格
                    df_context = runner.full_pool_df[:current_time]
                    df_context = runner.full_pool_df[runner.full_pool_df.index <= current_time].tail(settings.strategy_window)
                    hs300_context = runner.full_pool_hs300["close"].reindex(df_context.index).ffill().values
                    # 1. 先更新价格
                    GlobalState.tickers_price[ticker] = df_context['close'].iloc[-1]
                    # 2. 传入预测（确保包含当前的最后一根 Bar 用于特征计算）
                    pre_result = run_prediction(
                        df=df_context,
                        hs300_df=hs300_context,
                        ticker=ticker,
                        period=ticker_period,
                        eq_feat=None,
                    )
                   
                    conf = pre_result.model_score
                    median_prices = pre_result.median
                    current_price = GlobalState.tickers_price[ticker]
                    up_pct = (median_prices[0] - current_price) / current_price
                 
                    slope = compute_hybrid_slope(median_prices, df_context["close"].values)                    
                    
                    all_observed_slopes.append(float(slope))
                    # --- 核心诊断逻辑修正 ---
                    if conf < settings.MODEL_TH:
                        reject_reasons["model_confidence"] += 1
                        continue  # 拦截后跳过后续，防止重复计数

                    if slope < settings.SLOPE:
                        reject_reasons["slope_filter"] += 1
                        continue

                    if up_pct < settings.PREDICT_UP:
                        reject_reasons["predict_up_filter"] += 1
                        continue

                    # 计算强度
                    strength = compute_strength(
                        slope=slope,
                        gate=0.6,
                        alpha=settings.STRENGTH_ALPHA,  # 确保是从动态配置对象里取的
                        slope_threshold=settings.SLOPE,  # 同上
                    )

                    # 判定强度幽灵：有斜率但公式算出来没强度
                    if strength < 0.01:
                        reject_reasons["zero_strength_ghost"] += 1
                    else:
                        match_count += 1
                        sample_details.append(
                            {
                                "day": str(current_time),
                                "slope": float(slope),
                                "strength": float(strength),
                            }
                        )

                except Exception as e:
                    traceback.print_exc()
                    reject_reasons["other_error"] += 1
       
        if all_observed_slopes:
            slopes_array = np.array(all_observed_slopes)
            # 取 95 分位数作为“最大合理斜率”，防止异常值打乱搜索空间
            max_slope_ref = np.percentile(slopes_array[slopes_array > 0], 95) if any(slopes_array > 0) else 0.005
            avg_slope_ref = np.mean(slopes_array[slopes_array > 0]) if any(slopes_array > 0) else 0.001
        else:
            max_slope_ref, avg_slope_ref = 0.01, 0.001
        # 构建给 Gemini 的结构化字典
        body_check_report = {
            "ticker": ticker,
            "status": (
                "HEALTHY" if match_count > 2 else "ANEMIC"
            ),  # 匹配少于2次定义为“贫血”
            "success_count": match_count,
            "total_scans": len(trade_bars) // 8,
            "intercept_report": reject_reasons,
            "current_config": test_config,
            "slope_stats": {
                "max_ref": max_slope_ref,
                "avg_ref": avg_slope_ref,
                "all_positive_slopes": [s for s in all_observed_slopes if s > 0]
            }
        }

        return body_check_report
