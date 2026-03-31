import pandas as pd
import numpy as np
from simulator.run_backtest import BacktestRunner
from infra.core.dynamic_settings import settings, use_config
from predict.chronos_predict import run_prediction
from optimize.config_factory_v2 import ConfigFactory
from strategy.strength import compute_strength


class DiagnosticScanner:
    @staticmethod
    def run_body_check(ticker, ticker_interval, test_config):
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

        # 3. 使用上下文注入执行体检
        with use_config(final_inject_config):
            try:
                runner = BacktestRunner(
                    ticker=ticker, period=ticker_interval, total_limit=100
                )
                all_dates = (
                    pd.Series(runner.df_all.index.date).drop_duplicates().tolist()
                )
                trade_dates = all_dates[-20:]
            except Exception as e:
                print(f"❌ 初始化数据失败: {e}")
                return {"success_count": 0, "intercept_report": {"init_error": 1}}

            for day in trade_dates:
                try:
                    df_today = runner.df_all[runner.df_all.index.date == day]
                    df_hist = runner.get_history(day, runner.df_all)
                    ticker_df = pd.concat([df_hist, df_today.head(1)])

                    pre_result = run_prediction(
                        df=ticker_df,
                        hs300_df=runner.get_history(day, runner.hs300_df),
                        ticker=ticker,
                        period=ticker_interval,
                        eq_feat=None,
                    )

                    conf = pre_result.model_score
                    median_prices = pre_result.median
                    current_price = pre_result.price
                    up_pct = (median_prices[0] - current_price) / current_price
                    slope = (
                        (median_prices[-1] - median_prices[0])
                        / len(median_prices)
                        / current_price
                    )

                    # --- 诊断决策流 ---
                    if conf < settings.MODEL_TH:
                        reject_reasons["model_confidence"] += 1
                    elif slope < settings.SLOPE_MIN:
                        reject_reasons["slope_filter"] += 1
                    elif up_pct < settings.PREDICT_UP:
                        reject_reasons["predict_up_filter"] += 1
                    else:
                        # 重点：模拟实盘 Strength 计算逻辑
                        # 假设实盘公式为: (slope * 100) ** alpha 或类似
                        # 我们这里检查原始 slope 经过 alpha 后的存活情况
                        eff_slope = max(0, slope)
                        # 如果 slope 很小(0.01)，且 alpha 很大(2.0)，结果会变成 0.0001
                        sim_gate = 0.6
                       
                        # strength = sim_gate * (eff_slope - settings.SLOPE_MIN) * settings.STRENGTH_ALPHA
                        strength = compute_strength(
                            slope=slope,
                            gate=sim_gate,
                            alpha=settings.STRENGTH_ALPHA,  # 确保是从动态配置对象里取的
                            slope_min=settings.SLOPE_MIN,  # 同上
                        )
                        print(
                            f"📊 {day} |settings.STRENGTH_ALPHA={settings.STRENGTH_ALPHA} | Slope: {slope:.5f} | Eff_Slope: {eff_slope:.5f} | Strength: {strength:.5f}   "
                        )
                        if eff_slope > settings.SLOPE_MIN and strength < 0.01:
                            reject_reasons["zero_strength_ghost"] += 1

                        if strength < 0.01:  # 认定为“强度幽灵”，即信号存在但无法开仓
                            reject_reasons["zero_strength_ghost"] += 1
                        else:
                            match_count += 1
                            sample_details.append(
                                {"day": day, "slope": slope, "strength": strength}
                            )

                except Exception as e:
                    reject_reasons["other_error"] += 1

        # 4. 构建标准化报告
        report = {
            "success_count": match_count,
            "intercept_report": reject_reasons,
            "scan_days": len(trade_dates),
        }

        # 5. 🔥 深度调参指南打印
        DiagnosticScanner._print_doctor_advice(
            ticker, match_count, reject_reasons, final_inject_config
        )

        return report

    @staticmethod
    def _print_doctor_advice(ticker, match_count, reasons, config):
        print(f"\n{'*'*15} 🧠 {ticker} 策略医生诊断报告 {'*'*15}")

        # A. 诊断 Strength 消失问题 (针对你刚才的日志)
        if reasons["zero_strength_ghost"] > 0:
            print(f"🚨 [严重警告] 发现 {reasons['zero_strength_ghost']} 次“信号空转”！")
            print(f"   现象：Slope 为正但 Strength 趋近于 0，导致有信号无仓位。")
            print(
                f"   原因：当前 STRENGTH_ALPHA ({config.get('STRENGTH_ALPHA')}) 过大，过度抑制了小斜率信号。"
            )
            print(
                f"   👉 建议：将 initial_trial 中的 'strength_alpha' 调低至 0.2 ~ 0.5 之间。"
            )

        # B. 诊断 止损问题
        if match_count > 0:
            current_atr = config.get("ATR_STOP", 3.0)
            print(f"🚩 [生存检查] 发现 {match_count} 次入场机会。")
            if current_atr < 4.5:
                print(
                    f"   警惕：当前 ATR_STOP ({current_atr}) 对 {ticker} 来说可能过紧。"
                )
                print(
                    f"   👉 建议：若回测显示“秒止损”，请将 'atr_stop' 提升至 4.5 - 5.5。"
                )

        # C. 诊断 模型门槛
        if reasons["model_confidence"] > 10:
            current_th = config.get("MODEL_TH", 0.5)
            print(f"🚩 [信号稀缺] 模型置信度拦截了大部分行情。")
            print(
                f"   👉 建议：尝试将 'model_th' 调低 0.05（当前: {current_th} -> 目标: {current_th - 0.05:.2f}）。"
            )

        # D. 诊断 确认天数
        current_n = config.get("CONFIRM_N", 3)
        if current_n > 1:
            print(f"🚩 [时效提示] 当前 CONFIRM_N={current_n}。")
            print(
                f"   👉 建议：对于爆发性标的，务必确认 'confirm_n' 已设为 1，否则信号会被延时磨灭。"
            )

        print(f"{'*'*50}\n")
