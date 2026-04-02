import numpy as np
import optuna

from infra.core.dynamic_settings import use_config
from optimize import advanced_score
from optimize.config_factory_v2 import ConfigFactory
from optimize.diagnostic_scanner import DiagnosticScanner
from optimize.persist_manager import PersistManager
from simulator.run_backtest import run_backtest


def run_optuna_study(ticker: str, ticker_interval="30", n_trials=100):
    # 1. 创建研究
    study = optuna.create_study(
        study_name=f"opt_{ticker}",
        storage=ConfigFactory.get_db_url(ticker),
        load_if_exists=True,
        direction="maximize",
    )
    
    # 2. 注入经验
    ConfigFactory.enqueue_experience(study, ticker)

    # 3. 运行优化 (50 次尝试)
    study.optimize(lambda t: objective(t, [ticker], ticker_interval), n_trials)

    # --- 关键修正：体检对象改为 study.best_params ---
    print(f"🎯 优化结束，正在对最优参数进行深度体检...")
    
    # 注意：best_params 通常是不带前缀的大写或小写，
    # 确保 DiagnosticScanner 能处理这种格式。
    best_report = DiagnosticScanner.run_body_check(ticker, ticker_interval, study.best_params)

    # 4. 一键存档 (记录最优得分、最优参数及其拦截表现)
    ConfigFactory.save_results(
        ticker=ticker,
        best_params=study.best_params,
        best_value=study.best_value,
        intercept_report=best_report, # 存档最优参数的拦截情况
    )
    PersistManager.save_best_config(study, ticker)  # 可选：将整个 study 存档到文件系统或数据库

    # 1. 看看哪个参数最重要（必看！）
    importance = optuna.importance.get_param_importances(study)
    print(f"📊 参数重要性排行: {importance}")

    # 2. 看看搜索过程（有没有往高收益区靠拢）
    # 如果你在 Jupyter 里，可以直接显示图表
    import optuna.visualization as vis
    vis.plot_optimization_history(study).show()

    print(f"⭐ {ticker} 存档成功 | 最佳分值: {study.best_value:.4f}")

def objective(trial, tickers: list, ticker_interval="30"):
    # 使用列表中的第一个标的来决定参数分类 (ETF 或 STOCK)
    main_ticker = tickers[0]

    # 1. 获取动态配置 (带前缀)
    optuna_config = ConfigFactory.suggest_config(trial, main_ticker)

    # 2. 参数平铺化 (转大写并去前缀)
    category = ConfigFactory.get_ticker_category(main_ticker).lower()
    final_config = {
        k.replace(f"{category}_", "").upper(): v for k, v in optuna_config.items()
    }

    scores = []
    # 3. 声明式注入并循环跑分
    with use_config(final_config):
        
        for t_code in tickers:
            try:
                # 运行回测
                train_stats, test_stats = run_backtest(t_code,period=ticker_interval)

                # 评分逻辑
                train_val = max(-200, advanced_score.get_advanced_score(train_stats))
                test_val = max(-200, advanced_score.get_advanced_score(test_stats))

                # 惩罚项计算
                test_return = test_stats.get("Strategy_Return", 0)
                ticker_score = (train_val * 0.7) + (test_val * 0.3)

                if test_return < -0.05:
                    penalty = (abs(test_return) - 0.05) * 100
                    ticker_score -= penalty

                scores.append(ticker_score)

            except Exception as e:
                print(f"Error on {t_code}: {e}")
                scores.append(-300.0)  # 报错给极低分，让 Optuna 避开

    # 4. 汇总：均值 - 0.5 * 标准差 (追求多标的普适性，防止偏科)
    return np.mean(scores) - 0.5 * np.std(scores)
