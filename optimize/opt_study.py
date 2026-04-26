import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
from data.loader import GlobalState
from infra.core.dynamic_settings import use_config
from optimize.advanced_score import get_advanced_score
from optimize.config_factory import ConfigFactory
from optimize.diagnostic_scanner import DiagnosticScanner
from optimize.persist_manager import PersistManager
from simulator.run_backtest import run_backtest


def run_optuna_study(ticker: str, 
                     ticker_period="30", 
                     n_trials=100, 
                     reset_study=False,
                     slope_stats=None # 新增参数：传入体检得到的统计数据
                     ):
    
    study_name=f"opt_{ticker}"
    storage=ConfigFactory.get_db_url(ticker)
    if reset_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"成功移除旧的任务: {study_name}")
        except KeyError:
            print(f"没有找到名为 {study_name} 的任务，可以直接开始。")

    # 1. 定义采样器
    # warn_independent_sampling=False 可以减少初期的警告输出
    #sampler = CmaEsSampler(warn_independent_sampling=False)
    # 1. 创建研究
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        #sampler=sampler
    )

    if slope_stats:
        # 记录 95 分位最大值和平均值
        study.set_user_attr("dynamic_max_slope", float(slope_stats.get("max_ref", 0.02)))
        study.set_user_attr("dynamic_avg_slope", float(slope_stats.get("avg_ref", 0.001)))
        
    # 2. 注入初始化参数
    ConfigFactory.enqueue_experience(study, ticker)

    # 3. 运行优化 (50 次尝试)
    study.optimize(lambda t: objective(t, ticker, ticker_period), n_trials)

    # --- 关键修正：体检对象改为 study.best_params ---
    print(f"🎯 优化结束，正在对最优参数进行深度体检...")
    
    # 注意：best_params 通常是不带前缀的大写或小写，
    # 确保 DiagnosticScanner 能处理这种格式。
    best_report = DiagnosticScanner.run_body_check(ticker, ticker_period, study.best_params)

    # 4. 一键存档 (记录最优得分、最优参数及其拦截表现)
    ConfigFactory.save_results(
        ticker=ticker,
        best_params=study.best_params,
        best_value=study.best_value,
        intercept_report=best_report, # 存档最优参数的拦截情况
    )
    PersistManager.save_best_config(study, ticker)  # 可选：将整个 study 存档到文件系统或数据库

    # 1. 看看哪个参数最重要（必看！）
    try:
        importance = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError) as e:
        print(f"无法计算参数重要性: {e}")
        importance = {} # 或者返回默认值
    print(f"📊 参数重要性排行: {importance}")

    # 2. 看看搜索过程（有没有往高收益区靠拢）
    # 如果你在 Jupyter 里，可以直接显示图表
    import optuna.visualization as vis
    vis.plot_optimization_history(study).show()

    print(f"⭐ {ticker} 存档成功 | 最佳分值: {study.best_value:.4f}")
    return study

def objective(trial, ticker: str, ticker_period="30"):
    # 记录开始时间用于耗时惩罚
    import time

    # --- 1. 参数采样 ---
    window = trial.suggest_int("window", 20, 160, step=4)
    
    # 动态上下文长度
    chronos_context_len = min(int(window * 1.2), 192)

    # 获取动态 slope
    max_slope_limit = trial.study.user_attrs.get("dynamic_max_slope", 0.02)
    dynamic_slope = trial.suggest_float("slope", 1e-5, max(max_slope_limit, 0.001))

    # --- 2. 配置注入 ---
    # 注意：确保 suggest_config 内部不再包含 "window" 和 "slope"
    optuna_config = ConfigFactory.suggest_config(trial, ticker)
    
    optuna_config.update({
        "window":  window,
        "chronos_context_length": chronos_context_len, # 确保 Key 与模型要求一致
        "slop": dynamic_slope,
    })

    scores = []
    
    with use_config(optuna_config):
        try:
            train_stats, test_stats = run_backtest(ticker, period=ticker_period)
            print(f"Trial {trial.number} - {ticker} | Train Stats: {train_stats} | Test Stats: {test_stats}"    )
            # 记录 User Attrs (略过，保留原样)

            train_val = get_advanced_score(train_stats, is_test=False) # 约 +40
            test_val = get_advanced_score(test_stats, is_test=True)   # 约 -5
            
            # 加权：侧重测试集 (泛化性)
            ticker_score = (train_val * 0.3) + (test_val * 0.7)

            # 额外奖励逻辑 (极其克制)
            test_alpha = test_stats.get("Alpha", 0)  # 转换为百分数
            if test_alpha > 0:
                ticker_score += (test_alpha * 10.0) # 1% Alpha 给 10 分奖励
            elif test_alpha < -2.0:
                ticker_score -= 20.0 # Alpha 亏损超过 2% 才额外扣分
            print(f"火箭 Trial {trial.number}: {ticker} 得分 = {ticker_score:.2f} (Train: {train_val:.2f}, Test: {test_val:.2f}, Alpha: {test_alpha:.2f}%)"  )
            scores.append(ticker_score)

        except Exception as e:
            print(f"Error on {ticker}: {e}")
            scores.append(-2000.0) # 报错分值要比普通亏损更低

    # --- 3. 汇总与效率惩罚 ---

    AVG_SCORE = np.mean(scores)
    STABILITY_PENALTY = np.std(scores) * 0.3 
    
    FINAL_VALUE = AVG_SCORE - STABILITY_PENALTY


    return float(FINAL_VALUE)