import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
from data.loader import GlobalState
from infra.core.dynamic_settings import use_config
from optimize import advanced_score
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
    # --- 1. 参数采样 ---
    # 先选窗口
    strategy_window = trial.suggest_int("window", 20, 160, step=4)
    
    # 动态计算上下文长度
    chronos_context_len = min(int(strategy_window * 1.2), 192)

    # 获取动态 slope 上限
    max_slope_limit = trial.study.user_attrs.get("dynamic_max_slope", 0.02)
    dynamic_slope = trial.suggest_float(
        "slope", 1e-5, max(max_slope_limit, 0.001)
    )

    # --- 2. 统一构建配置字典 ---
    # 从工厂获取基础搜索空间参数 (model_th, atr_mult 等)
    optuna_config = ConfigFactory.suggest_config(trial, ticker)
    
    # 强制覆盖并注入动态计算的参数
    optuna_config.update({
        "window": strategy_window,
        "chronos_context_length": chronos_context_len,
        "slope": dynamic_slope,
        "ticker": ticker,
        "period": ticker_period
    })
    print(f"\n🔍 当前试验参数: {optuna_config}, max_slope_limit={max_slope_limit}" )
    scores = []
    # 3. 声明式注入并循环跑分
    with use_config(optuna_config):
        try:
            # 运行回测
            train_stats, test_stats = run_backtest(ticker,period=ticker_period)
            # ... 回测逻辑 ...
            trial.set_user_attr("train_strategy_return", train_stats["Strategy_Return"])
            trial.set_user_attr("train_max_drawdown", train_stats["Max_Drawdown"])      # 记录最大回撤
            trial.set_user_attr("train_alpha", train_stats["Alpha"])    # 记录胜率
            trial.set_user_attr("train_trade_count", train_stats["Trade_Count"]) # 记录交易次数
           
            trial.set_user_attr("test_strategy_return", test_stats["Strategy_Return"])
            trial.set_user_attr("test_max_drawdown", test_stats["Max_Drawdown"])      # 记录最大回撤
            trial.set_user_attr("test_alpha", test_stats["Alpha"])    # 记录胜率
            trial.set_user_attr("test_trade_count", test_stats["Trade_Count"]) # 记录交易次数
            # 评分逻辑
            train_val = max(-200, advanced_score.get_advanced_score(train_stats, is_test=False))
            test_val = max(-200, advanced_score.get_advanced_score(test_stats, is_test=True))
            print(f"📈 Train Score: {train_val:.2f}, Test Score: {test_val:.2f} (before penalty)")
            # 惩罚项计算
            test_return = test_stats.get("Strategy_Return", 0)
            ticker_score = (train_val * 0.7) + (test_val * 0.3)

            # 2. 平滑惩罚逻辑
            # 对于创业板股票，建议将容忍度放宽到 -8% 到 -10%
            tolerance = -0.08 

            if test_return < tolerance:
                # 采用平方惩罚或者更温和的倍数，避免分值瞬间跳变导致 Optuna 失去方向
                # 只有当亏损真正扩大时（如超过 8%），才开始显著扣分
                penalty = (abs(test_return) - abs(tolerance)) * 50 # 倍数从 100 降到 50
                ticker_score -= penalty

            # 3. 额外奖励：如果验证集盈利且跑赢了 BuyHold (Alpha > 0)
            if test_return > 0 and test_stats.get("Alpha", 0) > 0:
                ticker_score += 10.0 # 给予泛化能力优秀的额外加分

            scores.append(ticker_score)

        except Exception as e:
            print(f"Error on {ticker}: {e}")
            scores.append(-300.0)  # 报错给极低分，让 Optuna 避开

    # 4. 汇总：均值 - 0.5 * 标准差 (追求多标的普适性，防止偏科)
    return np.mean(scores) - 0.5 * np.std(scores)
