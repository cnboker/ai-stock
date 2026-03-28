import traceback
import numpy as np
import optuna
from optimize import advanced_score
from optimize.config_factory import ConfigFactory
from simulator.run_backtest import run_backtest 
from infra.core.dynamic_settings import use_config

def objective(trial, tickers: list):
    # 使用列表中的第一个标的来决定参数分类 (ETF 或 STOCK)
    main_ticker = tickers[0]
    
    # 1. 获取动态配置 (带前缀)
    optuna_config = ConfigFactory.suggest_config(trial, main_ticker)
    
    # 2. 参数平铺化 (转大写并去前缀)
    category = ConfigFactory.get_ticker_category(main_ticker).lower()
    final_config = {k.replace(f"{category}_", "").upper(): v for k, v in optuna_config.items()}
    
    scores = []
    # 3. 声明式注入并循环跑分
    with use_config(final_config):
        for t_code in tickers:
            try:
                # 运行回测
                train_stats, test_stats = run_backtest(t_code)

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
                scores.append(-300.0) # 报错给极低分，让 Optuna 避开

    # 4. 汇总：均值 - 0.5 * 标准差 (追求多标的普适性，防止偏科)
    return np.mean(scores) - 0.5 * np.std(scores)


def run_optimization(tickers: list):
    """
    tickers: 标的代码列表，例如 ["sz159908", "sh510300"]
    """
    if not tickers:
        return
        
    # 1. 自动路由数据库 (取第一个标的决定存到 etf.sqlite3 还是 stock.sqlite3)
    main_ticker = tickers[0]
    db_url = ConfigFactory.get_db_url(main_ticker)
    
    # 2. 创建 Study (名字包含所有标的，方便区分)
    study_id = "_".join([t.replace("sz", "").replace("sh", "") for t in tickers])
    study = optuna.create_study(
        study_name=f"multi_opt_{study_id}", 
        storage=db_url, 
        load_if_exists=True, 
        direction="maximize" 
    )

    # 3. 初始经验注入
    if len(study.trials) == 0:
        ConfigFactory.enqueue_initial_trial(study, main_ticker)

    # 4. 开始跑分：用 lambda 传递整个列表
    study.optimize(lambda t: objective(t, tickers), n_trials=2)
    
    print(f"🚀 优化完成！最佳标的组合评分: {study.best_value}")
    print(f"📌 最佳参数已存入数据库: {db_url}")
    # --- 核心：自动持久化结果 ---
    from optimize.persist_manager import PersistManager
    PersistManager.save_best_config(study, main_ticker)


if __name__ == "__main__":
    # 场景 A：跑 ETF 组合优化 (追求宽基普适性)
    run_optimization(["sz159908", "sh510300"])
    
    # 场景 B：跑个股组合优化 (如果你以后想试试几只个股能否共用一套参数)
    # run_optimization(["sz002137", "sz300750"])