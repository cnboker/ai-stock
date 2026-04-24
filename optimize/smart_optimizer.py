import time
from optimize.config_factory import ConfigFactory
from optimize.diagnostic_scanner import DiagnosticScanner

class SmartOptimizer:
    def __init__(self, ticker, ticker_period="30"):
        self.ticker = ticker
        self.ticker_period = ticker_period
        self.max_retries = 10  # 最大自动扩张次数

    def adapt_search_space(self, report):
        """根据拦截报告，智能化、步进式扩张搜索空间"""
        cfg = ConfigFactory.load_ticker_config(self.ticker)
        dist = report['intercept_report']
        # 计算有效拦截总数（排除 error）
        total_intercepts = sum([v for k, v in dist.items() if "error" not in k])
        
        if total_intercepts == 0:
            return False

        changed = False

        # 2. 针对 Model Confidence 的自适应
        if dist.get('model_confidence', 0) / total_intercepts > 0.3:
            old_th = cfg["search_space"]["model_th"]["low"]
            new_th = round(max(0.35, old_th - 0.05), 2) # 最低不降过 0.35
            cfg["search_space"]["model_th"]["low"] = new_th
            cfg["initial_trial"]["model_th"] = new_th
            print(f"🤖 [智能扩张] 置信度要求过严，已下调 low 至: {new_th}")
            changed = True

        # 3. 针对 Predict Up 的自适应
        if dist.get('predict_up_filter', 0) / total_intercepts > 0.2:
            # 如果因为预涨幅拦住了，直接尝试接受稍微平庸的涨幅
            cfg["search_space"]["predict_up"]["low"] = -0.05
            cfg["initial_trial"]["predict_up"] = -0.02
            print(f"🤖 [智能扩张] 预涨幅放宽，接受更早的入场时机")
            changed = True

        # 辅助修正：确保确认天数在探测期不要太长
        if cfg["initial_trial"].get("confirm_n", 3) > 1:
            cfg["initial_trial"]["confirm_n"] = 1
            changed = True
        
        if changed:
            # 使用我们 V3 版的保存方法，这会保留 search_space 并更新 _META
            # 注意：此时 best_value 还是旧的，因为还没开始正式优化
            ConfigFactory.save_results(self.ticker, cfg.get("best_params", {}), -137.169, report)
        
        return changed

    def start_auto_loop(self):
        """无人值守主循环：体检 -> 扩张 -> 调优"""
        for i in range(self.max_retries):
            print(f"\n{'#'*30}")
            print(f"### [轮次 {i+1}/{self.max_retries}] 自动化调优流程: {self.ticker}")
            print(f"{'#'*30}")
            
            # 1. 加载当前配置 (可能已经过上一轮扩张)
            cfg = ConfigFactory.load_ticker_config(self.ticker)
            #print(f"cfg={cfg}")
            # 2. 执行真实体检
            # 使用配置中的 initial_trial 作为测试样本
            report = DiagnosticScanner.run_body_check(self.ticker, self.ticker_period, cfg["initial_trial"])

            # 3. 判断是否具备调优条件 (成功入场次数 > 0)
            if report['success_count'] > 0:
                print(f"\n🚀 [触发成交] 探测到 {report['success_count']} 个入场信号！")
                print(f"🎯 正在以此空间启动 Optuna 深度进化...")
                
                # 这里调用你原来的 Optuna 启动逻辑
                # 示例: run_ticker_study(self.ticker)
                self.launch_optuna_work()
                return True
            
            # 4. 如果没有信号，尝试扩张搜索空间
            print(f"\n⚠️ 警告: 当前参数空间下无交易信号。")
            success_adaptive = self.adapt_search_space(report)
            
            if not success_adaptive:
                print("🛑 无法进一步扩张空间或无拦截数据，流程中断。")
                break
            
            print(f"⏳ 空间已更新，等待 2 秒后进入下一轮探测...")
            time.sleep(2)
                
        print(f"\n❌ 最终失败: 经过 {self.max_retries} 轮扩张，{self.ticker} 依然无法触发交易。")
        print("💡 建议：检查该标的是否处于长期停牌、退市或模型数据缺失。")
        return False

    def launch_optuna_work(self):
        """
        这里对接你之前的 Optuna 启动代码
        它会通过 ConfigFactory.suggest_config 自动读取你刚才扩张好的 search_space
        """
        print(f"🛠️  Optuna Study 启动中... 数据库路由: {ConfigFactory.get_db_url(self.ticker)}")
        # 调用你现有的 optimize 逻辑...
        pass

