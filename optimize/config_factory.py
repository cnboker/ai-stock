import json
import os
from datetime import datetime, timedelta

class ConfigFactory:
    CONFIG_DIR = "optimize/opt_configs"
    TEMPLATE_DIR = "optimize/templates"

    @classmethod
    def _get_ticker_file(cls, ticker: str) -> str:
        os.makedirs(cls.CONFIG_DIR, exist_ok=True)
        return f"{cls.CONFIG_DIR}/{ticker}.json"
    @staticmethod
    def get_ticker_category(ticker: str) -> str:
        """根据代码前缀判断标的类型"""
        # A股个股：00, 30, 60 开头
        if ticker.startswith(("sz00", "sz30", "sh60", "sh68")):
            return "STOCK"
        # ETF基金：51, 15, 58 开头
        elif ticker.startswith(("sh51", "sz15", "sh58", "sh56")):
            return "ETF"
        else:
            return "DEFAULT"
    # --- 新增 1: 数据库路由逻辑 ---
    @classmethod
    def get_db_url(cls, ticker: str) -> str:
        category = cls.get_ticker_category(ticker).lower()
        # 结果：sqlite:///stock.sqlite3 或 sqlite:///etf.sqlite3
        return f"sqlite:///sqllite/{category}.sqlite3"
    # --- 1. 从外部模板加载 ---
    @classmethod
    def _load_template(cls, category: str):
        path = f"{cls.TEMPLATE_DIR}/default_{category.lower()}.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        # 兜底：如果模板文件丢了，返回最简结构
        return {"search_space": {}, "initial_trial": {}}
    
    @staticmethod
    def enqueue_experience(study, ticker: str):
        """
        将 JSON 配置中的 initial_trial 作为第一个 Trial 注入 Optuna。
        """
        try:
            cfg = ConfigFactory.load_ticker_config(ticker)
            initial_params = cfg.get("initial_trial", {})
            search_space = cfg.get("search_space", {})

            if not initial_params:
                print(f"⚠️ {ticker}: No initial_trial found, skipping enqueue.")
                return

            # 过滤掉不在搜索空间内的参数，防止 Optuna 报错
            valid_initial = {
                k: v for k, v in initial_params.items() 
                if k in search_space
            }

            if valid_initial:
                study.enqueue_trial(valid_initial)
                print(f"📥 {ticker}: 已将初始经验 (Initial Trial) 注入 Optuna 队列。")
                
        except Exception as e:
            print(f"❌ {ticker} 注入经验失败: {e}")
            
    @staticmethod
    def suggest_config(trial, ticker: str):
        """
        根据 ticker 加载 search_space，并从 trial 中采样参数
        """
        cfg = ConfigFactory.load_ticker_config(ticker)
        search_space = cfg.get("search_space", {})
        
        params = {}
        for key, space in search_space.items():
            low = space.get("low")
            high = space.get("high")
            if key == "slope":
                continue  # slope 由动态调整逻辑生成，不直接采样
            # 根据参数类型和数值特征自动判断采样方式
            if isinstance(low, int) and isinstance(high, int):
                params[key] = trial.suggest_int(key, low, high)
            else:
                # 如果 low > 0 且跨度很大，可以用 log=True
                use_log = space.get("log", False)
                params[key] = trial.suggest_float(key, float(low), float(high), log=use_log)
                
        return params
   
    @classmethod
    def load_ticker_config(cls, ticker: str):
        file_path = cls._get_ticker_file(ticker)
        # 修正：调用你之前的 get_ticker_category
        category = "STOCK" if ticker.startswith(("sz00", "sz30", "sh60", "sh68")) else "ETF"
        
        # 加载基础模板
        template = cls._load_template(category)
        
        full_cfg = {
            "ticker": ticker,
            "category": category,
            "best_params": {},
            "intercept_stats": {}, # 初始化拦截统计
            "search_space": template.get("search_space", {}),
            "initial_trial": template.get("initial_trial", {}),
            "_META": template.get("_META", {}) # 包含 last_optimized, best_value 等元信息")
        }

        # 如果有专属配置，则覆盖
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                user_cfg = json.load(f)
                if "search_space" in user_cfg: full_cfg["search_space"].update(user_cfg["search_space"])
                if "initial_trial" in user_cfg: full_cfg["initial_trial"].update(user_cfg["initial_trial"])
                full_cfg["best_params"] = user_cfg.get("best_params", {})
                full_cfg["_META"] = user_cfg.get("_META", {})
                full_cfg["intercept_stats"] = user_cfg.get("intercept_stats", {})
        
        return full_cfg
    
    @classmethod
    def should_skip_optimization(cls, ticker: str) -> bool:
        
        """
        检查是否符合跳过优化的条件：
        1. 距离上次优化不足 2 天
        2. 且上次的最优值 (best_value) 大于 0
        """
        current_cfg = cls.load_ticker_config(ticker)
        meta = current_cfg.get("_META", {})
        last_time_str = meta.get("last_optimized")
        best_value = meta.get("best_value", 0)

        if not last_time_str:
            return False  # 从未优化过，不跳过

        try:
            # 将字符串转回 datetime 对象
            last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
            
            # 判断条件
            is_recent = datetime.now() - last_time < timedelta(days=3)
            is_good_enough = best_value > 0
            
            if is_recent:
                print(f"⏭️  [跳过] {ticker} 最近已优化 ({last_time_str}) 且表现良好 (Value: {best_value})")
                return True
                
        except ValueError:
            # 如果日期格式解析失败，安全起见不跳过
            print(f"⚠️ {ticker} 的 last_optimized 日期格式错误: {last_time_str}. 将进行优化。")
            return False

        return False

    # --- 2. 强化保存逻辑 (含拦截分析) ---
    @classmethod
    def save_results(cls, ticker: str, best_params: dict, best_value: float, intercept_report: dict = None):
        """
        intercept_report: 传入 {'slope_filter': 12, 'model_confidence': 8, 'success_count': 0}
        """
        file_path = cls._get_ticker_file(ticker)
        current_cfg = cls.load_ticker_config(ticker)
        
        # 更新最优参数
        current_cfg["best_params"] = best_params
        
        # 更新拦截诊断统计
        if intercept_report:
            total = sum(v for k, v in intercept_report.items() if isinstance(v, int) and k != 'success_count')
            success = intercept_report.get('success_count', 0)
            current_cfg["intercept_stats"] = {
                **intercept_report,
                "total_scans": total,
                "success_rate": f"{(success/total*100):.1f}%" if total > 0 else "0%"
            }

        # 更新元数据
        current_cfg["_META"] = {
            "last_optimized": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_value": round(best_value, 5),
            "status": "ACTIVE" if (intercept_report and intercept_report.get('success_count', 0) > 0) else "SIGNAL_BLOCKED"
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(current_cfg, f, indent=4, ensure_ascii=False)
        
        print(f"💾 [JSON] {ticker} 档案已更新。状态: {current_cfg['_META']['status']}")
