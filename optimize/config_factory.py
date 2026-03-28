import optuna


class ConfigFactory:
    @staticmethod
    def get_ticker_category(ticker: str) -> str:
        """根据代码前缀判断标的类型"""
        # A股个股：00, 30, 60 开头
        if ticker.startswith(("sz00", "sz30", "sh60", "sh68")):
            return "STOCK"
        # ETF基金：51, 15, 58 开头
        elif ticker.startswith(("sh51", "sz15", "sh58")):
            return "ETF"
        else:
            return "DEFAULT"

    # --- 新增 1: 数据库路由逻辑 ---
    @classmethod
    def get_db_url(cls, ticker: str) -> str:
        category = cls.get_ticker_category(ticker).lower()
        # 结果：sqlite:///stock.sqlite3 或 sqlite:///etf.sqlite3
        return f"sqlite:///sqllite/{category}.sqlite3"

    # --- 新增 2: 初始经验分发逻辑 ---
    @classmethod
    def enqueue_initial_trial(cls, study: optuna.Study, ticker: str):
        category = cls.get_ticker_category(ticker).lower()

        # 1. 通用参数 (不需要加前缀的)
        base_params = {
            "strength_alpha": 1.3,
            "confirm_n": 3,
            "tp2": 1.20,
        }

        # 2. 类别特有参数 (必须与 suggest_config 中的前缀 Key 严格对应)
        if category == "stock":
            specific_params = {
                f"{category}_model_th": 0.50,  # 个股门槛设高点
                f"{category}_atr_stop": 5.5,  # 个股止损设宽点
                f"{category}_risk": 0.008,  # 个股仓位设轻点
                f"{category}_kelly": 0.2,
                f"{category}_max_stop": 0.12,
            }
        else:  # ETF
            specific_params = {
                f"{category}_model_th": 0.45,
                f"{category}_atr_stop": 3.5,
                f"{category}_risk": 0.015,
                f"{category}_kelly": 0.5,
                f"{category}_max_stop": 0.08,
            }

        # 3. 其他默认参数补全 (这里要确认你的 Factory 里这些有没有加前缀)
        other_params = {
            f"predict_up": 0.001,  # 基础上涨预期 (0.1%)
            f"init_pt": 0.035,  # 3.5% 利润触发第一阶段保护
            f"tp1": 1.05,  # 5% 利润处进行第一次止盈/减仓
            f"trend_stage": 0.15,  # 15% 利润进入趋势跟踪模式
            f"min_stop": 0.01,  # 最小止损位 1% (防止滑点和手续费覆盖)
            f"slope_min": 0.02,  # 强度斜率阈值
            f"atr_mult": 3.5,  # 经典的 3.5 倍 ATR 移动止损
        }

        # 合并并注入
        initial_trial = {**base_params, **specific_params, **other_params}
        study.enqueue_trial(initial_trial)
        print(f"✅ 已为 {ticker} ({category}) 注入初始经验。")

    @classmethod
    def suggest_config(cls, trial: optuna.Trial, ticker: str):
        category = cls.get_ticker_category(ticker).lower()

        # 1. 共有参数 (Shared Parameters)
        config = {
            "STRENGTH_ALPHA": trial.suggest_float("strength_alpha", 1.2, 1.5),
            "CONFIRM_N": trial.suggest_int("confirm_n", 2, 5),
            "TP2": trial.suggest_float("tp2", 1.1, 1.25),
        }

        # 2. 分类差异化参数 (Category-Specific Parameters)
        if category == "stock":
            # 个股逻辑：高门槛、宽止损、低杠杆 (防噪音)
            slope_min = trial.suggest_float("slope_min", 0, 0.005)
            config.update(
                {
                    "MODEL_TH": trial.suggest_float(f"{category}_model_th", 0.45, 0.55),
                    "ATR_STOP": trial.suggest_float(f"{category}_atr_stop", 1.5, 4.0),
                    "RISK": trial.suggest_float(f"{category}_risk", 0.005, 0.01),
                    "KELLY": trial.suggest_float(f"{category}_kelly", 0.15, 0.3),
                    "SLOPE": trial.suggest_float(f"{category}_slope", 0.006, 0.02),
                    "MAX_STOP": trial.suggest_float(f"{category}_max_stop", 0.08, 0.15),
                    "SLOPE_MIN": slope_min,
                    "PREDICT_UP": trial.suggest_float(f"predict_up", -0.05, 0.01),
                }
            )
        else:
            # ETF逻辑：低门槛、窄止损、高杠杆 (抢趋势)
            config.update(
                {
                    "MODEL_TH": trial.suggest_float(f"{category}_model_th", 0.42, 0.48),
                    "ATR_STOP": trial.suggest_float(f"{category}_atr_stop", 2.5, 3.5),
                    "RISK": trial.suggest_float(f"{category}_risk", 0.01, 0.015),
                    "KELLY": trial.suggest_float(f"{category}_kelly", 0.4, 0.6),
                    "SLOPE": trial.suggest_float(f"{category}_slope", -0.001, 0.002),
                    "MAX_STOP": trial.suggest_float(f"{category}_max_stop", 0.03, 0.08),
                    "SLOPE_MIN": trial.suggest_float(
                        f"{category}_slope_min", 0.01, 0.05
                    ),
                    "PREDICT_UP": trial.suggest_float(f"predict_up", 0.0, 0.005),
                }
            )

        # 3. 补充其他通用建议
        config.update(
            {
                "INIT_PT": trial.suggest_float(f"init_pt", 0.02, 0.05),
                "TP1": trial.suggest_float(f"tp1", 1.02, 1.06),
                "TREND_STAGE": trial.suggest_float(f"trend_stage", 0.12, 0.25),
                "MIN_STOP": trial.suggest_float(f"min_stop", 0.005, 0.02),
                "ATR_MULT": trial.suggest_float(f"atr_mult", 2.5, 4.5),
            }
        )

        return config
