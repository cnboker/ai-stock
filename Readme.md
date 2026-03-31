```txt
           ┌─────────────┐
           │ Market Data │
           └─────┬───────┘
                 │
                 ▼
         ┌─────────────────┐
         │ Determine Regime │
         │ (neutral/bull/bear) │
         └─────┬───────────┘
               │
       ┌───────┴─────────┐
       │                 │
       ▼                 ▼
   regime = neutral   regime ≠ neutral
       │                 │
       │                 ▼
       │          ┌────────────────────┐
       │          │ Calculate slope     │
       │          │ slope_threshold ?   │
       │          └─────┬──────────────┘
       │                │
       │        slope*gate > threshold?
       │                │
       │        ┌───────┴─────────┐
       │        │                 │
       ▼        ▼                 ▼
  action=None  action=LONG     action=SHORT
  strength=0   strength= f(slope, gate) 

  逻辑说明

    Regime 判断

    neutral → 系统倾向 不操作

    bull / bear → 根据趋势和阈值决定多空

    Slope + Gate

    斜率 slope 表示趋势方向和强度

    gate 是调节系数

    系统只有当 slope*gate 超过内部阈值才会生成动作

    Action & Strength

    action=None → 不买卖

    strength=0 → 持仓变动力度为 0

    当条件满足时，strength 会随趋势强度变化

```

# 周期 (Timeframe),每天 K 线数量,1000 条数据可覆盖时长,是否满足 6 个月？
# 日线 (Daily),1 条,~1000 个交易日 (约 4 年),是 (过度覆盖)
# 1 小时 (1H),4 条,~250 个交易日 (约 12 个月),是 (完美契合)
# 30 分钟 (30M),8 条,~125 个交易日 (约 6 个月),是 (刚好满足)
# 15 分钟 (15M),16 条,~62.5 个交易日 (约 3 个月),否
# 5 分钟 (5M),48 条,~21 个交易日 (约 1 个月),否

# ETF 优化参数
```js
 config = {
        "MODEL_TH": trial.suggest_float("model_th", 0.42, 0.48),
        "SLOPE": trial.suggest_float("slope", -0.001, 0.005),
        "PREDICT_UP": trial.suggest_float("predict_up", 0.0, 0.005),
        "INIT_PT": trial.suggest_float("init_pt", 0.02, 0.04),
        "TREND_STAGE": trial.suggest_float("trend_pt", 0.05, 0.25),
        "TP1": trial.suggest_float("tp1", 1.03, 1.06),
        "TP2": trial.suggest_float("tp2", 1.1, 1.25),
        "KELLY": trial.suggest_float("kelly", 0.3, 0.5),
        "RISK": trial.suggest_float("risk", 0.01, 0.015),
        "ATR_STOP": trial.suggest_float("atr_stop", 2.5, 3.5),
        "MAX_STOP": trial.suggest_float("max_stop", 0.03, 0.10),
        "MIN_STOP": trial.suggest_float("min_stop", 0.01, 0.02),
        
        "STRENGTH_ALPHA": trial.suggest_float("strength_alpha", 1.2, 1.5),
        "CONFIRM_N": trial.suggest_float("confirm_n", 2, 5),
    }
```

### 全自动参数优化
    ```bash
    python -m optimize.auto_tune
    ```
