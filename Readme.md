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