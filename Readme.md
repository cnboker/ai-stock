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