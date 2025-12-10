# ================================================================
# Chronos-T5 / 15分钟A股预测示例 (002137.SZ)
# ================================================================

import os
os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from sina import download_15m

from matplotlib import rcParams
from matplotlib import font_manager

font_path = '/usr/share/fonts/truetype/arphic/ukai.ttc'
font_prop = font_manager.FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()
rcParams['axes.unicode_minus'] = False  # 负号正常显示

# === 1 获取A股 15m 数据 ===
ticker = "002137.SZ"
df = download_15m(ticker)
df = df.dropna()
df["return"] = df["Close"].pct_change().fillna(0)

series = df["Close"].reset_index(drop=True)

# === 2 加载 Chronos 模型 ===
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",   # 你的显存可以跑 mini
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# === 3 推理预测未来 10 根 15m K线 ===
context = torch.tensor(series.values)
prediction_length = 5

forecast = pipeline.predict(context, prediction_length=prediction_length)
fforecast = pipeline.predict(context, prediction_length=prediction_length)
forecast_np = forecast.numpy().squeeze()  # 去掉多余维度

print(f"forecast_np.shape = {forecast_np.shape}")  # debug

#low, median, high = np.quantile(forecast_np, [0.1,0.5,0.9], axis=0)

# 取最后一个时间点的预测样本 (num_samples, prediction_length)
last_forecast = forecast_np[-1]  # shape = (20, 10)

# 对 20 个采样取分位数
low, median, high = np.quantile(last_forecast, [0.1,0.5,0.9], axis=0)  # shape = (10,)


future_index = pd.date_range(start=df.index[-1], periods=prediction_length+1)[1:]

plt.style.use('dark_background')  # 深色背景
plt.figure(figsize=(14,7))

# 历史价格线
plt.plot(df.index, df["Close"], label="历史价格", color="#00BFFF", linewidth=1.5)

# 预测中位数
plt.plot(future_index, median, label="预测中位数", color="#FFA500", linewidth=2)

# 分位数阴影
plt.fill_between(future_index, low, high, color="#FFA500", alpha=0.3)

# 网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 坐标轴字体大小和标签
plt.xlabel("时间", fontsize=12)
plt.ylabel("价格", fontsize=12)
plt.title(f"{ticker} — Chronos 未来 {prediction_length} 根 15m K线预测", fontsize=14)

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()