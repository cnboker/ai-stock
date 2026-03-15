# config.py
import os
from model_torch import is_torch_available

# ================= 环境变量（必须最早） =================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= 全局参数 =================
PREDICTION_LENGTH = 10
TICKER_PERIOD = 3
UPDATE_INTERVAL_SEC = 60 * TICKER_PERIOD
ALL_TICKERS = [
    {"code": "sh600446", "name": "金证股份"},
    {"code": "sz300142", "name": "沃生生物"},
    {"code": "sh600771", "name": "广誉远"},
    {"code": "sz002137", "name": "实益达"},
    {"code": "sz000617", "name": "中油资本"},
    {"code": "sz159908", "name": "创业板etf"}
]
ticker_name_map = {t["code"]:t["name"] for t in ALL_TICKERS}
COLORS = ["#00ff00", "#ff8800", "#00cccc", "#ff66cc","#fa66ec","#f866c0","#a866a0","#b866b0","#c816c0","#f566c0"]
HISTORY_FILE = "data/prediction_history.pkl"
MODEL_NAME = "chronos-2"
#MODEL_NAME = "chronos-t5-large"
ENABLE_LIVE_PERSIST = True
# ================= CUDA 优化 =================

if is_torch_available():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

"""
AI信号必须 >0.6
趋势 slope >0.002
止损 = 2ATR
移动止盈 = 2.5ATR
止损后 20 bar 不再开仓
最多加仓3次
"""

MODEL_LONG_THRESHOLD = 0.55
TREND_SLOPE_THRESHOLD = 0.001

ATR_STOP_MULT = 2.0
ATR_TRAIL_MULT = 2.5

STOP_LOSS_COOLDOWN = 20
MAX_POSITION = 3