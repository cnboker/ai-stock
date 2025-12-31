# config.py
import os
import torch

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
]
ticker_name_map = {t["code"]:t["name"] for t in ALL_TICKERS}
COLORS = ["#00ff00", "#ff8800", "#00cccc", "#ff66cc"]
HISTORY_FILE = "data/prediction_history.pkl"
MODEL_NAME = "chronos-2"
ENABLE_LIVE_PERSIST = True
# ================= CUDA 优化 =================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
