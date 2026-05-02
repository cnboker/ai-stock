import csv
from functools import partial
import threading

from yaml import loader
from infra.utils.hot_reloader import HotReloader
from position.position_manager import PositionManager

def reload_watchlist(file_path, pos_mgr:PositionManager):
    tickers = load_watchlist(file_path)
    # 此时已经在外层的 HotReloader 里加过锁了，直接操作即可
    pos_mgr.load_watchlist_from_csv(tickers)
    print(f"✅ 观察池已同步: {len(tickers)} 只")

def live_watchlist_hot_load(pos_mgr:PositionManager):
    # 创建一个同步事件
    load_event = threading.Event()

    def wrapped_reload(file_path):
        reload_watchlist(file_path, pos_mgr)
        load_event.set()  # 只要加载成功一次，就发出信号

    bound_reload_func = partial(wrapped_reload)

    # 使用包装器
    watchlist_loader = HotReloader("state/watchlist.csv", reload_func=bound_reload_func)
    watchlist_loader.start()

    # --- 关键改进：等待信号 ---
    # 等待最多 5 秒，如果文件加载完成，提前解除阻塞
    is_loaded = load_event.wait(timeout=5.0)
    if not is_loaded:
        print("⚠️ 警告: 观察池加载超时，可能使用了空数据")

    return watchlist_loader

def load_watchlist(file_path):
    tickers = []
    print(f"📂 加载观察池文件: {file_path}" )
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        tickers = [row[0].strip() for row in reader if row and row[0]]
    return tickers