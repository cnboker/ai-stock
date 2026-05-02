import csv
from functools import partial

from yaml import loader
from infra.utils.hot_reloader import HotReloader
from position.position_manager import PositionManager

def reload_watchlist(file_path, pos_mgr:PositionManager):
    tickers = load_watchlist(file_path)
    print(f"🔄 重新加载观察池: {tickers}"   )
    # 此时已经在外层的 HotReloader 里加过锁了，直接操作即可
    pos_mgr.load_watchlist_from_csv(tickers)
    print(f"✅ 观察池已同步: {len(tickers)} 只")

def live_watchlist_hot_load(pos_mgr:PositionManager):
    # 1. 先手动执行一次，确保初始数据到位
    reload_watchlist("state/watchlist.csv", pos_mgr)
    # 2. 在启动时进行“注入”
    # 假设你已经实例化了 position_mgr
    # partial 会返回一个“新函数”，这个新函数只剩 file_path 一个参数了
    bound_reload_func = partial(reload_watchlist, pos_mgr=pos_mgr)

    # 使用包装器
    watchlist_loader = HotReloader("state/watchlist.csv", reload_func=bound_reload_func)
    watchlist_loader.start()
    return watchlist_loader

def load_watchlist(file_path):
    tickers = []
    print(f"📂 加载观察池文件: {file_path}" )
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        tickers = [row[0].strip() for row in reader if row and row[0]]
    return tickers