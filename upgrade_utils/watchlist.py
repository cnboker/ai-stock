import csv
import yaml
import os

from data.ticker_registry import TickerRegistry

def sync_account_and_watchlist(yaml_path: str = "./state/live_positions.yaml", watchlist_path: str = "./state/watchlist.csv"):
    """
    1. 读取 account.yaml
    2. 将 size: 0 的移入 watchlist.csv
    3. 将 size > 0 的重新写回 account.yaml
    """
    if not os.path.exists(yaml_path):
        print(f"❌ 错误: 找不到文件 {yaml_path}")
        return

    # --- 1. 读取 YAML 文件 ---
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            account_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"❌ 解析 YAML 失败: {e}")
            return

    if not account_data or "positions" not in account_data:
        return
    
    real_positions = {}
    new_watch_items = []

    # 1. 遍历原始数据进行分类
    for ticker, info in account_data["positions"].items():
        if info["size"] > 0:
            real_positions[ticker] = info
        else:
            # size 为 0，提取代码和名称（假设你有一个获取名称的工具）
            # --- 实战用法 ---
            # 示例 1: 直接调用类方法
            name = TickerRegistry.get_ticker_name(ticker)

            new_watch_items.append({"code": ticker, "name": name})

    # 2. 更新账户文件 (只保留真实持仓)
    account_data["positions"] = real_positions
    with open(yaml_path, "w") as f:
        yaml.dump(account_data, f)

    # 3. 追加到观察池 CSV (需去重)
    update_watchlist_csv(new_watch_items, watchlist_path)

def update_watchlist_csv(items, path):
    # 读取现有的，防止重复写入
    existing_codes = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_codes = {row[0] for row in reader if row}

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for item in items:
            if item["code"] not in existing_codes:
                writer.writerow([item["code"], item["name"]])


if __name__ == "__main__":
    sync_account_and_watchlist()