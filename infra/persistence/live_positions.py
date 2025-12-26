import shutil
import yaml
from datetime import datetime
from pathlib import Path

LIVE_FILE = Path("state/live_positions.yaml")
BACKUP_DIR = Path("state/backup")

BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def persist_live_positions(position_mgr):
    """
    1. 备份旧文件
    2. 原子写入新文件
    """

    # ===== 1. 备份旧版本 =====
    if LIVE_FILE.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = BACKUP_DIR / f"live_positions_{ts}.yaml"
        shutil.copy2(LIVE_FILE, backup_file)

    # ===== 2. 生成数据 =====
    data = {
        "account": position_mgr.account_name,
        "cash": position_mgr.cash,
        "positions": {},
    }

    for symbol, pos in position_mgr.positions.items():
        data["positions"][symbol] = {
            "direction": pos.direction,
            "size": pos.size,
            "entry_price": pos.entry_price,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "open_time": pos.open_time.isoformat(),
        }

    # ===== 3. 原子写入 =====
    tmp_file = LIVE_FILE.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)

    tmp_file.replace(LIVE_FILE)
