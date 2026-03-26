import shutil
import numpy as np
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from infra.core.runtime import RunMode
LIVE_FILE = Path("state/live_positions.yaml")
SIM_FILE = Path("state/sim_positions.yaml")
BACKUP_DIR = Path("state/backup")

BACKUP_DIR.mkdir(parents=True, exist_ok=True)


_last_hash = None

def _sanitize_data(obj):
    """递归转换字典/列表中的 NumPy 类型为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: _sanitize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_data(v) for v in obj]
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj

def _calc_position_hash(position_mgr):
    data = _sanitize_data(position_mgr.to_dict())
    text = yaml.safe_dump(
        data,
        sort_keys=True,
        allow_unicode=True
    )

    return hashlib.md5(text.encode()).hexdigest()


def persist_live_positions(position_mgr):
    """
    1. 检查数据是否改变
    2. 备份旧文件
    3. 原子写入新文件
    """
    global _last_hash

    new_hash = _calc_position_hash(position_mgr)

    if new_hash == _last_hash:
        return   # 没变化直接退出

    _last_hash = new_hash

    file = LIVE_FILE if position_mgr.run_mode == RunMode.LIVE else SIM_FILE

    # ===== 1. 备份旧版本 =====
    # if file.exists() and position_mgr.run_mode == RunMode.LIVE:
    #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     backup_file = BACKUP_DIR / f"live_positions_{ts}.yaml"
    #     shutil.copy2(file, backup_file)

    # ===== 2. 原子写入 =====
    tmp_file = file.with_suffix(".tmp")

    with open(tmp_file, "w", encoding="utf8") as f:
        yaml.safe_dump(_sanitize_data(position_mgr.to_dict()), f, allow_unicode=True)

    tmp_file.replace(file)