# log.py
import json
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.logging import RichHandler
from rich.traceback import install

# ==========================
# 美化 traceback（显示局部变量）
# ==========================
install(show_locals=True)

# ==========================
# 读取配置
# ==========================
CONFIG_PATH = Path("config/logger.yaml")

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.yaml not found")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["logging"]

_cfg = load_config()

# ==========================
# 基础参数
# ==========================
LOG_LEVEL = getattr(logging, _cfg.get("level", "INFO").upper())
LOG_DIR = Path(_cfg.get("log_dir", "logs"))
LOG_DIR.mkdir(exist_ok=True)

FILE_PREFIX = _cfg.get("file_prefix", "app")
TODAY = datetime.now().strftime("%Y-%m-%d")

# ==========================
# 避免重复初始化
# ==========================
_initialized = False

def setup_logging():
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.handlers.clear()

    # ---------- Console ----------
    if _cfg["console"]["enabled"]:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        console_handler.setLevel(LOG_LEVEL)
        root.addHandler(console_handler)

    # ---------- File ----------
    if _cfg["file"]["enabled"]:
        log_file = LOG_DIR / f"{FILE_PREFIX}-{TODAY}.log"

        if _cfg["file"]["rotate"] == "size":
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=_cfg["file"]["max_size_mb"] * 1024 * 1024,
                backupCount=_cfg["file"]["backup_count"],
                encoding="utf-8",
            )
        else:
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when="midnight",
                backupCount=_cfg["file"]["backup_count"],
                encoding="utf-8",
            )

        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        ))
        file_handler.setLevel(LOG_LEVEL)
        root.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


# log.py（底部追加）

def signal_log(msg: Any, logger_name: str = "signal"):
    if isinstance(msg, (dict, list)):
        msg = json.dumps(msg, indent=2, ensure_ascii=False)
    get_logger(logger_name).info(f"[cyan][SIGNAL][/cyan] {msg}")

def order_log(msg: Any, logger_name: str = "order"):
    if isinstance(msg, (dict, list)):
        msg = json.dumps(msg, indent=2, ensure_ascii=False)
    get_logger(logger_name).info(f"[green][ORDER][/green] {msg}")


def risk_log(msg: Any, *, title: str = "RISK"):
    if isinstance(msg, (dict, list)):
        msg = json.dumps(msg, indent=2, ensure_ascii=False)
    get_logger("risk").warning(f"[red][{title}][/red]\n{msg}")