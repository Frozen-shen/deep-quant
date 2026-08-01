"""
logger.py — 统一日志模块

替代所有 print() 语句。提供结构化日志输出到 console + 文件。

Usage:
    from logger import get_logger
    log = get_logger("factor_scorer")
    log.info("因子计算完成: %d 只股票", n_stocks)
    log.warning("数据缺失: %s", symbol)
    log.error("回测失败: %s", str(e))
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

_LOGS_DIR_CREATED = False


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """获取命名 logger，输出到 console + 文件。"""
    global _LOGS_DIR_CREATED

    logger = logging.getLogger(f"quant.{name}")
    logger.setLevel(logging.DEBUG)

    # 已有 handler 则直接返回，避免重复添加
    if logger.handlers:
        return logger

    today = datetime.now()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # --- File handler ---
    if not _LOGS_DIR_CREATED:
        Path("logs").mkdir(exist_ok=True)
        _LOGS_DIR_CREATED = True

    file_handler = logging.FileHandler(
        f"logs/quant_{today:%Y%m%d}.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
