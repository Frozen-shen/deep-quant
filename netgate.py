"""netgate.py — 数据获取离线守卫 (2026-08-16)。

约定: 网络访问只允许发生在获取阶段 (fetch_* 脚本 + daily_pipeline 数据更新步骤)。
训练/回测/IC 验证/信号生成在入口调用 set_offline_mode(True) 开启离线模式;
各 fetcher 的网络分支在离线模式下直接拒绝 (MinuteFetcher 回退本地/None,
其余抛 OfflineViolation)。获取脚本与 paper_executor 盘中路径不开启。

本模块零依赖 (防循环 import)。
"""

_OFFLINE = False


class OfflineViolation(RuntimeError):
    """离线模式下尝试网络获取数据 (违反数据获取纪律)。"""


def set_offline_mode(on: bool) -> None:
    """开启/关闭全局离线模式。训练/回测/信号入口置 True。"""
    global _OFFLINE
    _OFFLINE = bool(on)


def is_offline() -> bool:
    """当前是否处于离线模式。"""
    return _OFFLINE
