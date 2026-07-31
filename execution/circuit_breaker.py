"""
回撤熔断器 — 模拟盘风控核心

功能:
  - 从 equity_log 读取权益历史, 计算当前回撤
  - 两级熔断: WARNING (回撤>8%) → HALT (回撤>12%)
  - HALT 状态下暂停所有买入, 只允许卖出/持有
  - 熔断状态持久化到 storage.config

用法:
  from execution.circuit_breaker import CircuitBreaker

  cb = CircuitBreaker()
  status = cb.check()          # 返回 "active" / "warning" / "halted"
  if cb.is_halted():
      print("熔断中, 禁止买入")

配置 (可在 config.yaml 中覆盖):
  max_drawdown_warning: 0.08   # 8% → WARNING
  max_drawdown_halt: 0.12      # 12% → HALT
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import storage


class CircuitBreaker:
    """回撤熔断器。"""

    def __init__(self,
                 warning_threshold: float = 0.08,
                 halt_threshold: float = 0.12):
        """
        Args:
          warning_threshold: 警告阈值 (回撤比例)
          halt_threshold: 熔断阈值 (回撤比例)
        """
        self.warning_threshold = warning_threshold
        self.halt_threshold = halt_threshold
        storage.init_db()

    def _get_peak_equity(self) -> float:
        """获取历史最高权益。"""
        initial = float(storage.get_config("initial_capital", "100000"))
        log = storage.get_equity_log(limit=99999)
        if not log:
            return initial
        return max(e["total_equity"] for e in log)

    def _get_current_equity(self) -> Optional[float]:
        """获取当前权益。"""
        log = storage.get_equity_log(limit=1)
        if not log:
            return None
        return log[0]["total_equity"]

    def calculate_drawdown(self) -> Tuple[float, float, float]:
        """
        计算当前回撤。

        Returns:
          (peak_equity, current_equity, drawdown_pct)
        """
        peak = self._get_peak_equity()
        current = self._get_current_equity()
        if current is None:
            initial = float(storage.get_config("initial_capital", "100000"))
            return peak, initial, 0.0

        dd = (current / peak - 1) if peak > 0 else 0.0
        return peak, current, dd

    def check(self) -> str:
        """
        检查熔断状态。

        Returns:
          "active" / "warning" / "halted"
        """
        # 如果已被手动暂停, 不覆盖
        current_state = storage.get_config("circuit_breaker", "active")
        if current_state == "halted_manual":
            return "halted_manual"

        peak, current, dd = self.calculate_drawdown()

        if dd <= -self.halt_threshold:
            new_state = "halted"
        elif dd <= -self.warning_threshold:
            new_state = "warning"
        else:
            new_state = "active"

        # 更新状态
        if new_state != current_state:
            storage.set_config("circuit_breaker", new_state)
            storage.set_config("circuit_breaker_dd", f"{dd:.4f}")
            storage.set_config("circuit_breaker_updated",
                             datetime.now().isoformat())

        return new_state

    def is_halted(self) -> bool:
        """是否处于熔断状态 (禁止买入)。"""
        state = self.check()
        return state in ("halted", "halted_manual")

    def is_warning(self) -> bool:
        """是否处于警告状态。"""
        return self.check() == "warning"

    def allow_buy(self) -> Tuple[bool, str]:
        """
        检查是否允许买入。

        Returns:
          (allowed, reason)
        """
        state = self.check()
        if state == "halted":
            return False, f"回撤熔断: 当前回撤超过{self.halt_threshold:.0%}"
        if state == "halted_manual":
            return False, "人工暂停"
        return True, "OK"

    def resume(self):
        """人工恢复 (解除熔断)。"""
        storage.set_config("circuit_breaker", "active")
        storage.set_config("circuit_breaker_updated",
                         datetime.now().isoformat())
        print("✅ 熔断已解除, 恢复正常交易")

    def halt_manual(self, reason: str = "人工暂停"):
        """人工暂停。"""
        storage.set_config("circuit_breaker", "halted_manual")
        storage.set_config("circuit_breaker_reason", reason)
        storage.set_config("circuit_breaker_updated",
                         datetime.now().isoformat())
        print(f"🛑 已人工暂停: {reason}")

    def get_status_report(self) -> dict:
        """获取完整状态报告。"""
        peak, current, dd = self.calculate_drawdown()
        state = self.check()
        initial = float(storage.get_config("initial_capital", "100000"))

        return {
            "state": state,
            "initial_capital": initial,
            "peak_equity": round(peak, 2),
            "current_equity": round(current, 2),
            "drawdown_pct": round(dd * 100, 2),
            "total_return_pct": round((current / initial - 1) * 100, 2),
            "warning_threshold": f"{self.warning_threshold:.0%}",
            "halt_threshold": f"{self.halt_threshold:.0%}",
            "last_updated": storage.get_config("circuit_breaker_updated", ""),
        }


# ════════════════════════════════════════
#  CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="回撤熔断器")
    parser.add_argument("--check", action="store_true", help="检查熔断状态")
    parser.add_argument("--resume", action="store_true", help="解除熔断")
    parser.add_argument("--halt", type=str, default=None, help="人工暂停 (原因)")
    args = parser.parse_args()

    cb = CircuitBreaker()

    if args.resume:
        cb.resume()
    elif args.halt:
        cb.halt_manual(args.halt)
    else:
        report = cb.get_status_report()
        print(f"熔断状态: {report['state']}")
        print(f"初始资金: ¥{report['initial_capital']:,.0f}")
        print(f"历史最高: ¥{report['peak_equity']:,.0f}")
        print(f"当前权益: ¥{report['current_equity']:,.0f}")
        print(f"总收益率: {report['total_return_pct']:+.2f}%")
        print(f"当前回撤: {report['drawdown_pct']:+.2f}%")
        print(f"警告阈值: {report['warning_threshold']}")
        print(f"熔断阈值: {report['halt_threshold']}")
