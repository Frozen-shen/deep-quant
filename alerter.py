"""
异常告警模块 — 多通道告警推送

支持通道:
  - 钉钉 Webhook (DINGTALK_WEBHOOK 环境变量)
  - Server酱微信推送 (WECHAT_SENDKEY 环境变量)
  - JSONL 文件 (data/alerts.jsonl, 始终启用)
  - Console 输出 (始终启用)

告警事件类型:
  - data_fetch_failed: 数据拉取失败 (连续2天)
  - signal_empty: 信号为空 (某天0信号)
  - equity_jump: 净值单日跳变 > 5%
  - circuit_breaker: 回撤熔断触发
  - ic_decay: 因子IC连续衰减

用法:
  from alerter import AlertManager

  am = AlertManager()
  am.send("data_fetch_failed", {"symbols": ["600519"], "error": "timeout"})

  # 或用于调度器中的批量检查
  from alerter import check_and_alert
  check_and_alert()
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── 配置文件路径 ──
ALERT_LOG = os.path.join(BASE_DIR, "data", "alerts.jsonl")
DEDUP_WINDOW = 1800  # 去重窗口: 30分钟内同类型事件不重复推送


def _load_env():
    """加载 .env 文件中的环境变量。"""
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


_load_env()


class AlertManager:
    """多通道告警管理器。"""

    def __init__(self):
        self.dingtalk_webhook = os.environ.get("DINGTALK_WEBHOOK", "")
        self.wechat_sendkey = os.environ.get("WECHAT_SENDKEY", "")
        self._recent_alerts: Dict[str, float] = {}  # key → last_sent_time

    def send(self, alert_type: str, detail: dict = None,
             level: str = "WARNING"):
        """
        发送告警。

        Args:
          alert_type: 告警类型
          detail: 详细信息 dict
          level: INFO / WARNING / ERROR
        """
        detail = detail or {}

        # 去重检查
        dedup_key = self._dedup_key(alert_type, detail)
        now = time.time()
        if dedup_key in self._recent_alerts:
            if now - self._recent_alerts[dedup_key] < DEDUP_WINDOW:
                return  # 30分钟内不重复推送
        self._recent_alerts[dedup_key] = now

        # 清理过期去重记录
        self._recent_alerts = {
            k: v for k, v in self._recent_alerts.items()
            if now - v < DEDUP_WINDOW * 2
        }

        # 构建消息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = self._format_message(alert_type, detail, level, timestamp)

        # ── 1. JSONL 文件 (始终启用) ──
        self._log_to_file(alert_type, detail, level, timestamp)

        # ── 2. Console ──
        prefix = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🚨"}.get(level, "📢")
        print(f"\n{prefix} [{level}] {alert_type}")
        print(f"   {message}")

        # ── 3. 钉钉 ──
        if self.dingtalk_webhook:
            self._send_dingtalk(message, level)

        # ── 4. Server酱 ──
        if self.wechat_sendkey:
            self._send_wechat(message, level)

    def _dedup_key(self, alert_type: str, detail: dict) -> str:
        """生成去重键。"""
        content = alert_type + json.dumps(detail, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()

    def _format_message(self, alert_type: str, detail: dict,
                        level: str, timestamp: str) -> str:
        """格式化告警消息。"""
        lines = [
            f"【{level}】{alert_type}",
            f"时间: {timestamp}",
        ]

        type_messages = {
            "data_fetch_failed": [
                f"数据拉取失败: {detail.get('error', '')}",
                f"失败股票: {detail.get('symbols', [])[:5]}",
                f"连续失败天数: {detail.get('consecutive_days', 0)}",
            ],
            "signal_empty": [
                f"交易信号为空: {detail.get('date', '')}",
                f"可能原因: {detail.get('reason', '未知')}",
            ],
            "equity_jump": [
                f"净值异常跳变: {detail.get('change_pct', 0):+.2f}%",
                f"日期: {detail.get('date', '')}",
                f"从 {detail.get('from', 0):,.0f} → {detail.get('to', 0):,.0f}",
            ],
            "circuit_breaker": [
                f"回撤熔断: {detail.get('state', '')}",
                f"当前回撤: {detail.get('drawdown', 0):.2f}%",
                f"当前权益: {detail.get('equity', 0):,.0f}",
            ],
            "ic_decay": [
                f"因子IC衰减: {detail.get('factor', '')}",
                f"当前IC: {detail.get('current_ic', 0):.4f}",
                f"基线IC: {detail.get('baseline_ic', 0):.4f}",
                f"连续下降周数: {detail.get('weeks', 0)}",
            ],
        }

        lines.extend(type_messages.get(alert_type, [json.dumps(detail, ensure_ascii=False)]))
        return "\n".join(lines)

    def _log_to_file(self, alert_type: str, detail: dict,
                     level: str, timestamp: str):
        """写入 JSONL 文件。"""
        os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
        entry = {
            "timestamp": timestamp,
            "type": alert_type,
            "level": level,
            "detail": detail,
        }
        with open(ALERT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _send_dingtalk(self, message: str, level: str):
        """通过钉钉 Webhook 推送。"""
        try:
            import requests
            payload = {
                "msgtype": "text",
                "text": {
                    "content": f"[DeepQuant {level}]\n{message}"
                }
            }
            requests.post(self.dingtalk_webhook, json=payload, timeout=10)
        except Exception as e:
            print(f"  钉钉推送失败: {e}")

    def _send_wechat(self, message: str, level: str):
        """通过 Server酱 推送。"""
        try:
            import requests
            url = f"https://sctapi.ftqq.com/{self.wechat_sendkey}.send"
            requests.post(url, data={
                "title": f"[DeepQuant {level}] 告警",
                "desp": message,
            }, timeout=10)
        except Exception as e:
            print(f"  微信推送失败: {e}")


# ════════════════════════════════════════
#  批量检查函数
# ════════════════════════════════════════

def check_and_alert():
    """
    执行所有异常检查并告警。
    由 scheduler.py 的每日流程调用。
    """
    am = AlertManager()
    from datetime import datetime, timedelta

    # ── 1. 检查数据更新日志 ──
    log_path = os.path.join(BASE_DIR, "data", "update_log.jsonl")
    if os.path.exists(log_path):
        try:
            recent = []
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    recent.append(json.loads(line))
            # 只看最近2天
            cutoff = (datetime.now() - timedelta(days=2)).isoformat()
            recent_errors = [
                r for r in recent
                if r.get("level") == "ERROR" and r["timestamp"] > cutoff
            ]
            if len(recent_errors) >= 2:
                am.send("data_fetch_failed", {
                    "error": "连续2天数据更新失败",
                    "recent_errors": recent_errors[-3:],
                }, "ERROR")
        except Exception:
            pass

    # ── 2. 检查信号是否为空 ──
    signal_path = os.path.join(BASE_DIR, "data", "paper_signals.jsonl")
    if os.path.exists(signal_path):
        try:
            lines = []
            with open(signal_path, encoding="utf-8") as f:
                for line in f:
                    lines.append(json.loads(line))
            if lines:
                last = lines[-1]
                last_date = last.get("signal_date", "")
                today = datetime.now().strftime("%Y-%m-%d")
                if last_date >= today:
                    if not last.get("buy") and not last.get("sell"):
                        am.send("signal_empty", {
                            "date": last_date,
                            "reason": "策略无信号输出",
                        })
        except Exception:
            pass

    # ── 3. 检查净值跳变 ──
    try:
        import storage
        storage.init_db()
        log = storage.get_equity_log(limit=2)
        if len(log) >= 2:
            prev = log[1]["total_equity"]
            curr = log[0]["total_equity"]
            if prev > 0:
                change = (curr / prev - 1)
                if abs(change) > 0.05:
                    am.send("equity_jump", {
                        "date": log[0]["date"],
                        "from": prev,
                        "to": curr,
                        "change_pct": change * 100,
                    }, "WARNING")
    except Exception:
        pass

    # ── 4. 检查熔断状态 ──
    try:
        from execution.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        state = cb.check()
        if state in ("halted", "halted_manual"):
            report = cb.get_status_report()
            am.send("circuit_breaker", {
                "state": state,
                "drawdown": report["drawdown_pct"],
                "equity": report["current_equity"],
            }, "ERROR" if state == "halted" else "WARNING")
    except Exception:
        pass

    # ── 5. IC 衰减检查 (简化版, 完整版在 run_ic_monitor.py) ──
    ic_monitor_path = os.path.join(BASE_DIR, "data", "ic_monitor.json")
    if os.path.exists(ic_monitor_path):
        try:
            with open(ic_monitor_path, encoding="utf-8") as f:
                ic_history = json.load(f)
            # 检查最新一条
            if ic_history:
                last_ic = ic_history[-1]
                if last_ic.get("decay_warning"):
                    am.send("ic_decay", last_ic)
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="异常告警")
    parser.add_argument("--test", action="store_true", help="发送测试告警")
    parser.add_argument("--check", action="store_true", help="运行批量检查")
    args = parser.parse_args()

    am = AlertManager()
    if args.test:
        print("发送测试告警...")
        am.send("signal_empty", {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "reason": "这是一条测试告警",
        })
        print("✅ 测试告警已发送")
    elif args.check:
        check_and_alert()
    else:
        print(f"钉钉 Webhook: {'✅ 已配置' if am.dingtalk_webhook else '❌ 未配置'}")
        print(f"Server酱:    {'✅ 已配置' if am.wechat_sendkey else '❌ 未配置'}")
        print(f"JSONL 文件:  {ALERT_LOG}")
        print(f"\n使用 --test 发送测试告警")
        print(f"使用 --check 运行批量检查")
