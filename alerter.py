"""
告警通知模块 — 多渠道推送 (微信/钉钉/邮件/本地日志)

用法:
    alerter = Alerter(wechat_sendkey="SCT123456...")
    alerter.send("交易信号", "BUY 01810 @35.20, 置信度0.85")
    alerter.signal_alert(decision)   # 专用: 信号通知
    alerter.daily_summary(summary)   # 专用: 日报
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, Dict


class Alerter:
    """
    多渠道告警通知。

    渠道:
    - wechat:   Server酱 Turbo (https://sc3.ft07.com)
    - dingtalk: 钉钉机器人 Webhook
    - email:    SMTP (smtplib)
    - log:      本地日志文件 (始终启用，兜底)

    优先级: wechat > dingtalk > email (任一成功即停止)
    """

    def __init__(
        self,
        wechat_sendkey: Optional[str] = None,
        dingtalk_webhook: Optional[str] = None,
        email_config: Optional[Dict] = None,
        log_file: Optional[str] = None,
    ):
        self.wechat_sendkey = wechat_sendkey or os.environ.get("WECHAT_SENDKEY")
        self.dingtalk_webhook = dingtalk_webhook or os.environ.get("DINGTALK_WEBHOOK")
        self.email_config = email_config
        self.log_file = log_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "alerts.log"
        )
        self._last_sent: Dict[str, str] = {}  # 去重: key → 上次发送时间

    def send(self, title: str, content: str, dedup_key: str = "",
             dedup_minutes: int = 30):
        """
        发送通知。

        参数
        ----
        title, content : str
        dedup_key : str
            去重键，相同 key 在 dedup_minutes 分钟内不重复发送
        dedup_minutes : int
        """
        # 去重
        if dedup_key:
            now = datetime.now()
            last = self._last_sent.get(dedup_key)
            if last:
                last_dt = datetime.fromisoformat(last)
                if (now - last_dt).total_seconds() < dedup_minutes * 60:
                    return
            self._last_sent[dedup_key] = now.isoformat()

        # 尝试多渠道
        sent = False

        if self.wechat_sendkey:
            sent = self._send_wechat(title, content)
        if not sent and self.dingtalk_webhook:
            sent = self._send_dingtalk(title, content)
        if not sent and self.email_config:
            sent = self._send_email(title, content)

        # 日志兜底
        self._log(title, content, sent)

    # ---------- 专用快捷方法 ----------
    def signal_alert(self, decision):
        """发送交易信号通知。"""
        from signal_hub import HubDecision
        if not decision.should_trade:
            return

        title = f"📊 {decision.action} {decision.symbol}"
        content = (
            f"信号: {decision.action}\n"
            f"标的: {decision.symbol}\n"
            f"日期: {decision.date}\n"
            f"置信度: {decision.confidence:.2f}\n"
            f"理由: {decision.reason}\n"
            f"策略: {', '.join(s.strategy for s in decision.signals)}"
        )
        self.send(title, content, dedup_key=f"{decision.symbol}_{decision.date}")

    def daily_summary(self, summary: Dict, close_prices: Dict):
        """发送每日汇总。"""
        equity = summary.get("total_equity", 0)
        init = summary.get("initial_capital", 100000)
        ret = (equity / init - 1) * 100 if init > 0 else 0
        pos_count = summary.get("position_count", 0)

        title = f"📈 日报 {datetime.now().strftime('%Y-%m-%d')}"
        lines = [
            f"总权益: {equity:,.0f} ({ret:+.2f}%)",
            f"持仓数: {pos_count}",
        ]
        for p in summary.get("positions", []):
            lines.append(
                f"  {p['symbol']}: {p['qty']}股 "
                f"@{p['avg_cost']:.2f} → {p.get('current_price', '?')} "
                f"({p.get('pnl_pct', 0):+.2f}%)"
            )
        content = "\n".join(lines)
        self.send(title, content, dedup_key=f"daily_{datetime.now().strftime('%Y%m%d')}",
                  dedup_minutes=1440)

    def error_alert(self, module: str, error: str):
        """发送异常告警。"""
        title = f"⚠️ 异常: {module}"
        content = f"模块: {module}\n时间: {datetime.now()}\n错误: {error}"
        self.send(title, content)

    # ---------- 内部实现 ----------
    def _send_wechat(self, title: str, content: str) -> bool:
        """Server酱 Turbo 推送。"""
        if not self.wechat_sendkey:
            return False
        try:
            resp = requests.post(
                f"https://sc3.ft07.com/{self.wechat_sendkey}.send",
                data={"text": title, "desp": content},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"[Alerter] 微信通知失败: {e}")
            return False

    def _send_dingtalk(self, title: str, content: str) -> bool:
        """钉钉机器人推送。"""
        if not self.dingtalk_webhook:
            return False
        try:
            resp = requests.post(
                self.dingtalk_webhook,
                json={
                    "msgtype": "text",
                    "text": {"content": f"{title}\n{content}"},
                },
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"[Alerter] 钉钉通知失败: {e}")
            return False

    def _send_email(self, title: str, content: str) -> bool:
        """SMTP 邮件（暂未实现，保留接口）。"""
        return False

    def _log(self, title: str, content: str, sent: bool):
        """写入本地日志。"""
        status = "✓" if sent else "✗"
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {status} {title} | {content[:100]}"
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass  # 日志写入失败不阻塞
