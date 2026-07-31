"""
数据质量校验模块 — 增量更新后自动校验数据完整性

用法:
  from data.validator import validate_all, validate_symbol
  report = validate_all()             # 校验所有缓存股票
  report = validate_symbol("600519")  # 校验单只股票

校验项目:
  1. 空数据/行数异常 (akshare偶尔返回空或重复行)
  2. 价格跳变检测 (除权日: 单日涨跌>20%且volume>0)
  3. 停牌检测 (volume==0连续天数)
  4. 日期连续性与重叠 (跳空>5天 → 告警)
  5. OHLC 合理性 (high>=low, high>=open/close, low<=open/close)
  6. 重复日期检测

输出: data/dq_report_YYYYMMDD.json
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path = __import__('sys')
sys_path.path.insert(0, BASE_DIR)

from data_cache import get_cached_symbols, CACHE_DIR

REPORT_DIR = os.path.join(BASE_DIR, "data")


class DataQualityReport:
    """数据质量报告容器。"""

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.total_symbols = 0
        self.checked_symbols = 0
        self.errors = []          # 严重错误 (数据不可用)
        self.warnings = []        # 警告 (需关注但不致命)
        self.symbol_details = {}  # {symbol: {issues: [...], row_count: int, date_range: str}}
        self.summary = {}

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "total_symbols": self.total_symbols,
            "checked_symbols": self.checked_symbols,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings[:50],  # 截断
            "summary": self.summary,
        }

    def save(self, filename: str = None):
        """保存报告到 JSON 文件。"""
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"dq_report_{date_str}.json"
        path = os.path.join(REPORT_DIR, filename)
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path


def validate_ohlc(df: pd.DataFrame) -> List[str]:
    """校验 OHLC 数据合理性。"""
    issues = []

    # High >= Low
    bad = df[df["high"] < df["low"]]
    if len(bad) > 0:
        dates = bad["date"].dt.strftime("%Y-%m-%d").tolist()[:5]
        issues.append(f"high<low: {len(bad)}行, 日期: {dates}")

    # High >= max(open, close)
    bad = df[df["high"] < df[["open", "close"]].max(axis=1)]
    if len(bad) > 0:
        issues.append(f"high<max(open,close): {len(bad)}行")

    # Low <= min(open, close)
    bad = df[df["low"] > df[["open", "close"]].min(axis=1)]
    if len(bad) > 0:
        issues.append(f"low>min(open,close): {len(bad)}行")

    # 零或负价格
    for col in ["open", "high", "low", "close"]:
        bad = df[df[col] <= 0]
        if len(bad) > 0:
            issues.append(f"{col}<=0: {len(bad)}行")

    return issues


def validate_price_jumps(df: pd.DataFrame, threshold: float = 0.20) -> List[str]:
    """
    检测价格跳变 (除权日信号)。

    规则: abs(return) > threshold 且 volume > 0 → 可能是除权
          abs(return) > threshold 且 volume == 0 → 可能是停牌复牌
    """
    issues = []
    if len(df) < 2:
        return issues

    df = df.sort_values("date").reset_index(drop=True)
    rets = df["close"].pct_change()

    for i in range(1, len(df)):
        r = rets.iloc[i]
        if pd.isna(r):
            continue
        if abs(r) > threshold:
            vol = df["volume"].iloc[i]
            date_str = df["date"].iloc[i].strftime("%Y-%m-%d")
            if vol > 0:
                issues.append(
                    f"price_jump: {date_str} 涨跌{r:+.1%} vol={vol:.0f} (疑似除权)")
            else:
                issues.append(
                    f"price_jump_no_vol: {date_str} 涨跌{r:+.1%} vol=0 (疑似停牌复牌)")

    return issues


def validate_suspension(df: pd.DataFrame, max_suspend_days: int = 60) -> List[str]:
    """
    检测停牌天数是否异常。

    - 连续 volume==0 天数 > max_suspend_days → 告警 (可能是数据缺失)
    - 排除正常停牌 (< max_suspend_days)
    """
    issues = []
    if len(df) < 2:
        return issues

    df = df.sort_values("date").reset_index(drop=True)

    # 计算连续 volume==0 的天数
    is_zero = (df["volume"] == 0).astype(int)
    # 检测连续段
    groups = (is_zero.diff() != 0).cumsum()
    zero_groups = df[is_zero == 1].groupby(groups[is_zero == 1])

    for _, group in zero_groups:
        if len(group) >= max_suspend_days:
            start = group["date"].iloc[0].strftime("%Y-%m-%d")
            end = group["date"].iloc[-1].strftime("%Y-%m-%d")
            issues.append(
                f"long_suspension: {start}~{end} 连续{len(group)}天停牌 (可能数据缺失)")

    return issues


def validate_date_continuity(df: pd.DataFrame, max_gap_days: int = 5) -> List[str]:
    """
    检测日期连续性。

    - 相邻交易日间隔 > max_gap_days → 告警
    - 注意: 正常周末/节假日间隔约2-5天 (含长假7天)
    """
    issues = []
    if len(df) < 2:
        return issues

    df = df.sort_values("date").reset_index(drop=True)
    gaps = df["date"].diff().dropna()

    for i, gap in enumerate(gaps):
        if gap.days > max_gap_days:
            date_str = df["date"].iloc[i + 1].strftime("%Y-%m-%d")
            issues.append(
                f"date_gap: {date_str} 距前一日 {gap.days}天 (可能数据缺失)")

    return issues


def validate_duplicates(df: pd.DataFrame) -> List[str]:
    """检测重复日期。"""
    issues = []
    dups = df[df.duplicated(subset=["date"], keep=False)]
    if len(dups) > 0:
        dates = dups["date"].dt.strftime("%Y-%m-%d").unique().tolist()[:5]
        issues.append(f"duplicate_dates: {len(dups)}行重复, 日期: {dates}")
    return issues


def validate_volume_anomaly(df: pd.DataFrame, threshold: float = 10.0) -> List[str]:
    """
    检测成交量异常。

    - 成交量突然放大 > threshold 倍 (相对20日均量) → 告警
    - 可能正常 (事件驱动), 但值得标记
    """
    issues = []
    if len(df) < 22:
        return issues

    df = df.sort_values("date").reset_index(drop=True)
    avg_vol_20 = df["volume"].rolling(20).mean()
    vol_ratio = df["volume"] / avg_vol_20

    for i in range(20, len(df)):
        if pd.notna(vol_ratio.iloc[i]) and vol_ratio.iloc[i] > threshold:
            date_str = df["date"].iloc[i].strftime("%Y-%m-%d")
            issues.append(
                f"volume_spike: {date_str} 成交量{vol_ratio.iloc[i]:.1f}倍均量")

    return issues


def validate_symbol(symbol: str) -> Tuple[Dict, List[str], List[str]]:
    """
    校验单只股票。

    Returns:
      (details_dict, errors, warnings)
    """
    path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    details = {"symbol": symbol, "row_count": 0, "date_range": ""}
    errors = []
    warnings = []

    if not os.path.exists(path):
        errors.append(f"missing_cache: {symbol}.parquet 不存在")
        details["error"] = "cache_missing"
        return details, errors, warnings

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        errors.append(f"read_error: {symbol} 读取失败: {e}")
        details["error"] = "read_error"
        return details, errors, warnings

    if len(df) == 0:
        errors.append(f"empty_data: {symbol} 数据为空")
        details["error"] = "empty"
        return details, errors, warnings

    df["date"] = pd.to_datetime(df["date"])
    details["row_count"] = len(df)
    details["date_range"] = f"{df['date'].min().date()} ~ {df['date'].max().date()}"

    # ── 执行各项校验 ──
    all_checks = [
        ("OHLC合理性", validate_ohlc(df)),
        ("价格跳变", validate_price_jumps(df)),
        ("停牌检测", validate_suspension(df)),
        ("日期连续性", validate_date_continuity(df)),
        ("重复日期", validate_duplicates(df)),
        ("成交量异常", validate_volume_anomaly(df)),
    ]

    issues_list = []
    for check_name, check_issues in all_checks:
        if check_issues:
            for issue in check_issues:
                full_issue = f"[{check_name}] {symbol}: {issue}"
                issues_list.append(full_issue)

    # 数据行数太少 (< 100行)
    if details["row_count"] < 100:
        warnings.append(f"low_row_count: {symbol} 仅{details['row_count']}行数据")

    details["issue_count"] = len(issues_list)
    details["issues"] = issues_list[:10]  # 保留前10个

    return details, errors, issues_list


def validate_all(symbols: List[str] = None) -> DataQualityReport:
    """
    校验所有缓存股票。

    Args:
      symbols: 股票列表, None=全部缓存

    Returns:
      DataQualityReport
    """
    if symbols is None:
        symbols = get_cached_symbols()

    report = DataQualityReport()
    report.total_symbols = len(symbols)

    print(f"数据质量校验: {len(symbols)} 只股票...")

    for i, sym in enumerate(symbols):
        details, errs, issues = validate_symbol(sym)
        report.checked_symbols += 1

        if errs:
            report.errors.extend(errs)
        if issues:
            # 区分 errors 和 warnings
            for issue in issues:
                if any(kw in issue for kw in ["empty_data", "missing_cache",
                                                "read_error", "high<low"]):
                    report.errors.append(issue)
                else:
                    report.warnings.append(issue)

        report.symbol_details[sym] = details

        if (i + 1) % 100 == 0:
            print(f"  已校验: {i+1}/{len(symbols)}")

    # ── 汇总 ──
    total_issues = sum(
        d.get("issue_count", 0) for d in report.symbol_details.values())

    report.summary = {
        "symbols_with_errors": sum(1 for d in report.symbol_details.values()
                                   if d.get("error")),
        "symbols_with_warnings": sum(1 for d in report.symbol_details.values()
                                     if d.get("issue_count", 0) > 0),
        "total_issues": total_issues,
        "avg_row_count": int(np.mean([d.get("row_count", 0)
                                       for d in report.symbol_details.values()])),
    }

    # ── 输出报告 ──
    report_path = report.save()
    print(f"\n数据质量报告已保存: {report_path}")
    print(f"  总股票: {report.total_symbols}")
    print(f"  错误: {len(report.errors)}")
    print(f"  警告: {len(report.warnings)}")
    print(f"  通过: {'✅' if report.passed else '❌ 发现问题'}")

    if report.errors:
        print(f"\n  严重错误 (前10):")
        for e in report.errors[:10]:
            print(f"    ❌ {e}")

    if report.warnings:
        print(f"\n  警告 (前5):")
        for w in report.warnings[:5]:
            print(f"    ⚠️ {w}")

    return report


# ════════════════════════════════════════
#  CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数据质量校验")
    parser.add_argument("--symbol", type=str, default=None,
                       help="校验单只股票")
    parser.add_argument("--output", type=str, default=None,
                       help="报告输出路径")
    args = parser.parse_args()

    if args.symbol:
        details, errs, issues = validate_symbol(args.symbol)
        print(f"\n{args.symbol}:")
        print(f"  行数: {details['row_count']}")
        print(f"  日期: {details['date_range']}")
        if errs:
            print(f"  错误: {errs}")
        if issues:
            print(f"  问题: {issues[:10]}")
    else:
        validate_all()
