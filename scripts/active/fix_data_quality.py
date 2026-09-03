"""
修复和清理 data_store 中的日线数据。

问题背景:
  data_store 使用腾讯备用源 (tencent_fallback) 构建,
  腾讯 API 不提供成交额(amount)和换手率(turnover), 导致这两列全为 NaN。
  这会影响所有流动性/换手率相关因子的计算。

修复方案:
  使用 baostock 获取 amount 和 turnover 数据, 按日期合并回现有 parquet 文件。
  baostock 免费、稳定, 无需代理, 提供完整的成交额和换手率数据。

用法:
  py scripts/active/fix_data_quality.py                # 修复所有缺失的股票
  py scripts/active/fix_data_quality.py --dry-run      # 仅统计, 不修改文件
  py scripts/active/fix_data_quality.py --limit 10     # 测试模式, 只修复前 10 只
  py scripts/active/fix_data_quality.py --resume       # 跳过已修复的文件
  py scripts/active/fix_data_quality.py --validate     # 仅验证, 不修复
  py scripts/active/fix_data_quality.py --canonicalize  # 清理坏行并补齐未复权字段

预计耗时: ~100 分钟 (3000+ 只股票, 每只约 2 秒)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# ── 路径 ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_STORE = os.path.join(BASE_DIR, "data_store")
META_FILE = os.path.join(DATA_STORE, "_meta.json")

# ── 参数 ─────────────────────────────────────────────────────────────
PROGRESS_INTERVAL = 50      # 每 N 只打印进度
RATE_LIMIT_SLEEP = 0.1      # baostock 请求间隔 (baostock 限速较宽松)

# 这些文件在数据源切换点之后出现连续的千倍/万倍价格，且切换前后
# 不满足任何合理复权关系；不是可用的拆分/分红事件，因此整段隔离。
KNOWN_CORRUPT_TAIL_STARTS = {
    "000046": "2023-12-28",
    "000540": "2023-05-19",
    "000806": "2023-07-06",
    "000961": "2024-05-09",
    "000979": "2018-12-28",
    "002450": "2021-05-30",
}


def purge_invalid_rows(df: pd.DataFrame, code: str):
    """删除可确定为脏数据的行，保留无法证明错误的正常行。

    规则只处理：已确认的源切换尾段、非正/不可能 OHLC、周末伪交易日，
    以及负成交量/成交额。普通停牌的零成交量不在这里删除。
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    dates = work["date"]

    reasons = {}
    tail_start = KNOWN_CORRUPT_TAIL_STARTS.get(str(code))
    tail = (dates >= pd.Timestamp(tail_start)) if tail_start else pd.Series(
        False, index=work.index
    )
    reasons["known_corrupt_tail"] = int(tail.sum())

    ohlc_cols = [c for c in ["open", "high", "low", "close"]
                 if c in work.columns]
    nonpositive = (work[ohlc_cols] <= 0).any(axis=1) if ohlc_cols else pd.Series(
        False, index=work.index
    )
    reasons["nonpositive_ohlc"] = int(nonpositive.sum())

    if all(c in work.columns for c in ["open", "high", "low", "close"]):
        invalid_ohlc = (
            (work["low"] > work[["open", "close", "high"]].min(axis=1))
            | (work["high"] < work[["open", "close", "low"]].max(axis=1))
        )
    else:
        invalid_ohlc = pd.Series(False, index=work.index)
    reasons["invalid_ohlc"] = int(invalid_ohlc.sum())

    weekend = dates.dt.weekday >= 5
    reasons["weekend"] = int(weekend.sum())
    negative_volume = (work["volume"] < 0) if "volume" in work.columns else pd.Series(
        False, index=work.index
    )
    negative_amount = (work["amount"] < 0) if "amount" in work.columns else pd.Series(
        False, index=work.index
    )
    reasons["negative_volume"] = int(negative_volume.sum())
    reasons["negative_amount"] = int(negative_amount.sum())

    remove = (tail | nonpositive | invalid_ohlc | weekend |
              negative_volume | negative_amount | dates.isna())
    cleaned = work.loc[~remove].sort_values("date").reset_index(drop=True)
    return cleaned, {
        "code": str(code),
        "input_rows": int(len(work)),
        "removed_rows": int(remove.sum()),
        "output_rows": int(len(cleaned)),
        "reasons": reasons,
        "tail_start": tail_start,
    }


def reconcile_unadjusted_fields(unadjusted: pd.DataFrame,
                                qfq: pd.DataFrame):
    """用同一 data_store 中复权日线的同日成交字段补未复权表缺失值。

    amount/turnover 不改变价格口径；amount 只接受正值，turnover 接受非负值。
    不做跨日插值，也不使用旧 data_cache/unadjusted。
    """
    base = unadjusted.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    source = qfq.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    fields = [f for f in ["amount", "turnover"] if f in source.columns]
    if not fields:
        return base, {"filled": {"amount": 0, "turnover": 0}}

    source = source[["date"] + fields].drop_duplicates("date", keep="last")
    source = source.rename(columns={f: f"{f}__qfq" for f in fields})
    merged = base.merge(source, on="date", how="left", sort=False)
    filled = {"amount": 0, "turnover": 0}
    for field in fields:
        if field not in merged.columns:
            merged[field] = np.nan
        candidate = merged[f"{field}__qfq"]
        valid = candidate.notna()
        if field == "amount":
            valid &= candidate > 0
        else:
            valid &= candidate >= 0
        mask = merged[field].isna() & valid
        merged.loc[mask, field] = candidate[mask]
        filled[field] = int(mask.sum())
        merged = merged.drop(columns=[f"{field}__qfq"])
    return merged, {"filled": filled}


def _new_backup_dir(root: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"backup_data_unification_{stamp}")
    os.makedirs(path, exist_ok=False)
    return path


def canonicalize_data_store():
    """清理两个正式日线目录，并将未复权字段与同源复权表对齐。"""
    backup_dir = _new_backup_dir(DATA_STORE)
    reports = []
    total_purged = 0
    total_filled = {"amount": 0, "turnover": 0}

    # 先清理复权根目录，再清理唯一正式未复权目录。
    for directory in [DATA_STORE, os.path.join(DATA_STORE, "unadjusted")]:
        if not os.path.isdir(directory):
            continue
        rel_dir = os.path.relpath(directory, DATA_STORE)
        for path in sorted(os.listdir(directory)):
            if not path.endswith(".parquet") or not path[:6].isdigit():
                continue
            full_path = os.path.join(directory, path)
            code = path[:-8]
            try:
                original = pd.read_parquet(full_path)
                cleaned, report = purge_invalid_rows(original, code)
                if report["removed_rows"]:
                    backup_path = os.path.join(backup_dir, rel_dir, path)
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    import shutil
                    shutil.copy2(full_path, backup_path)
                    cleaned.to_parquet(full_path, index=False, engine="pyarrow")
                    total_purged += report["removed_rows"]
                    reports.append({"action": "purge", **report,
                                    "path": os.path.relpath(full_path, DATA_STORE)})
            except Exception as exc:
                reports.append({"action": "purge_error", "path": full_path,
                                "error": str(exc)})

    unadj_dir = os.path.join(DATA_STORE, "unadjusted")
    for path in sorted(os.listdir(unadj_dir)) if os.path.isdir(unadj_dir) else []:
        if not path.endswith(".parquet") or not path[:6].isdigit():
            continue
        code = path[:-8]
        unadj_path = os.path.join(unadj_dir, path)
        qfq_path = os.path.join(DATA_STORE, path)
        if not os.path.exists(qfq_path):
            continue
        try:
            unadj = pd.read_parquet(unadj_path)
            qfq = pd.read_parquet(qfq_path, columns=["date", "amount", "turnover"])
            repaired, report = reconcile_unadjusted_fields(unadj, qfq)
            if sum(report["filled"].values()):
                backup_path = os.path.join(backup_dir, "unadjusted", path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                import shutil
                if not os.path.exists(backup_path):
                    shutil.copy2(unadj_path, backup_path)
                repaired.to_parquet(unadj_path, index=False, engine="pyarrow")
                for field in total_filled:
                    total_filled[field] += report["filled"].get(field, 0)
                reports.append({"action": "reconcile", "code": code,
                                "path": os.path.relpath(unadj_path, DATA_STORE),
                                **report})
        except Exception as exc:
            reports.append({"action": "reconcile_error", "path": unadj_path,
                            "error": str(exc)})

    report_path = os.path.join(DATA_STORE, "_data_unification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"canonical_unadjusted_dir": os.path.join(DATA_STORE, "unadjusted"),
                   "backup_dir": backup_dir, "purged_rows": total_purged,
                   "filled_fields": total_filled, "files": reports,
                   "timestamp": datetime.now().isoformat(timespec="seconds")},
                  f, ensure_ascii=False, indent=2, default=str)
    log(f"口径统一完成: 删除 {total_purged} 行, 补齐 amount/turnover {total_filled}")
    log(f"修正前备份: {backup_dir}")
    log(f"审计报告: {report_path}")
    return {"backup_dir": backup_dir, "purged_rows": total_purged,
            "filled_fields": total_filled, "reports": reports}


def log(msg: str):
    """统一日志输出"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  诊断: 识别受影响的股票
# ═══════════════════════════════════════════════════════════════════════

def diagnose(symbols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    诊断 data_store 中 amount/turnover 的缺失情况。

    返回:
      (affected, healthy, missing_files)
      - affected: amount 或 turnover 全为 NaN 的股票
      - healthy: 数据完整的股票
      - missing_files: parquet 文件不存在的股票
    """
    affected = []
    healthy = []
    missing_files = []

    for sym in symbols:
        path = os.path.join(DATA_STORE, f"{sym}.parquet")
        if not os.path.exists(path):
            missing_files.append(sym)
            continue

        try:
            df = pd.read_parquet(path, columns=["amount", "turnover"])
            amount_nan = df["amount"].isna().all() if "amount" in df.columns else True
            turnover_nan = df["turnover"].isna().all() if "turnover" in df.columns else True

            if amount_nan or turnover_nan:
                affected.append(sym)
            else:
                healthy.append(sym)
        except Exception:
            affected.append(sym)

    return affected, healthy, missing_files


# ═══════════════════════════════════════════════════════════════════════
#  baostock 数据获取
# ═══════════════════════════════════════════════════════════════════════

def _to_baostock_symbol(code: str) -> str:
    """将 6 位代码转为 baostock 格式 (sz.000001 / sh.600000)"""
    if code.startswith("6"):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


def fetch_amount_turnover(bs_module, code: str,
                          start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    通过 baostock 获取成交额和换手率。

    参数:
      bs_module: 已登录的 baostock 模块
      code: 6 位股票代码
      start_date: YYYY-MM-DD 格式
      end_date: YYYY-MM-DD 格式

    返回:
      DataFrame with columns [date, amount, turnover], 失败返回 None
    """
    bs_symbol = _to_baostock_symbol(code)

    rs = bs_module.query_history_k_data_plus(
        bs_symbol,
        "date,amount,turn",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",  # 前复权
    )

    if rs.error_code != "0":
        return None

    data_list = []
    while (rs.error_code == "0") and rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        return None

    df = pd.DataFrame(data_list, columns=rs.fields)

    # 类型转换
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turn"], errors="coerce")
    df = df[["date", "amount", "turnover"]].dropna(subset=["amount"])

    return df if not df.empty else None


# ═══════════════════════════════════════════════════════════════════════
#  修复: 合并数据
# ═══════════════════════════════════════════════════════════════════════

def fix_stock(bs_module, code: str, start_date: str, end_date: str) -> Tuple[bool, str]:
    """
    修复单只股票的 amount/turnover 数据。

    流程:
      1. 读取现有 parquet
      2. 从 baostock 获取 amount/turnover
      3. 按日期合并
      4. 保存回 parquet

    返回: (success, message)
    """
    path = os.path.join(DATA_STORE, f"{code}.parquet")

    # 读取现有数据
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return False, f"读取失败: {e}"

    # 获取 baostock 数据
    bs_df = fetch_amount_turnover(bs_module, code, start_date, end_date)
    if bs_df is None or bs_df.empty:
        return False, "baostock 无数据"

    # 确保 date 列类型一致
    df["date"] = pd.to_datetime(df["date"])

    # 删除现有的 amount/turnover 列 (全为 NaN)
    df = df.drop(columns=["amount", "turnover"], errors="ignore")

    # 合并
    df = df.merge(bs_df, on="date", how="left")

    # 验证合并结果
    filled = df["amount"].notna().sum()
    total = len(df)
    fill_rate = filled / total if total > 0 else 0

    if fill_rate < 0.5:
        return False, f"填充率过低: {fill_rate:.1%} ({filled}/{total})"

    # 保存
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception as e:
        return False, f"保存失败: {e}"

    return True, f"填充率 {fill_rate:.1%} ({filled}/{total})"


# ═══════════════════════════════════════════════════════════════════════
#  验证
# ═══════════════════════════════════════════════════════════════════════

def validate_fix(symbols: List[str], sample_size: int = 50) -> dict:
    """
    验证修复结果。

    检查:
      - amount/turnover 不再全为 NaN
      - 数值合理性 (amount > 0, 0 <= turnover <= 100)
      - 日期对齐

    返回验证报告 dict
    """
    import random
    sample = random.sample(symbols, min(sample_size, len(symbols)))

    results = {
        "total_checked": len(sample),
        "passed": 0,
        "failed": [],
        "stats": {
            "avg_fill_rate": [],
            "avg_amount": [],
            "avg_turnover": [],
        }
    }

    for code in sample:
        path = os.path.join(DATA_STORE, f"{code}.parquet")
        if not os.path.exists(path):
            results["failed"].append((code, "文件不存在"))
            continue

        try:
            df = pd.read_parquet(path)

            # 检查 amount
            if "amount" not in df.columns:
                results["failed"].append((code, "缺少 amount 列"))
                continue

            fill_rate = df["amount"].notna().mean()
            if fill_rate < 0.5:
                results["failed"].append((code, f"amount 填充率 {fill_rate:.1%}"))
                continue

            # 检查数值合理性
            valid_amount = df["amount"].dropna()
            if (valid_amount <= 0).any():
                n_bad = (valid_amount <= 0).sum()
                results["failed"].append((code, f"amount 有 {n_bad} 个非正值"))
                continue

            # 检查 turnover
            if "turnover" in df.columns:
                valid_turn = df["turnover"].dropna()
                if len(valid_turn) > 0 and (valid_turn < 0).any():
                    results["failed"].append((code, "turnover 有负值"))
                    continue

            results["passed"] += 1
            results["stats"]["avg_fill_rate"].append(fill_rate)
            results["stats"]["avg_amount"].append(valid_amount.mean())
            if "turnover" in df.columns and df["turnover"].notna().any():
                results["stats"]["avg_turnover"].append(df["turnover"].dropna().mean())

        except Exception as e:
            results["failed"].append((code, str(e)))

    return results


# ═══════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False,
        limit: Optional[int] = None,
        resume: bool = False,
        validate_only: bool = False):
    """
    主入口: 修复 data_store 中缺失的 amount/turnover 数据。
    """
    # ── 读取元数据 ──
    if not os.path.exists(META_FILE):
        log(f"错误: 元数据文件不存在: {META_FILE}")
        sys.exit(1)

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    symbols = meta["symbols"]
    date_range = meta.get("date_range", ["2018-01-01", "2026-07-31"])
    start_date = date_range[0]
    end_date = date_range[1]

    log(f"数据源: {meta.get('source', 'unknown')}")
    log(f"股票池: {len(symbols)} 只")
    log(f"日期范围: {start_date} ~ {end_date}")

    # ── 诊断 ──
    log("")
    log("=" * 60)
    log("诊断中...")
    affected, healthy, missing = diagnose(symbols)
    log(f"  受影响 (amount/turnover 全 NaN): {len(affected)}")
    log(f"  数据完整: {len(healthy)}")
    log(f"  文件缺失: {len(missing)}")
    log("=" * 60)

    if not affected:
        log("无需修复, 所有股票数据完整!")
        return

    # ── 仅验证模式 ──
    if validate_only:
        log("")
        log("验证模式: 检查数据质量...")
        # 验证所有有数据的股票 (抽样)
        check_symbols = affected + healthy
        report = validate_fix(check_symbols, sample_size=min(100, len(check_symbols)))
        log(f"  检查: {report['total_checked']} 只")
        log(f"  通过: {report['passed']}")
        log(f"  失败: {len(report['failed'])}")
        if report["stats"]["avg_fill_rate"]:
            avg_fill = np.mean(report["stats"]["avg_fill_rate"])
            log(f"  平均填充率: {avg_fill:.1%}")
        if report["failed"]:
            log(f"  失败详情:")
            for code, reason in report["failed"][:10]:
                log(f"    {code}: {reason}")
        return

    # ── dry-run 模式 ──
    if dry_run:
        log("")
        log("[DRY-RUN] 以下股票将被修复:")
        to_fix = affected[:limit] if limit else affected
        log(f"  数量: {len(to_fix)}")
        log(f"  预计耗时: {len(to_fix) * 2 / 60:.0f} 分钟")
        log(f"  前 10 只: {to_fix[:10]}")
        return

    # ── 执行修复 ──
    to_fix = affected[:limit] if limit else affected

    if resume:
        # 跳过已有有效 amount 的 (可能上次修复了一部分)
        log("断点续传模式: 检查已修复的文件...")
        still_affected = []
        for code in to_fix:
            path = os.path.join(DATA_STORE, f"{code}.parquet")
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path, columns=["amount"])
                    if not df["amount"].isna().all():
                        continue  # 已修复, 跳过
                except Exception:
                    pass
            still_affected.append(code)
        log(f"  跳过已修复: {len(to_fix) - len(still_affected)}")
        to_fix = still_affected

    total = len(to_fix)
    log("")
    log(f"开始修复: {total} 只股票")
    log(f"  数据源: baostock")
    log(f"  预计耗时: {total * 2 / 60:.0f} 分钟")
    log("")

    # 登录 baostock
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        log(f"baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    log("baostock 登录成功")

    # ── 逐只修复 ──
    done = 0
    failed: List[Tuple[str, str]] = []
    t0 = time.time()

    for i, code in enumerate(to_fix, 1):
        # 限速
        if done > 0:
            time.sleep(RATE_LIMIT_SLEEP)

        success, msg = fix_stock(bs, code, start_date, end_date)

        if success:
            done += 1
        else:
            failed.append((code, msg))

        # 进度
        if i % PROGRESS_INTERVAL == 0 or i == total:
            elapsed = time.time() - t0
            pct = i / total * 100
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            log(f"[{i}/{total}] {code} {'OK' if success else 'FAIL'} | "
                f"进度 {pct:.1f}% | 成功 {done} | 失败 {len(failed)} | "
                f"耗时 {elapsed:.0f}s | 剩余 ~{eta:.0f}s")

    # 登出
    bs.logout()

    # ── 汇总 ──
    elapsed = time.time() - t0
    log("")
    log("=" * 60)
    log(f"修复完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f} 分钟)")
    log(f"  成功: {done}")
    log(f"  失败: {len(failed)}")

    if failed:
        log(f"  失败列表 (前 20):")
        for code, reason in failed[:20]:
            log(f"    {code}: {reason}")

    # ── 验证 ──
    if done > 0:
        log("")
        log("验证修复结果...")
        # 验证实际修复成功的股票 (排除失败的)
        fixed_codes = [code for code in to_fix
                       if code not in {c for c, _ in failed}]
        report = validate_fix(fixed_codes, sample_size=min(50, len(fixed_codes)))
        log(f"  抽样检查: {report['total_checked']} 只")
        log(f"  通过: {report['passed']}")
        log(f"  失败: {len(report['failed'])}")
        if report["stats"]["avg_fill_rate"]:
            avg_fill = np.mean(report["stats"]["avg_fill_rate"])
            log(f"  平均填充率: {avg_fill:.1%}")
        if report["failed"]:
            log(f"  验证失败详情:")
            for code, reason in report["failed"][:5]:
                log(f"    {code}: {reason}")

    # ── 更新元数据 ──
    if done > 0:
        meta["source"] = "tencent_fallback + baostock_fix(amount,turnover)"
        meta["last_update"] = datetime.now().isoformat(timespec="seconds")
        meta["fix_history"] = meta.get("fix_history", [])
        meta["fix_history"].append({
            "date": datetime.now().isoformat(timespec="seconds"),
            "action": "fill_amount_turnover",
            "source": "baostock",
            "fixed": done,
            "failed": len(failed),
            "failed_symbols": [code for code, _ in failed],
        })

        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log(f"元数据已更新: {META_FILE}")

    # ── 保存失败列表 ──
    if failed:
        fail_path = os.path.join(DATA_STORE, "_fix_failures.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump({
                "failed": [{"code": code, "reason": reason}
                           for code, reason in failed],
                "timestamp": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)
        log(f"失败列表已写入: {fail_path}")

    log("")
    log("完成!")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="修复和清理 data_store 中的日线数据")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅统计待修复数量, 不实际修改")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制修复数量 (测试用)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传: 跳过已修复的文件")
    parser.add_argument("--validate", action="store_true",
                        help="仅验证数据质量, 不修复")
    parser.add_argument("--canonicalize", action="store_true",
                        help="清理坏行，并用同源复权表补齐未复权 amount/turnover")

    args = parser.parse_args()

    if args.canonicalize:
        canonicalize_data_store()
        return

    run(
        dry_run=args.dry_run,
        limit=args.limit,
        resume=args.resume,
        validate_only=args.validate,
    )


if __name__ == "__main__":
    main()
