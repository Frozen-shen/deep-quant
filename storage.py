"""
状态持久化层 — SQLite 数据库

表结构:
  positions    — 当前持仓 (symbol, market, qty, avg_cost)
  trades       — 交易记录 (日期、标的、方向、数量、价格)
  equity_log   — 每日权益快照
  signals      — 策略信号历史
  config       — 键值配置 (上次运行日期、初始资金等)
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    symbol      TEXT PRIMARY KEY,
    market      TEXT NOT NULL,
    qty         INTEGER NOT NULL DEFAULT 0,
    avg_cost    REAL NOT NULL DEFAULT 0.0,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    market      TEXT NOT NULL,
    date        TEXT NOT NULL,
    action      TEXT NOT NULL,       -- BUY / SELL
    qty         INTEGER NOT NULL,
    price       REAL NOT NULL,
    commission  REAL DEFAULT 0.0,
    reason      TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS equity_log (
    date            TEXT PRIMARY KEY,
    cash            REAL NOT NULL,
    holdings_value  REAL NOT NULL,
    total_equity    REAL NOT NULL,
    daily_return    REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    signal      INTEGER NOT NULL,     -- 1=BUY, -1=SELL, 0=HOLD
    confidence  REAL DEFAULT 0.0,
    reason      TEXT DEFAULT '',
    executed    INTEGER DEFAULT 0     -- 0=未执行, 1=已执行
);

CREATE TABLE IF NOT EXISTS config (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- 回测历史 (记录每次回测的参数和结果)
CREATE TABLE IF NOT EXISTS backtests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT NOT NULL,           -- 运行时间
    symbol          TEXT NOT NULL,
    market          TEXT NOT NULL,
    strategy        TEXT NOT NULL,           -- 策略名称
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    params          TEXT DEFAULT '{}',       -- 参数字典 JSON
    -- 结果指标
    total_return    REAL,
    annual_return   REAL,
    sharpe_ratio    REAL,
    sortino_ratio   REAL,
    max_drawdown    REAL,
    calmar_ratio    REAL,
    total_trades    INTEGER,
    win_rate        REAL,
    final_equity    REAL,
    benchmark_return REAL,
    excess_return   REAL,
    -- 备注
    notes           TEXT DEFAULT ''
);

-- 因子快照 (缓存已计算的因子，避免重复计算)
CREATE TABLE IF NOT EXISTS factor_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    factor_name TEXT NOT NULL,
    factor_value REAL,
    computed_at TEXT NOT NULL,
    UNIQUE(symbol, date, factor_name)
);

CREATE INDEX IF NOT EXISTS idx_backtests_symbol ON backtests(symbol);
CREATE INDEX IF NOT EXISTS idx_backtests_run_at ON backtests(run_at);
CREATE INDEX IF NOT EXISTS idx_factors_lookup ON factor_snapshots(symbol, date, factor_name);
"""


def get_db(path: str = DB_PATH) -> sqlite3.Connection:
    """获取数据库连接。"""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # 允许并发读写
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: str = DB_PATH):
    """初始化数据库（幂等）。"""
    conn = get_db(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


# ================================================================
#  Position CRUD
# ================================================================

def get_position(symbol: str, path: str = DB_PATH) -> Optional[Dict]:
    """获取单只股票持仓。"""
    conn = get_db(path)
    row = conn.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_positions(path: str = DB_PATH) -> List[Dict]:
    """获取全部持仓。"""
    conn = get_db(path)
    rows = conn.execute("SELECT * FROM positions WHERE qty > 0").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def upsert_position(symbol: str, market: str, qty: int, avg_cost: float,
                    path: str = DB_PATH):
    """插入或更新持仓。"""
    conn = get_db(path)
    conn.execute(
        """INSERT OR REPLACE INTO positions (symbol, market, qty, avg_cost, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (symbol, market, qty, avg_cost, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def clear_positions(path: str = DB_PATH):
    """清空持仓（重置用）。"""
    conn = get_db(path)
    conn.execute("DELETE FROM positions")
    conn.commit()
    conn.close()


# ================================================================
#  Trade CRUD
# ================================================================

def record_trade(symbol: str, market: str, date: str, action: str,
                 qty: int, price: float, commission: float = 0.0,
                 reason: str = "", path: str = DB_PATH) -> int:
    """记录一笔交易，返回 trade ID。"""
    conn = get_db(path)
    cur = conn.execute(
        """INSERT INTO trades (symbol, market, date, action, qty, price, commission, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, market, date, action, qty, price, commission, reason)
    )
    conn.commit()
    tid = cur.lastrowid
    conn.close()
    return tid


def get_trades(symbol: Optional[str] = None, limit: int = 50,
               path: str = DB_PATH) -> List[Dict]:
    """获取交易记录。"""
    conn = get_db(path)
    if symbol:
        rows = conn.execute(
            "SELECT * FROM trades WHERE symbol=? ORDER BY date DESC LIMIT ?",
            (symbol, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY date DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ================================================================
#  Equity Log
# ================================================================

def log_equity(date: str, cash: float, holdings_value: float,
               daily_return: float = 0.0, path: str = DB_PATH):
    """记录每日权益快照。"""
    conn = get_db(path)
    conn.execute(
        """INSERT OR REPLACE INTO equity_log (date, cash, holdings_value, total_equity, daily_return)
           VALUES (?, ?, ?, ?, ?)""",
        (date, cash, holdings_value, cash + holdings_value, daily_return)
    )
    conn.commit()
    conn.close()


def get_equity_log(limit: int = 252, path: str = DB_PATH) -> List[Dict]:
    """获取权益历史。"""
    conn = get_db(path)
    rows = conn.execute(
        "SELECT * FROM equity_log ORDER BY date DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ================================================================
#  Signals
# ================================================================

def record_signal(date: str, symbol: str, strategy: str, signal: int,
                  confidence: float = 0.0, reason: str = "",
                  path: str = DB_PATH) -> int:
    """记录策略信号。"""
    conn = get_db(path)
    cur = conn.execute(
        """INSERT INTO signals (date, symbol, strategy, signal, confidence, reason)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (date, symbol, strategy, signal, confidence, reason)
    )
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def get_pending_signals(path: str = DB_PATH) -> List[Dict]:
    """获取未执行的信号。"""
    conn = get_db(path)
    rows = conn.execute(
        "SELECT * FROM signals WHERE executed=0 ORDER BY date"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_signal_executed(signal_id: int, path: str = DB_PATH):
    """标记信号已执行。"""
    conn = get_db(path)
    conn.execute("UPDATE signals SET executed=1 WHERE id=?", (signal_id,))
    conn.commit()
    conn.close()


# ================================================================
#  Config
# ================================================================

def get_config(key: str, default: str = "", path: str = DB_PATH) -> str:
    """读取配置。"""
    conn = get_db(path)
    row = conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_config(key: str, value: str, path: str = DB_PATH):
    """写入配置。"""
    conn = get_db(path)
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()
    conn.close()


# ================================================================
#  Backtests — 回测历史
# ================================================================

def save_backtest(symbol: str, market: str, strategy: str, start_date: str,
                  end_date: str, params: dict, metrics: dict,
                  notes: str = "", path: str = DB_PATH) -> int:
    """保存一次回测结果，返回 backtest ID。"""
    from datetime import datetime
    import json
    conn = get_db(path)

    # 安全取值，处理 numpy 类型
    def _val(key, default=None):
        v = metrics.get(key, default)
        if v is None:
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    cur = conn.execute(
        """INSERT INTO backtests (run_at, symbol, market, strategy, start_date, end_date,
           params, total_return, annual_return, sharpe_ratio, sortino_ratio,
           max_drawdown, calmar_ratio, total_trades, win_rate, final_equity,
           benchmark_return, excess_return, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            symbol, market, strategy, start_date, end_date,
            json.dumps(params, ensure_ascii=False),
            _val("total_return"), _val("annual_return"),
            _val("sharpe_ratio"), _val("sortino_ratio"),
            _val("max_drawdown"), _val("calmar_ratio"),
            int(_val("total_trades", 0)),
            _val("win_rate"), _val("final_equity"),
            _val("benchmark_return"), _val("excess_vs_benchmark"),
            notes,
        )
    )
    conn.commit()
    bid = cur.lastrowid
    conn.close()
    return bid


def get_backtests(symbol: Optional[str] = None, limit: int = 20,
                  path: str = DB_PATH) -> List[Dict]:
    """获取回测历史。"""
    conn = get_db(path)
    if symbol:
        rows = conn.execute(
            "SELECT * FROM backtests WHERE symbol=? ORDER BY run_at DESC LIMIT ?",
            (symbol, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM backtests ORDER BY run_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compare_backtests(symbol: str, strategy: str, limit: int = 10,
                      path: str = DB_PATH) -> List[Dict]:
    """对比同一策略的多次回测结果。"""
    conn = get_db(path)
    rows = conn.execute(
        """SELECT run_at, symbol, strategy, params, sharpe_ratio, total_return,
           max_drawdown, total_trades, excess_return
           FROM backtests WHERE symbol=? AND strategy=?
           ORDER BY run_at DESC LIMIT ?""",
        (symbol, strategy, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ================================================================
#  Factor Snapshots — 因子缓存
# ================================================================

def save_factor_snapshot(symbol: str, date: str, factor_name: str,
                         factor_value: float, path: str = DB_PATH):
    """保存单个因子值到缓存。"""
    from datetime import datetime
    conn = get_db(path)
    conn.execute(
        """INSERT OR REPLACE INTO factor_snapshots
           (symbol, date, factor_name, factor_value, computed_at)
           VALUES (?, ?, ?, ?, ?)""",
        (symbol, str(date), factor_name, factor_value, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def save_factor_snapshots_batch(symbol: str, df_factors: "pd.DataFrame",
                                path: str = DB_PATH):
    """批量保存因子快照。"""
    from datetime import datetime
    conn = get_db(path)
    now = datetime.now().isoformat()
    factor_cols = [c for c in df_factors.columns if c not in ("date", "symbol")]
    rows = []
    for _, row in df_factors.iterrows():
        date = str(row["date"])[:10]
        for col in factor_cols:
            val = row[col]
            if pd.notna(val):
                rows.append((symbol, date, col, float(val), now))
    if rows:
        conn.executemany(
            """INSERT OR REPLACE INTO factor_snapshots
               (symbol, date, factor_name, factor_value, computed_at)
               VALUES (?, ?, ?, ?, ?)""", rows
        )
    conn.commit()
    conn.close()


def get_factor_snapshot(symbol: str, factor_name: str, start_date: str = None,
                        end_date: str = None, path: str = DB_PATH) -> "pd.DataFrame":
    """从缓存读取因子。"""
    import pandas as pd
    conn = get_db(path)
    query = "SELECT date, factor_value FROM factor_snapshots WHERE symbol=? AND factor_name=?"
    params = [symbol, factor_name]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", factor_name])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# 需要 pandas 引用
import pandas as pd
