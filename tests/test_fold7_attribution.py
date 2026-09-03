"""fold_7 归因诊断的可复核敏感性统计测试。"""

import pandas as pd

import scripts.active.run_fold7_attribution as attribution


def test_h4_reports_metrics_after_excluding_suspect_rows(tmp_path, monkeypatch):
    """H4 必须把 suspect 行剔除后的统计写入 JSON，而不是只打标签。"""
    store = tmp_path / "store"
    minute = tmp_path / "minute"
    store.mkdir()
    minute.mkdir()
    monkeypatch.setattr(attribution, "STORE_DIR", str(store))
    monkeypatch.setattr(attribution, "MINUTE_DIR", str(minute))

    pd.DataFrame({
        "date": pd.to_datetime(["2026-01-06"]),
        "close": [8.5],
    }).to_parquet(store / "600508.parquet")
    pd.DataFrame({
        "day": ["2026-01-06"],
        "amount": [1000.0],
        "volume": [100.0],
    }).to_parquet(minute / "600508.parquet")

    out = attribution.h4({"trades": [{
        "date": "2026-01-06", "symbol": "600508", "action": "BUY",
        "qty": 100, "price": 8.5, "fill_times": [],
    }]})

    clean = out["excluding_suspect"]["by_side"]["BUY"]
    assert clean["n"] == 0
    assert out["suspect_threshold_bps"] == 500.0


def test_h3_replays_holdings_at_each_event_date(tmp_path, monkeypatch):
    """H3 不能用期末持仓回填早期日期，必须按成交日期回放。"""
    store = tmp_path / "store"
    industry = tmp_path / "industry_map.parquet"
    store.mkdir()
    for sym, close in (("000001", 10.0), ("000002", 20.0)):
        pd.DataFrame({
            "date": pd.to_datetime(["2026-02-02", "2026-06-30"]),
            "close": [close, close],
        }).to_parquet(store / f"{sym}.parquet")
    pd.DataFrame({
        "code": ["000001", "000002"], "industry": ["IA", "IB"],
    }).to_parquet(industry)
    monkeypatch.setattr(attribution, "STORE_DIR", str(store))

    out = attribution.h3({
        "trades": [
            {"date": "2026-01-06", "symbol": "000001", "action": "BUY", "qty": 100},
            {"date": "2026-06-09", "symbol": "000001", "action": "SELL", "qty": 100},
            {"date": "2026-06-09", "symbol": "000002", "action": "BUY", "qty": 100},
        ],
        "positions_history": [
            {"date": "2026-02-02", "positions": ["000001"]},
            {"date": "2026-06-30", "positions": ["000002"]},
        ],
    }, industry_map=str(industry))

    assert [row["industry"] for row in out["rows"]] == [
        {"IA": 100.0}, {"IB": 100.0},
    ]
    assert out["max_known_industry_pct_max"] == 100.0
