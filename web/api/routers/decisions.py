"""决策层 API — 多因子信号生成与查询。"""
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from web.api.config import ttl_cache, read_json, SIGNALS_FILE, IC_DIR

router = APIRouter(prefix="/api/decisions", tags=["decisions"])


# ── 请求/响应模型 ──

class GenerateRequest(BaseModel):
    date: Optional[str] = None       # "2026-07-31", None=最新
    execute: bool = False            # 是否执行订单


class GenerateResponse(BaseModel):
    status: str                      # "ok" | "error" | "initializing"
    message: str = ""
    result: Optional[dict] = None


# ── 引擎管理 ──

def _get_engine():
    """延迟导入 + 获取单例。"""
    from decision_engine import get_engine
    return get_engine()


# ── 端点 ──

@router.get("/status")
def engine_status():
    """引擎状态 (是否已初始化、因子数、覆盖股票数)。"""
    engine = _get_engine()
    return engine.get_status()


@router.post("/initialize")
def initialize_engine(background_tasks: BackgroundTasks):
    """
    触发引擎初始化 (后台执行, ~5分钟)。
    初始化完成前 /generate 会返回 503。
    """
    engine = _get_engine()
    if engine.initialized:
        return {"status": "already_initialized", **engine.get_status()}

    def _init():
        try:
            engine.initialize()
        except Exception as e:
            import logging
            logging.getLogger("quant.decision").error("初始化失败: %s", e)

    background_tasks.add_task(_init)
    return {"status": "initializing", "message": "引擎正在后台初始化, 预计5分钟"}


@router.post("/generate", response_model=GenerateResponse)
def generate_signals(req: GenerateRequest):
    """
    生成交易信号。

    - 未初始化时返回 503
    - date=None 使用最新交易日
    - execute=true 执行订单 (写入持仓/交易记录)
    """
    engine = _get_engine()
    if not engine.initialized:
        raise HTTPException(
            status_code=503,
            detail="引擎未初始化, 请先调用 POST /api/decisions/initialize")

    try:
        result = engine.generate_signals(
            as_of_date=req.date, execute=req.execute)
        if "error" in result:
            return GenerateResponse(
                status="error", message=result["error"])
        return GenerateResponse(status="ok", result=result)
    except Exception as e:
        return GenerateResponse(status="error", message=str(e))


@router.get("/latest")
@ttl_cache(seconds=30)
def latest_signals():
    """最新一条信号 (从 JSONL 日志读取)。"""
    if not SIGNALS_FILE.exists():
        return {"signals": []}

    lines = SIGNALS_FILE.read_text(encoding="utf-8").strip().split("\n")
    if not lines:
        return {"signals": []}

    # 返回最后一条
    try:
        latest = json.loads(lines[-1])
        return {"signal": latest, "total_records": len(lines)}
    except json.JSONDecodeError:
        return {"signals": []}


@router.get("/history")
@ttl_cache(seconds=60)
def signal_history(limit: int = 20):
    """历史信号列表。"""
    if not SIGNALS_FILE.exists():
        return {"signals": []}

    lines = SIGNALS_FILE.read_text(encoding="utf-8").strip().split("\n")
    signals = []
    for line in lines[-limit:]:
        try:
            signals.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"signals": signals, "total_records": len(lines)}


@router.get("/scores")
def top_scores(top: int = 30):
    """
    最新一次评分的 Top-N 股票。
    需要先调用 /generate 产生评分。
    """
    engine = _get_engine()
    if not engine.initialized:
        raise HTTPException(status_code=503, detail="引擎未初始化")

    if not engine._last_scores:
        raise HTTPException(
            status_code=404,
            detail="无评分数据, 请先调用 POST /api/decisions/generate")

    sorted_scores = sorted(
        engine._last_scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "top": [{
            "rank": i + 1,
            "symbol": sym,
            "score": round(sc, 4),
        } for i, (sym, sc) in enumerate(sorted_scores[:top])],
        "total_scored": len(sorted_scores),
    }


@router.get("/score/{symbol}")
def stock_score(symbol: str, date: Optional[str] = None):
    """单只股票的因子分解。"""
    engine = _get_engine()
    if not engine.initialized:
        raise HTTPException(status_code=503, detail="引擎未初始化")

    result = engine.score_stock(symbol, as_of_date=date)
    if result is None:
        raise HTTPException(status_code=404, detail=f"无法计算 {symbol} 的因子")
    return result


@router.get("/factors")
@ttl_cache(seconds=300)
def factor_config():
    """因子配置与 IC 统计。"""
    engine = _get_engine()
    factors = engine.get_factor_summary()

    # 补充 IC 验证详情
    ic_path = IC_DIR / "p9_minute_ic.json"
    minute_ic = read_json(ic_path) if ic_path.exists() else None

    return {
        "factors": factors,
        "total": len(factors),
        "categories": engine._count_categories(),
        "minute_ic": minute_ic,
    }
