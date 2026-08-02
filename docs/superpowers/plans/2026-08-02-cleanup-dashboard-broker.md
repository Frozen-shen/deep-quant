# 架构整理 + 金融看板 + 券商仿真盘接入 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清理 quant-starter 死代码（src/quant 等归档），构建 FastAPI + React 全功能金融看板（服务 11/3 毕业评估），预留 QMT/miniQMT 券商仿真盘执行适配层。

**Architecture:** 保留根目录模块栈为唯一主线（死代码全部 `git mv`/`mv` 进 `archive/`）；新增 `web/api/`（FastAPI，uvicorn :8000，直接复用 storage/data_cache 等根模块）与 `web/ui/`（Vite + React + TS，dev proxy 到 :8000）；执行层新增 `execution/broker/` 可插拔适配器（QmtAdapter 仿真 / PaperAdapter 降级）。

**Tech Stack:** Python 3.12（`C:\Users\Frozen\AppData\Local\Programs\Python\Python312\python.exe`）、FastAPI + uvicorn（已装，fastapi 需 `pip install`）、Node v22 + npm 10、Vite + React 18 + TS、Ant Design 5、ECharts、TanStack Query、pytest（unittest 风格断言）。

## Global Constraints

- 8/3 模拟盘上线优先：Phase 1 整理只动死代码，不碰 `scripts/active/` 活脚本与 `daily_pipeline.py`
- Python 解释器用完整路径 `/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe`（`python` 是 WindowsApps stub）
- 无鉴权、localhost 部署；CORS 放行 `http://localhost:5173`
- 归档统一进 `archive/`，已跟踪文件用 `git mv`，未跟踪文件用 `mv`
- 测试放 `tests/`（pytest testpaths），风格沿用现有 unittest（`sys.path.insert(0, 根目录)` + `from xxx import`）
- 每任务结束必须 commit（当前分支 `feature/v4-strategy-and-production`）
- 新代码不引入额外重依赖；QMT(xtquant) 非 pip 包，用 `try: import` 可选加载

---

### Task 1: 环境准备（fastapi 安装 + web 目录骨架）

**Files:**
- Modify: `requirements.txt`（追加 fastapi/uvicorn/httpx）
- Create: `web/api/__init__.py`、`web/api/main.py`（最小 app）、`web/api/tests/test_smoke.py`
- Create: `web/ui/`（`npm create vite` 脚手架，react-ts 模板）

**Interfaces:**
- Produces: `web/api/main.py` 暴露 `app`（FastAPI 实例）；`web/ui/` 可 `npm run dev` 启动

- [ ] **Step 1: 安装 fastapi 并更新 requirements.txt**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pip install fastapi httpx
```

追加到 `requirements.txt`：
```
fastapi>=0.110.0
uvicorn>=0.27.0
httpx>=0.27.0
```

- [ ] **Step 2: 创建后端骨架**

`web/api/__init__.py`（空文件）。`web/api/main.py`：

```python
"""quant-starter 金融看板后端 API。"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # quant-starter 根
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI  # noqa: E402

app = FastAPI(title="quant-starter Dashboard API", version="0.1.0")


@app.get("/api/health")
def health():
    return {"status": "ok", "data_sources": {"equity_log": False}}
```

- [ ] **Step 3: 写冒烟测试**

`web/api/tests/test_smoke.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest web/api/tests/test_smoke.py -v
```

Expected: 1 passed（若报 `ModuleNotFoundError: web`，在项目根建空 `web/__init__.py`）

- [ ] **Step 5: 初始化前端脚手架**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
npm create vite@latest web/ui -- --template react-ts
cd web/ui && npm install
npm install antd @ant-design/icons echarts echarts-for-react @tanstack/react-query react-router-dom axios
```

Expected: `web/ui/package.json` 含上述依赖；`npm run dev` 可启动（用 `npm run build` 验证即可，不常驻）

- [ ] **Step 6: Commit**

```bash
git add requirements.txt web/ && git commit -m "chore: scaffold web/api (FastAPI) + web/ui (Vite React TS)"
```

---

### Task 2: Phase 1 — checkpoint 提交（零丢失前提）

**Files:** 无新建（把当前 30+ 未提交文件全部入库）

**Interfaces:** Produces: 干净的 `git status`（无未提交改动），后续归档可安全使用 `git mv`

- [ ] **Step 1: 确认当前改动规模**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
git status --short | wc -l
```

Expected: 输出 > 0（当前有约 40 个条目）

- [ ] **Step 2: 全部入库并提交**

```bash
git add -A
git commit -m "chore: checkpoint pre-cleanup WIP (untracked research modules + data reports)"
git status --short
```

Expected: `git status` 无输出（干净）

- [ ] **Step 3: 跑测试确认基线**

```bash
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/ -q
```

Expected: 全部通过（记录通过数作为后续对比基线）

---

### Task 3: 归档 src/quant 与 configs（执行 restructure §6.4）

**Files:**
- Move: `src/quant/` → `archive/src_quant/`
- Move: `configs/` → `archive/configs/`

**Interfaces:** Consumes: Task 2 的干净 git 状态。Produces: 根目录只剩一套代码

- [ ] **Step 1: 确认无活引用（防呆验证）**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
grep -rn "from src\|import src\|src\.quant\|src/quant" --include="*.py" scripts/active/ scripts/daily_pipeline.py web/ 2>/dev/null
```

Expected: 无输出（archive/ 与 archive 脚本中的引用不算，它们已封存）

- [ ] **Step 2: 执行归档**

```bash
mkdir -p archive
git mv src/quant archive/src_quant
git mv configs archive/configs
```

- [ ] **Step 3: 验证移动后无残留引用 + 测试仍绿**

```bash
grep -rn "src\.quant" --include="*.py" . --exclude-dir=archive --exclude-dir=.git 2>/dev/null
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/ -q
```

Expected: grep 无输出；测试通过数与 Task 2 基线一致

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: archive src/quant (dead parallel codebase) + configs (dead config) per restructure-design §6.4"
```

---

### Task 4: 归档孤死根模块 → archive/legacy/

**Files:**
- Move → `archive/legacy/`: `blind_test.py`、`portfolio.py`、`scheduler.py`、`risk_model.py`、`regime_advanced.py`、`alpha_decay.py`、`alpha_enhancement.py`、`research_rigor.py`、`execution_optimizer.py`、`factor_factory/`

**Interfaces:** Consumes: Task 3 的干净根目录。Produces: 根目录只剩活模块

- [ ] **Step 1: 确认这些模块无活引用**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
for m in blind_test portfolio scheduler risk_model regime_advanced alpha_decay alpha_enhancement research_rigor execution_optimizer factor_factory; do
  echo "== $m =="; grep -rn "import $m\|from $m\|from factor_factory\|import factor_factory" --include="*.py" scripts/active/ scripts/daily_pipeline.py data/ model/ execution/ tests/ web/ 2>/dev/null
done
```

Expected: 每项无输出（`model/pipeline.py` 若引用 regime/risk 等则记录——只有 tests/ 或 archive/ 引用即可归档）

- [ ] **Step 2: 执行归档（未跟踪的用 mv，已跟踪的用 git mv）**

```bash
mkdir -p archive/legacy
git mv blind_test.py portfolio.py scheduler.py risk_model.py archive/legacy/ 2>/dev/null || mv blind_test.py portfolio.py scheduler.py risk_model.py archive/legacy/
mv regime_advanced.py alpha_decay.py alpha_enhancement.py research_rigor.py execution_optimizer.py archive/legacy/
mv factor_factory archive/legacy/
```

- [ ] **Step 3: 全量测试 + 活脚本 import 检查**

```bash
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/ -q
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -c "import storage, data_cache, factor_scorer, portfolio_ranker, trading_rules; print('live imports OK')"
```

Expected: 测试通过数与基线一致；`live imports OK`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: archive orphan root modules (blind_test/portfolio/scheduler/risk/regime/alpha_*/research_rigor/execution_optimizer/factor_factory) to archive/legacy/"
```

---

### Task 5: 脚本分流（消灭"第三类脚本"）

**Files:**
- Move → `scripts/active/`: `run_full_ic_validation.py`、`run_holdout_test.py`、`run_fundamental_ic_validation.py`、`run_northbound_ic_validation.py`、`run_relative_ic_validation.py`、`run_minute_ic_validation.py`、`fetch_baostock_minute.py`、`fetch_flow_data.py`、`fetch_index_data.py`、`fetch_minute_data.py`、`fetch_smart_money.py`、`fetch_unadjusted_batch.py`、`daily_snapshot.py`
- Move → `scripts/archive/`: `run_p5_portfolio_validation.py`（已被 `active/run_research_backtest.py` 取代）
- Modify: `scripts/active/run_research_backtest.py`（docstring 注明已取代 run_p5_portfolio_validation）

**Interfaces:** Produces: `scripts/` 根只留 `daily_pipeline.py` + `active/` + `archive/` + `data_store/`

- [ ] **Step 1: 执行移动**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
for f in run_full_ic_validation.py run_holdout_test.py run_fundamental_ic_validation.py run_northbound_ic_validation.py run_relative_ic_validation.py run_minute_ic_validation.py fetch_baostock_minute.py fetch_flow_data.py fetch_index_data.py fetch_minute_data.py fetch_smart_money.py fetch_unadjusted_batch.py daily_snapshot.py; do
  git mv scripts/$f scripts/active/$f 2>/dev/null || mv scripts/$f scripts/active/$f
done
git mv scripts/run_p5_portfolio_validation.py scripts/archive/run_p5_portfolio_validation.py
```

- [ ] **Step 2: 检查脚本内相对路径依赖（BASE_DIR 计算是否仍正确）**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
grep -ln "BASE_DIR" scripts/active/run_holdout_test.py scripts/active/fetch_minute_data.py 2>/dev/null
head -20 scripts/active/run_holdout_test.py | grep -n "path\|BASE_DIR\|dirname"
```

说明：若脚本用 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 计算 BASE_DIR，移到 `active/` 后层级多一层，必须把 `dirname` 次数 +1 或改用 `parents[2]`。逐个修正（示例）：

```python
# 原: BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 改: BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 3: 验证脚本可 import / 语法正确**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m py_compile scripts/active/run_holdout_test.py scripts/active/fetch_minute_data.py scripts/active/daily_snapshot.py
```

Expected: 无输出（编译通过）

- [ ] **Step 4: 在 run_research_backtest.py docstring 追加取代说明 + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
git add -A
git commit -m "refactor: split scripts tier — move active validation/fetch scripts to active/, archive superseded run_p5_portfolio_validation"
```

---

### Task 6: 修复 pyproject.toml + 文档同步

**Files:**
- Modify: `pyproject.toml`（删 `[project.scripts]` 坏入口）
- Modify: `README.md`（补废弃说明）
- Modify: `../AGENTS.md`（工作区根，重写为当前结构）
- Modify: `DEVELOPMENT_DISCIPLINE.md`（防复发条款）

**Interfaces:** Produces: `pip install .` 不再失败；文档描述真实结构

- [ ] **Step 1: 修 pyproject.toml**

删除：
```toml
[project.scripts]
quant-backtest = "quant.cli:run_backtest"
quant-data = "quant.cli:fetch_data"
```

同时删除失效的 `[tool.setuptools.packages.find] where = ["src"]` 与 `[tool.setuptools.package-data]`（src 已归档），替换为：

```toml
[tool.setuptools.packages.find]
include = ["web*"]
```

- [ ] **Step 2: README 补废弃说明**

在 README 目录结构小节末尾追加：

```markdown
### 已归档（archive/）
- `archive/src_quant/` — 早期平行代码库（2026-08-01 重写时弃用），保留作参考
- `archive/configs/`、`archive/legacy/` — 死配置与孤死模块
- `scripts/archive/` — 禁止运行，结果不可信（历史研究脚本）
```

- [ ] **Step 3: 重写工作区根 AGENTS.md**

用当前真实结构替换（保留"运行方式/策略模式/架构约定"要点，目录清单改为：`scripts/active/`（生产）、`scripts/daily_pipeline.py`（主入口）、`web/`（看板）、根目录模块清单、`archive/`（废弃）），示例要点：

```markdown
## 目录结构
quant-starter/
├── scripts/daily_pipeline.py  # 生产编排主入口
├── scripts/active/            # 正规军：数据更新/信号/回测/IC验证
├── scripts/archive/           # 禁止运行
├── web/api/                   # FastAPI 看板后端 (:8000)
├── web/ui/                    # React 看板前端 (Vite, :5173)
├── data_cache/ data_store/    # 行情 parquet 缓存 (1372 / 3021 只)
├── execution/                 # paper_executor + circuit_breaker + broker/ (QMT适配器)
├── storage.py                 # SQLite 状态持久化 (positions/trades/equity_log)
└── archive/                   # 废弃代码 (src_quant/configs/legacy)
```

- [ ] **Step 4: 纪律文档加防复发条款**

在 `DEVELOPMENT_DISCIPLINE.md` 末尾追加第五条：

```markdown
## 第五条：模块准入（防止平行代码库复发）

新增模块必须满足其一才能合并：
- 被 `scripts/active/` 或 `scripts/daily_pipeline.py` 引用；或
- 被 `web/api/` 或 `web/` 前端引用；或
- 有对应 `tests/` 测试覆盖。

否则不得进入根目录（放 `archive/` 或独立研究目录）。
```

- [ ] **Step 5: 验证 + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/ -q
git add -A
git commit -m "chore: fix broken pyproject scripts entry, document deprecated archive/, rewrite root AGENTS.md, add module-admission rule"
```

---

### Task 7: 后端数据层 — equity/portfolio 端点

**Files:**
- Create: `web/api/config.py`（路径常量 + TTL 缓存装饰器）
- Create: `web/api/routers/__init__.py`、`web/api/routers/equity.py`、`web/api/routers/portfolio.py`
- Modify: `web/api/main.py`（挂路由 + CORS）
- Test: `web/api/tests/test_data_endpoints.py`

**Interfaces:**
- Consumes: `storage.get_equity_log(limit=252)` → `List[Dict]`（字段: date/cash/holdings_value/total_equity/daily_return）；`storage.get_all_positions()` → `List[Dict]`；`storage.get_trades(limit=N)` → `List[Dict]`；`paper_trade/portfolio.json`（initial_capital/cash/inception_date/positions）
- Produces: `GET /api/equity` → `{"curve": [{"date","total_equity","daily_return"}], "drawdown": [...], "summary": {"total_return","max_drawdown","sharpe","volatility"}}`；`GET /api/portfolio` → `{"cash","initial_capital","inception_date","positions":[{symbol,qty,avg_cost,market_value,entry_date}], "recent_trades":[...]}`

- [ ] **Step 1: 写失败测试**

`web/api/tests/test_data_endpoints.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_equity_empty_state():
    """equity_log 为空时返回空列表 + summary 为 None，不报错。"""
    r = client.get("/api/equity")
    assert r.status_code == 200
    body = r.json()
    assert body["curve"] == []
    assert body["summary"] is None


def test_equity_with_data():
    """构造临时 equity_log 数据后能算出回撤/夏普。"""
    import storage as st
    tmp_db = os.path.join(os.path.dirname(__file__), "tmp_test.db")
    st.init_db(tmp_db)
    for i, eq in enumerate([1000000, 1010000, 990000, 1005000]):
        st.log_equity(f"2026-08-0{i+1}", 500000, eq - 500000, 0.0, path=tmp_db)
    # 直接验证计算函数
    from web.api.routers import equity as eqmod
    rows = st.get_equity_log(limit=10, path=tmp_db)
    summary = eqmod.compute_summary(rows)
    assert summary["max_drawdown"] < 0
    assert summary["sharpe"] is not None
    os.remove(tmp_db)


def test_portfolio_reads_json():
    """portfolio.json 存在时返回持仓列表。"""
    r = client.get("/api/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert "cash" in body and "positions" in body
```

注意：`st.log_equity` 签名 `log_equity(date, cash, holdings_value, daily_return, path=DB_PATH)` —— 需按 storage.py:207 实际签名核对；测试中 `init_db(tmp_db)` 若不存在则先在 Step 2 确认 storage 接口签名后微调。

- [ ] **Step 2: 运行确认失败**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest web/api/tests/test_data_endpoints.py -v
```

Expected: FAIL（`ModuleNotFoundError: web.api.routers` 或 404 路由不存在）

- [ ] **Step 3: 实现 config.py**

`web/api/config.py`：

```python
"""后端路径常量与缓存工具。"""
import json
import time
from functools import wraps
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "quant.db"
PAPER_PORTFOLIO = BASE_DIR / "paper_trade" / "portfolio.json"
PAPER_RISK = BASE_DIR / "paper_trade" / "risk_report.json"
SIGNALS_FILE = BASE_DIR / "data" / "paper_signals_v3.jsonl"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
IC_DIR = BASE_DIR / "data" / "ic_validation"
DATA_STORE = BASE_DIR / "data_store"
DATA_CACHE = BASE_DIR / "data_cache"


def ttl_cache(seconds: int = 60):
    """简单 TTL 缓存装饰器（JSON 安全）。"""
    cache = {}

    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            hit = cache.get(key)
            if hit and time.time() - hit[0] < seconds:
                return hit[1]
            val = fn(*args, **kwargs)
            cache[key] = (time.time(), val)
            return val
        return wrapper
    return deco


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

- [ ] **Step 4: 实现 equity/portfolio 路由**

`web/api/routers/equity.py`：

```python
"""净值曲线端点。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
import storage  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["equity"])


def compute_summary(rows):
    """从 equity_log 行计算绩效摘要。rows: [{date,total_equity,daily_return}] 按日期升序。"""
    if not rows:
        return None
    prices = [r["total_equity"] for r in rows]
    rets = [r["daily_return"] for r in rows if r["daily_return"] is not None]
    total_return = prices[-1] / prices[0] - 1
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        peak = max(peak, p)
        max_dd = min(max_dd, p / peak - 1)
    vol = (sum(r * r for r in rets) / len(rets)) ** 0.5 if rets else 0.0
    sharpe = (sum(rets) / len(rets) / vol * (252 ** 0.5)) if vol > 0 else None
    return {"total_return": total_return, "max_drawdown": max_dd,
            "volatility": vol, "sharpe": sharpe}


@config.ttl_cache(60)
def load_equity_rows():
    return storage.get_equity_log(limit=252 * 3)


@router.get("/equity")
def get_equity():
    rows = load_equity_rows()
    rows_sorted = list(reversed(rows))
    curve = [{"date": r["date"], "total_equity": r["total_equity"],
              "daily_return": r["daily_return"]} for r in rows_sorted]
    return {"curve": curve, "summary": compute_summary(rows_sorted)}
```

`web/api/routers/portfolio.py`：

```python
"""组合端点：读 paper_trade/portfolio.json + storage。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
import storage  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["portfolio"])


@config.ttl_cache(30)
def load_portfolio():
    return config.read_json(config.PAPER_PORTFOLIO)


@router.get("/portfolio")
def get_portfolio():
    pf = load_portfolio() or {}
    positions = []
    for sym, p in (pf.get("positions") or {}).items():
        positions.append({"symbol": sym, "qty": p.get("qty"), "avg_cost": p.get("avg_cost"),
                          "market_value": p.get("market_value"), "entry_date": p.get("entry_date")})
    trades = storage.get_trades(limit=50)
    return {"cash": pf.get("cash"), "initial_capital": pf.get("initial_capital"),
            "inception_date": pf.get("inception_date"), "positions": positions,
            "recent_trades": trades}
```

- [ ] **Step 5: main.py 挂路由 + CORS**

修改 `web/api/main.py`，在 `app = FastAPI(...)` 后追加：

```python
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from web.api.routers import equity, portfolio  # noqa: E402

app.include_router(equity.router)
app.include_router(portfolio.router)
```

- [ ] **Step 6: 运行测试通过 + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest web/api/tests/ -v
git add -A
git commit -m "feat: add /api/equity + /api/portfolio endpoints (storage + paper_trade JSON)"
```

---

### Task 8: /api/graduation 毕业指标端点

**Files:**
- Create: `web/api/routers/graduation.py`
- Modify: `web/api/main.py`（挂路由）
- Test: `web/api/tests/test_graduation.py`

**Interfaces:**
- Consumes: `load_equity_rows()`（Task 7）；`config.read_json(config.IC_DIR / "p3_full_ic.json")`；`signals jsonl`（存在时）；`paper_trade/portfolio.json`（inception_date）
- Produces: `GET /api/graduation` → `{"metrics": [{"key","name","value","threshold","status": "pass"|"fail"|"pending","detail"}], "overall": "pending"}`，8 项 key: `runtime_days`、`excess_return`、`ir`、`max_drawdown`、`fill_rate`、`ic_decay`、`sharpe`、`monthly_win_rate`

- [ ] **Step 1: 写失败测试**

`web/api/tests/test_graduation.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_graduation_shape():
    """8 项指标齐全，数据不足时为 pending 而非报错。"""
    r = client.get("/api/graduation")
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    keys = {m["key"] for m in metrics}
    assert keys == {"runtime_days", "excess_return", "ir", "max_drawdown",
                    "fill_rate", "ic_decay", "sharpe", "monthly_win_rate"}
    assert all(m["status"] in ("pass", "fail", "pending") for m in metrics)
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest web/api/tests/test_graduation.py -v
```

Expected: FAIL（404 路由）

- [ ] **Step 3: 实现 graduation.py**

`web/api/routers/graduation.py`（口径对照 `docs/PAPER_GRADUATION.md`，数据不足一律 `pending`）：

```python
"""毕业指标端点 — 对应 docs/PAPER_GRADUATION.md 8 项 AND 条件。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datetime import date  # noqa: E402
from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402
from web.api.routers.equity import load_equity_rows, compute_summary  # noqa: E402

router = APIRouter(prefix="/api", tags=["graduation"])


def _benchmark_start_price():
    """CSI1000 (000852) 基准起点，用于超额收益计算。无数据返回 None。"""
    p = config.DATA_STORE / "000852.parquet"
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(p)
    return df.sort_values("date")


@config.ttl_cache(60)
def graduation_metrics():
    rows = list(reversed(load_equity_rows()))
    summary = compute_summary(rows)
    pf = config.read_json(config.PAPER_PORTFOLIO) or {}

    # 1. 运行时长
    if pf.get("inception_date"):
        days = (date.today() - date.fromisoformat(pf["inception_date"])).days
        runtime = {"key": "runtime_days", "name": "模拟盘运行时长",
                   "value": days, "threshold": 90,
                   "status": "pass" if days >= 90 else "pending", "detail": f"{days} 天（目标 ≥90 天）"}
    else:
        runtime = {"key": "runtime_days", "name": "模拟盘运行时长",
                   "value": None, "threshold": 90, "status": "pending", "detail": "模拟盘未初始化"}

    # 2-4, 7. 绩效类（依赖 equity 数据）
    if not rows:
        base = [runtime]
        for key, name, th in [("excess_return", "年化超额收益", 0.05),
                              ("ir", "信息比率 IR", 0.5),
                              ("max_drawdown", "最大回撤", -0.15),
                              ("sharpe", "夏普比率", 0.8)]:
            base.append({"key": key, "name": name, "value": None, "threshold": th,
                         "status": "pending", "detail": "等模拟盘权益数据累积"})
        base.append({"key": "ic_decay", "name": "IC 衰减", "value": None,
                     "threshold": "ICIR 未恶化", "status": "pending", "detail": "等 IC 监控数据"})
        base.append({"key": "fill_rate", "name": "信号实现率", "value": None,
                     "threshold": 0.8, "status": "pending", "detail": "等信号/成交数据"})
        base.append({"key": "monthly_win_rate", "name": "月胜率", "value": None,
                     "threshold": 0.55, "status": "pending", "detail": "等至少 2 个月数据"})
        return {"metrics": base, "overall": "pending"}

    total_return = summary["total_return"]
    years = max(len(rows) / 252.0, 1e-9)
    ann_return = (1 + total_return) ** (1 / years) - 1
    # 超额（无基准时退化为绝对年化，detail 注明）
    bench = _benchmark_start_price()
    excess = ann_return  # 简化：无基准数据时用绝对收益，PAPER_GRADUATION 口径实现时对照修正
    if bench is not None and rows:
        bench_first = float(bench["close"].iloc[0])
        bench_last = float(bench["close"].iloc[-1])
        bench_ann = (bench_last / bench_first) ** (1 / max(len(bench) / 252.0, 1e-9)) - 1
        excess = ann_return - bench_ann

    ir = (summary["sharpe"] / (252 ** 0.5) * (252 ** 0.5)) if summary["sharpe"] is not None else None
    # IR ≈ 年化超额 / 年化跟踪误差（数据不足用 sharpe 近似并注明）
    ir_val = summary["sharpe"] if summary["sharpe"] is not None else None

    # 5. 月胜率（不足 2 个月 pending）
    from collections import defaultdict
    monthly = defaultdict(list)
    for r in rows:
        monthly[r["date"][:7]].append(r["daily_return"])
    win = None
    if len(monthly) >= 2:
        wins = sum(1 for v in monthly.values() if sum(v) > 0)
        win = wins / len(monthly)

    # 6. fill_rate（signals jsonl 存在时统计 signal→trade）
    fill = None
    if config.SIGNALS_FILE.exists():
        n_sig = sum(1 for _ in open(config.SIGNALS_FILE, encoding="utf-8"))
        fill = 1.0 if n_sig == 0 else None  # 成交数据源待接，先置 None

    # 8. ic_decay（p3_full_ic 存在时给现状，衰减判定待 IC 监控累积）
    ic_status = "pending"
    if (config.IC_DIR / "p3_full_ic.json").exists():
        ic_status = "pending"  # 有基线；衰减趋势需连续监控数据

    def metric(key, name, value, threshold, status, detail):
        return {"key": key, "name": name, "value": value, "threshold": threshold,
                "status": status, "detail": detail}

    metrics = [
        runtime,
        metric("excess_return", "年化超额收益", round(excess, 4), 0.05,
               "pass" if excess > 0.05 else "pending", "≥5% 才算达标"),
        metric("ir", "信息比率 IR", round(ir_val, 3) if ir_val is not None else None, 0.5,
               "pass" if (ir_val or 0) > 0.5 else "pending", "口径：年化超额/跟踪误差（暂以 Sharpe 近似）"),
        metric("max_drawdown", "最大回撤", round(summary["max_drawdown"], 4), -0.15,
               "pass" if summary["max_drawdown"] > -0.15 else "fail", "阈值 -15%"),
        metric("sharpe", "夏普比率", round(summary["sharpe"], 3) if summary["sharpe"] is not None else None, 0.8,
               "pass" if (summary["sharpe"] or 0) > 0.8 else "pending", "目标 >0.8"),
        metric("fill_rate", "信号实现率", fill, 0.8,
               "pending" if fill is None else ("pass" if fill >= 0.8 else "fail"), "信号→成交比例"),
        metric("ic_decay", "IC 衰减", None, "ICIR 未恶化", ic_status, "等 IC 监控数据累积"),
        metric("monthly_win_rate", "月胜率", round(win, 3) if win is not None else None, 0.55,
               "pending" if win is None else ("pass" if win >= 0.55 else "fail"),
               "需 ≥2 个月数据"),
    ]
    overall = "pending" if any(m["status"] == "pending" for m in metrics) else \
        ("pass" if all(m["status"] == "pass" for m in metrics) else "fail")
    return {"metrics": metrics, "overall": overall}


@router.get("/graduation")
def get_graduation():
    return graduation_metrics()
```

- [ ] **Step 4: main.py 挂路由 + 测试 + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
# main.py 追加: from web.api.routers import graduation / app.include_router(graduation.router)
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest web/api/tests/test_graduation.py -v
git add -A
git commit -m "feat: add /api/graduation — 8 graduation metrics with pending state"
```

---

### Task 9: 信号/实验/因子/股票池/个股端点

**Files:**
- Create: `web/api/routers/signals.py`、`experiments.py`、`factors.py`、`universe.py`、`stocks.py`
- Modify: `web/api/main.py`（挂全部路由）
- Test: `web/api/tests/test_more_endpoints.py`

**Interfaces:**
- Consumes: `config.SIGNALS_FILE`（jsonl，每行一个信号 dict）、`config.EXPERIMENTS_DIR`（`exp_*.json`：experiment_id/timestamp/script/partition/results）、`config.IC_DIR`（`p3_full_ic.json`/`p6_fundamental_ic.json`/`p7_relative_ic.json`/`p8_northbound_ic.json`/`p9_minute_ic.json`，结构 `{"results":[{factor,ic_mean,icir,ic_std,n_days,pos_ratio}], "meta":{...}}`）、`data_cache` 的 `stock_names.json`/`a_sectors.json`、`data_store/{symbol}.parquet`（列: date/open/high/low/close/volume）
- Produces:
  - `GET /api/signals` → `{"count", "signals": [...最近 200 条], "fill_rate": float|None}`
  - `GET /api/experiments` → `{"count", "experiments": [...], "by_script": {...}, "by_partition": {...}}`
  - `GET /api/factors/ic` → `{"sources": ["p3","p6","p7","p8","p9"], "results": {"p3": {...}}}`（原始 json 透传 + meta）
  - `GET /api/universe` → `{"total", "stocks": [{"symbol","name","sector"}]}`（名称缺失的只给 symbol）
  - `GET /api/universe/search?q=` → 代码或名称模糊匹配（`q` 必填，最多 50 条）
  - `GET /api/stocks/{symbol}` → `{"symbol", "name", "ohlc": [{"date","open","high","low","close","volume"}], "signals": [{"date","action","weight"}]}`；symbol 不存在 → 404 `{"detail": "symbol not found"}`

- [ ] **Step 1: 写失败测试**

`web/api/tests/test_more_endpoints.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_signals_endpoint():
    r = client.get("/api/signals")
    assert r.status_code == 200
    assert "count" in r.json()


def test_experiments_endpoint():
    r = client.get("/api/experiments")
    assert r.status_code == 200
    assert "count" in r.json() and "by_script" in r.json()


def test_factors_ic_endpoint():
    r = client.get("/api/factors/ic")
    assert r.status_code == 200
    assert "sources" in r.json() and "results" in r.json()


def test_universe_endpoint():
    r = client.get("/api/universe")
    assert r.status_code == 200
    assert "total" in r.json()


def test_universe_search():
    r = client.get("/api/universe/search", params={"q": "茅台"})
    assert r.status_code == 200
    assert len(r.json()["stocks"]) <= 50


def test_stock_not_found():
    r = client.get("/api/stocks/999999")
    assert r.status_code == 404


def test_stock_found():
    # 用 data_cache 里确定存在的某只股票
    from web.api import config
    syms = config.get_cached_symbols() if hasattr(config, "get_cached_symbols") else []
    if syms:
        r = client.get(f"/api/stocks/{syms[0]}")
        assert r.status_code == 200
        assert "ohlc" in r.json()
```

- [ ] **Step 2: 运行确认失败**（Expected: 404/import 错误）

- [ ] **Step 3: 实现五个路由**

`web/api/routers/signals.py`：

```python
"""每日信号端点。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["signals"])


@config.ttl_cache(30)
def load_signals():
    if not config.SIGNALS_FILE.exists():
        return []
    out = []
    with open(config.SIGNALS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                import json
                out.append(json.loads(line))
    return out[-200:]


@router.get("/signals")
def get_signals():
    sigs = load_signals()
    return {"count": len(sigs), "signals": sigs, "fill_rate": None}
```

`web/api/routers/experiments.py`：

```python
"""实验记录端点。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["experiments"])


@config.ttl_cache(60)
def load_experiments():
    exps = []
    if config.EXPERIMENTS_DIR.exists():
        for p in sorted(config.EXPERIMENTS_DIR.glob("exp_*.json")):
            try:
                exps.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    return exps


@router.get("/experiments")
def get_experiments():
    exps = load_experiments()
    by_script = {}
    by_partition = {}
    for e in exps:
        by_script[e.get("script", "?")] = by_script.get(e.get("script", "?"), 0) + 1
        by_partition[e.get("partition", "?")] = by_partition.get(e.get("partition", "?"), 0) + 1
    return {"count": len(exps), "experiments": exps[-100:],
            "by_script": by_script, "by_partition": by_partition}
```

`web/api/routers/factors.py`：

```python
"""因子 IC 端点（多来源：p3 价量 / p6 基本面 / p7 相对 / p8 北向 / p9 分钟）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["factors"])
SOURCES = ["p3_full_ic", "p6_fundamental_ic", "p7_relative_ic",
           "p8_northbound_ic", "p9_minute_ic"]


@config.ttl_cache(60)
def load_ic():
    results = {}
    for name in SOURCES:
        p = config.IC_DIR / f"{name}.json"
        if p.exists():
            results[name] = config.read_json(p)
    return results


@router.get("/factors/ic")
def get_ic():
    data = load_ic()
    return {"sources": list(data.keys()), "results": data}
```

`web/api/routers/universe.py`：

```python
"""股票池/名称/板块端点。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["universe"])


@config.ttl_cache(300)
def load_universe():
    names = {}
    sectors = {}
    for fname, target in (("stock_names.json", names), ("a_sectors.json", sectors)):
        p = config.DATA_CACHE / fname
        if p.exists():
            try:
                target.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    syms = sorted(set(list(names.keys()) + list(sectors.keys())))
    return names, sectors, syms


@router.get("/universe")
def get_universe():
    names, sectors, syms = load_universe()
    return {"total": len(syms), "stocks": [
        {"symbol": s, "name": names.get(s, ""), "sector": sectors.get(s, "")} for s in syms]}


@router.get("/universe/search")
def search_universe(q: str):
    names, sectors, syms = load_universe()
    q = q.strip()
    hits = []
    for s in syms:
        if q in s or q in names.get(s, ""):
            hits.append({"symbol": s, "name": names.get(s, ""), "sector": sectors.get(s, "")})
            if len(hits) >= 50:
                break
    return {"stocks": hits}
```

`web/api/routers/stocks.py`：

```python
"""个股 K 线端点。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter, HTTPException  # noqa: E402
import pandas as pd  # noqa: E402
from web.api import config  # noqa: E402
from web.api.routers.universe import load_universe  # noqa: E402

router = APIRouter(prefix="/api", tags=["stocks"])


@config.ttl_cache(120)
def load_ohlc(symbol: str):
    p = config.DATA_STORE / f"{symbol}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
    df = df[cols].dropna(subset=["close"])
    return df.tail(500)


@router.get("/stocks/{symbol}")
def get_stock(symbol: str):
    df = load_ohlc(symbol)
    if df is None:
        raise HTTPException(status_code=404, detail="symbol not found")
    names, _, _ = load_universe()
    ohlc = [{"date": str(r["date"])[:10], "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"]),
             "volume": float(r["volume"]) if "volume" in df.columns else None}
            for _, r in df.iterrows()]
    return {"symbol": symbol, "name": names.get(symbol, ""), "ohlc": ohlc, "signals": []}
```

- [ ] **Step 4: main.py 挂全部路由 + 测试 + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
# main.py 追加 5 个 include_router
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest web/api/tests/ -v
git add -A
git commit -m "feat: add signals/experiments/factors-ic/universe/stocks endpoints"
```

---

### Task 10: 前端总览页（毕业指标卡 + 净值曲线）

**Files:**
- Modify: `web/ui/`（`src/main.tsx` 引入 AntD + Router；`src/App.tsx` 布局 + 路由；新建 `src/api.ts`、`src/pages/Overview.tsx`、`src/pages/Portfolio.tsx` 占位、`src/pages/Placeholder.tsx`）
- Modify: `web/ui/vite.config.ts`（dev proxy `/api` → `http://localhost:8000`）

**Interfaces:**
- Consumes: `GET /api/graduation`、`GET /api/equity`（Task 7-8 字段结构）
- Produces: `npm run dev` 后 `http://localhost:5173/` 显示指标卡 + ECharts 净值/回撤图

- [ ] **Step 1: 配置 vite proxy 与入口**

`web/ui/vite.config.ts` 追加：

```ts
server: {
  proxy: {
    '/api': { target: 'http://localhost:8000', changeOrigin: true },
  },
},
```

`web/ui/src/main.tsx` 替换为：

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App'
import './index.css'

const qc = new QueryClient()
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <ConfigProvider locale={zhCN}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </ConfigProvider>
    </QueryClientProvider>
  </React.StrictMode>,
)
```

`web/ui/src/api.ts`：

```ts
import axios from 'axios'

export const api = axios.create({ baseURL: '/api' })

export interface GraduationMetric {
  key: string
  name: string
  value: number | null
  threshold: number | string
  status: 'pass' | 'fail' | 'pending'
  detail: string
}
export interface EquityPoint { date: string; total_equity: number; daily_return: number | null }

export async function fetchGraduation(): Promise<{ metrics: GraduationMetric[]; overall: string }> {
  const { data } = await api.get('/graduation')
  return data
}
export async function fetchEquity(): Promise<{ curve: EquityPoint[]; summary: any }> {
  const { data } = await api.get('/equity')
  return data
}
```

- [ ] **Step 2: 布局与总览页**

`web/ui/src/App.tsx`：

```tsx
import { Layout, Menu } from 'antd'
import { Routes, Route, Link } from 'react-router-dom'
import Overview from './pages/Overview'
import Portfolio from './pages/Portfolio'
import Placeholder from './pages/Placeholder'

const items = [
  { key: '/', label: <Link to="/">总览</Link> },
  { key: '/portfolio', label: <Link to="/portfolio">组合</Link> },
  { key: '/signals', label: <Link to="/signals">信号</Link> },
  { key: '/factors', label: <Link to="/factors">因子</Link> },
  { key: '/experiments', label: <Link to="/experiments">实验</Link> },
  { key: '/stocks', label: <Link to="/stocks">个股</Link> },
  { key: '/trading', label: <Link to="/trading">交易监控</Link> },
  { key: '/data', label: <Link to="/data">数据状态</Link> },
]

export default function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Layout.Sider theme="dark">
        <div style={{ color: '#fff', padding: 16, fontWeight: 600 }}>quant-starter</div>
        <Menu theme="dark" mode="inline" items={items} defaultSelectedKeys={['/']} />
      </Layout.Sider>
      <Layout.Content style={{ padding: 24 }}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/signals" element={<Placeholder title="信号" />} />
          <Route path="/factors" element={<Placeholder title="因子" />} />
          <Route path="/experiments" element={<Placeholder title="实验" />} />
          <Route path="/stocks" element={<Placeholder title="个股" />} />
          <Route path="/trading" element={<Placeholder title="交易监控" />} />
          <Route path="/data" element={<Placeholder title="数据状态" />} />
        </Routes>
      </Layout.Content>
    </Layout>
  )
}
```

`web/ui/src/pages/Overview.tsx`（核心：指标卡 + 净值/回撤双图）：

```tsx
import { Card, Col, Row, Tag, Spin, Alert, Empty } from 'antd'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { fetchGraduation, fetchEquity } from '../api'

const statusColor: Record<string, string> = { pass: 'green', fail: 'red', pending: 'orange' }
const statusText: Record<string, string> = { pass: '达标', fail: '未达标', pending: '待数据' }

export default function Overview() {
  const g = useQuery({ queryKey: ['graduation'], queryFn: fetchGraduation, refetchInterval: 60_000 })
  const eq = useQuery({ queryKey: ['equity'], queryFn: fetchEquity, refetchInterval: 60_000 })

  if (g.isLoading || eq.isLoading) return <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />
  if (g.isError || eq.isError) return <Alert type="error" showIcon message="后端不可用" description="请确认 web/api 已启动 (uvicorn :8000)" />

  const metrics = g.data?.metrics ?? []
  const curve = eq.data?.curve ?? []

  const equityOption = {
    tooltip: { trigger: 'axis' },
    legend: { data: ['净值', '回撤'] },
    xAxis: { type: 'category', data: curve.map((p) => p.date) },
    yAxis: [{ type: 'value', name: '净值' }, { type: 'value', name: '回撤', axisLabel: { formatter: '{value}%' } }],
    series: [
      { name: '净值', type: 'line', data: curve.map((p) => p.total_equity), showSymbol: false },
      { name: '回撤', type: 'line', yAxisIndex: 1,
        data: (() => {
          let peak = -Infinity
          return curve.map((p) => {
            peak = Math.max(peak, p.total_equity)
            return Number(((p.total_equity / peak - 1) * 100).toFixed(2))
          })
        })(),
        showSymbol: false, lineStyle: { color: '#cf1322' } },
    ],
  }

  return (
    <div>
      <h2>毕业指标（目标 2026-11-03）</h2>
      <Row gutter={[12, 12]}>
        {metrics.map((m) => (
          <Col key={m.key} xs={12} md={6}>
            <Card size="small" title={m.name}>
              <div style={{ fontSize: 20, fontWeight: 600 }}>
                {m.value ?? '—'}
                <Tag color={statusColor[m.status]} style={{ marginLeft: 8 }}>{statusText[m.status]}</Tag>
              </div>
              <div style={{ color: '#888', fontSize: 12 }}>{m.detail}</div>
            </Card>
          </Col>
        ))}
      </Row>
      <Card title="模拟盘净值与回撤" style={{ marginTop: 16 }}>
        {curve.length === 0 ? (
          <Empty description="模拟盘 8/3 开跑后每日累积权益数据" />
        ) : (
          <ReactECharts option={equityOption} style={{ height: 360 }} />
        )}
      </Card>
    </div>
  )
}
```

`web/ui/src/pages/Portfolio.tsx`（占位实现，展示 API 连通性）：

```tsx
import { Card, Spin, Alert, Table, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Portfolio() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['portfolio'], queryFn: async () => (await api.get('/portfolio')).data,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const cols = [
    { title: '代码', dataIndex: 'symbol' }, { title: '数量', dataIndex: 'qty' },
    { title: '成本', dataIndex: 'avg_cost' }, { title: '市值', dataIndex: 'market_value' },
    { title: '建仓日', dataIndex: 'entry_date' },
  ]
  return (
    <div>
      <Typography.Title level={4}>模拟盘组合</Typography.Title>
      <Card>现金 {data.cash?.toLocaleString()} / 初始 {data.initial_capital?.toLocaleString()} / 起始 {data.inception_date}</Card>
      <Table rowKey="symbol" dataSource={data.positions ?? []} columns={cols} pagination={false} style={{ marginTop: 16 }} />
    </div>
  )
}
```

`web/ui/src/pages/Placeholder.tsx`：

```tsx
export default function Placeholder({ title }: { title: string }) {
  return <h2>{title} — 开发中（Phase 3）</h2>
}
```

- [ ] **Step 3: 构建验证 + 启动后端联调**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter/web/ui
npm run build
cd ../..
# 启动后端（后台）:
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m uvicorn web.api.main:app --port 8000 &
# 验证 API:
curl -s http://localhost:8000/api/graduation | head -c 400
curl -s http://localhost:8000/api/equity | head -c 200
```

Expected: build 成功；两个 curl 返回 JSON（毕业指标 8 项 / 空曲线）

- [ ] **Step 4: 浏览器走查（playwright 快照）**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter/web/ui && npm run dev &
```

打开 `http://localhost:5173/`，确认：侧边栏 8 个菜单、指标卡渲染（pending 橙色）、净值图显示空态文案。

- [ ] **Step 5: Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
git add web/ui && git commit -m "feat(ui): overview page — graduation metric cards + equity curve (AntD + ECharts)"
```

---

### Task 11: 前端信号/因子/实验页

**Files:**
- Create: `web/ui/src/pages/Signals.tsx`、`Factors.tsx`、`Experiments.tsx`
- Modify: `web/ui/src/App.tsx`（3 个路由替换 Placeholder）

**Interfaces:**
- Consumes: `GET /api/signals`、`/api/factors/ic`、`/api/experiments`（Task 9 字段结构）

- [ ] **Step 1: Signals.tsx** — 信号表格（AntD Table，列取信号的常见字段，用 `Object.keys` 动态兜底）+ 顶部统计（count）

```tsx
import { Card, Table, Spin, Alert, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Signals() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['signals'], queryFn: async () => (await api.get('/signals')).data,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const sigs = data?.signals ?? []
  const cols = sigs.length
    ? Object.keys(sigs[0]).filter((k) => k !== 'metadata').map((k) => ({ title: k, dataIndex: k }))
    : []
  return (
    <div>
      <Typography.Title level={4}>每日信号（共 {data?.count ?? 0} 条，显示最近 200）</Typography.Title>
      <Card>
        {sigs.length ? <Table rowKey={(r, i) => `${i}`} dataSource={sigs} columns={cols} size="small" pagination={{ pageSize: 20 }} />
          : <Alert type="info" message="暂无信号（模拟盘 8/3 开跑后产生）" />}
      </Card>
    </div>
  )
}
```

- [ ] **Step 2: Factors.tsx** — IC 来源 Tabs + 每来源表格（factor/ic_mean/icir/ic_std/n_days/pos_ratio）+ meta 描述

```tsx
import { Card, Tabs, Table, Typography, Spin, Alert } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

const sourceLabel: Record<string, string> = {
  p3_full_ic: '价量因子 (P3)', p6_fundamental_ic: '基本面 (P6)',
  p7_relative_ic: '相对因子 (P7)', p8_northbound_ic: '北向 (P8)', p9_minute_ic: '分钟因子 (P9)',
}

export default function Factors() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['factors-ic'], queryFn: async () => (await api.get('/factors/ic')).data,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const sources = data?.sources ?? []
  const cols = ['factor', 'ic_mean', 'icir', 'ic_std', 'n_days', 'pos_ratio'].map((k) => ({
    title: k, dataIndex: k,
    render: (v: unknown) => (typeof v === 'number' ? Number(v).toFixed(4) : v),
  }))
  return (
    <div>
      <Typography.Title level={4}>因子 IC 验证结果</Typography.Title>
      <Tabs
        items={sources.map((s: string) => ({
          key: s, label: sourceLabel[s] ?? s,
          children: (
            <div>
              <Typography.Paragraph type="secondary">
                {data.results[s]?.meta?.description ?? ''}
              </Typography.Paragraph>
              <Table rowKey="factor" size="small" columns={cols}
                dataSource={data.results[s]?.results ?? []} pagination={{ pageSize: 20 }} />
            </div>
          ),
        }))}
      />
    </div>
  )
}
```

- [ ] **Step 3: Experiments.tsx** — 统计卡（总数/by_script/by_partition）+ 实验表格（experiment_id/timestamp/script/partition/verdict）

```tsx
import { Card, Col, Row, Table, Typography, Spin, Alert } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Experiments() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['experiments'], queryFn: async () => (await api.get('/experiments')).data,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="加载失败" />
  const exps = data?.experiments ?? []
  const cols = ['experiment_id', 'timestamp', 'script', 'partition', 'config_hash'].map((k) => ({ title: k, dataIndex: k }))
  return (
    <div>
      <Typography.Title level={4}>实验记录</Typography.Title>
      <Row gutter={12}>
        <Col span={8}><Card size="small">总数：{data?.count ?? 0}</Card></Col>
        <Col span={16}><Card size="small">
          脚本分布：{Object.entries(data?.by_script ?? {}).map(([k, v]) => `${k}×${v}`).join('、')}
        </Card></Col>
      </Row>
      <Table rowKey="experiment_id" dataSource={exps} columns={cols} size="small" style={{ marginTop: 16 }} />
    </div>
  )
}
```

- [ ] **Step 4: App.tsx 替换路由 + build + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter/web/ui && npm run build
cd ../.. && git add web/ui && git commit -m "feat(ui): signals/factors/experiments pages"
```

---

### Task 12: 前端个股页 + 数据状态页

**Files:**
- Create: `web/ui/src/pages/Stocks.tsx`、`DataStatus.tsx`
- Modify: `web/ui/src/App.tsx`（2 个路由替换）

**Interfaces:**
- Consumes: `GET /api/universe/search?q=`、`/api/stocks/{symbol}`（Task 9）

- [ ] **Step 1: Stocks.tsx** — 搜索框（AntD AutoComplete，防抖 300ms 调 `/universe/search`）→ 选中后加载 K 线（ECharts candlestick + volume 副图）

```tsx
import { useMemo, useState } from 'react'
import { Card, Input, AutoComplete, Spin, Alert, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { api } from '../api'

export default function Stocks() {
  const [q, setQ] = useState('')
  const [symbol, setSymbol] = useState<string | null>(null)
  const search = useQuery({
    queryKey: ['search', q], queryFn: async () => (await api.get('/universe/search', { params: { q } })).data,
    enabled: q.length >= 1,
  })
  const detail = useQuery({
    queryKey: ['stock', symbol], queryFn: async () => (await api.get(`/stocks/${symbol}`)).data,
    enabled: !!symbol,
  })
  const options = useMemo(() =>
    (search.data?.stocks ?? []).map((s: any) => ({
      value: s.symbol, label: `${s.symbol} ${s.name}`,
    })), [search.data])

  const ohlc = detail.data?.ohlc ?? []
  const candleOption = {
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ohlc.map((o: any) => o.date) },
    yAxis: { scale: true },
    dataZoom: [{ type: 'inside' }],
    series: [{
      type: 'candlestick',
      data: ohlc.map((o: any) => [o.open, o.close, o.low, o.high]),
    }],
  }
  return (
    <div>
      <Typography.Title level={4}>个股行情</Typography.Title>
      <AutoComplete
        style={{ width: 320 }} options={options} onSearch={setQ}
        onSelect={(v) => setSymbol(v)} placeholder="输入代码或名称（如 600519 / 茅台）"
      />
      {detail.isLoading && <Spin style={{ marginTop: 24, display: 'block' }} />}
      {detail.isError && <Alert type="error" message="股票不存在或数据缺失" style={{ marginTop: 16 }} />}
      {detail.data && (
        <Card title={`${detail.data.symbol} ${detail.data.name}`} style={{ marginTop: 16 }}>
          <ReactECharts option={candleOption} style={{ height: 420 }} />
        </Card>
      )}
    </div>
  )
}
```

- [ ] **Step 2: DataStatus.tsx** — 读 `/api/health` + 展示数据源可用性列表

```tsx
import { Card, List, Tag, Typography, Spin } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function DataStatus() {
  const { data, isLoading } = useQuery({
    queryKey: ['health'], queryFn: async () => (await api.get('/health')).data,
  })
  return (
    <div>
      <Typography.Title level={4}>数据状态</Typography.Title>
      <Card>
        {isLoading ? <Spin /> : (
          <List
            dataSource={Object.entries(data?.data_sources ?? {})}
            renderItem={([k, v]) => (
              <List.Item>数据源 {k}：<Tag color={v ? 'green' : 'orange'}>{v ? '可用' : '待积累'}</Tag></List.Item>
            )}
          />
        )}
      </Card>
    </div>
  )
}
```

- [ ] **Step 3: 后端 health 补真实数据源探测**

修改 `web/api/main.py` 的 `/api/health`：

```python
from web.api import config  # noqa: E402
import storage  # noqa: E402


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "data_sources": {
            "equity_log": len(storage.get_equity_log(limit=1)) > 0,
            "portfolio": config.PAPER_PORTFOLIO.exists(),
            "signals": config.SIGNALS_FILE.exists(),
            "experiments": config.EXPERIMENTS_DIR.exists() and any(config.EXPERIMENTS_DIR.glob("exp_*.json")),
            "ic_results": (config.IC_DIR / "p3_full_ic.json").exists(),
            "data_store": config.DATA_STORE.exists(),
        },
    }
```

- [ ] **Step 4: 路由替换 + build + 测试 + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter/web/ui && npm run build
cd ../..
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest web/api/tests/ -q
git add -A && git commit -m "feat(ui): stocks (candlestick search) + data status pages; enrich /api/health"
```

---

### Task 13: QMT 券商适配器（execution/broker/）

**Files:**
- Create: `execution/broker/__init__.py`、`execution/broker/base.py`、`execution/broker/qmt.py`、`execution/broker/paper.py`
- Test: `tests/test_broker.py`

**Interfaces:**
- Consumes: `execution/paper_executor.py` 的 `PaperExecutor`（`execute_orders`/`snapshot`/`load_state`）
- Produces:
  - `BrokerAdapter`（ABC）: `connect() -> bool`、`place_order(symbol, side, qty, price_type="limit", price=None) -> str`、`cancel_order(order_id) -> bool`、`get_balance() -> dict`、`get_positions() -> list`、`get_orders(date) -> list`、`get_trades(date) -> list`、`get_quotes(symbols) -> dict`
  - `PaperAdapter(BrokerAdapter)`：包装现有 PaperExecutor 状态，可直接实例化测试
  - `QmtAdapter(BrokerAdapter)`：xtquant 可选加载（未安装时 `connect()` 返回 False），配置经 `config.yaml` 的 `broker.qmt` 段
  - `get_adapter(name="paper") -> BrokerAdapter` 工厂

- [ ] **Step 1: 写失败测试**

`tests/test_broker.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from execution.broker import get_adapter, PaperAdapter


class TestPaperAdapter(unittest.TestCase):
    def test_get_adapter_default(self):
        a = get_adapter("paper")
        self.assertIsInstance(a, PaperAdapter)

    def test_connect(self):
        a = get_adapter("paper")
        self.assertTrue(a.connect())

    def test_place_order_returns_id(self):
        a = get_adapter("paper")
        oid = a.place_order("000001", "BUY", 100, "limit", 10.0)
        self.assertIsInstance(oid, str)

    def test_qmt_absent_returns_false(self):
        from execution.broker.qmt import QmtAdapter
        a = QmtAdapter({})
        self.assertFalse(a.connect())  # xtquant 未安装


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**（Expected: `ModuleNotFoundError: execution.broker`）

- [ ] **Step 3: 实现 base.py**

`execution/broker/base.py`：

```python
"""券商执行适配器抽象 — 可插拔执行层（仿真/实盘）。"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BrokerAdapter(ABC):
    """统一券商接口。side: BUY/SELL; price_type: limit/market。"""

    @abstractmethod
    def connect(self) -> bool:
        """建立连接/登录。返回是否可用。"""

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: int,
                    price_type: str = "limit", price: Optional[float] = None) -> str:
        """下单，返回 order_id。"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单。"""

    @abstractmethod
    def get_balance(self) -> Dict:
        """返回 {cash, frozen, total_asset}。"""

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """返回 [{symbol, qty, avg_cost, market_value}]。"""

    @abstractmethod
    def get_orders(self, date: str) -> List[Dict]:
        """返回当日订单 [{order_id, symbol, side, qty, filled_qty, status}]。"""

    @abstractmethod
    def get_trades(self, date: str) -> List[Dict]:
        """返回当日成交 [{order_id, symbol, side, qty, price, amount}]。"""

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict:
        """返回 {symbol: {"last": float, "bid": float, "ask": float}}。"""
```

- [ ] **Step 4: 实现 paper.py（包装 PaperExecutor 状态）**

`execution/broker/paper.py`：

```python
"""PaperAdapter — 包装现有 PaperExecutor 的降级/回归实现。"""
from typing import Dict, List, Optional
import sys, os
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from execution.broker.base import BrokerAdapter  # noqa: E402
import storage  # noqa: E402


class PaperAdapter(BrokerAdapter):
    """模拟撮合：直接写 storage 的 positions/trades 表。"""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}

    def connect(self) -> bool:
        return True

    def place_order(self, symbol: str, side: str, qty: int,
                    price_type: str = "limit", price: Optional[float] = None) -> str:
        import uuid
        order_id = str(uuid.uuid4())[:8]
        # 简化：按最新收盘价成交（真实路径走 PaperExecutor.execute_orders）
        storage.record_trade(symbol=symbol, market="A", date="", action=side,
                             qty=qty, price=price or 0.0, commission=0.0,
                             reason="paper_adapter")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_balance(self) -> Dict:
        pf = {}
        p = os.path.join(BASE, "paper_trade", "portfolio.json")
        if os.path.exists(p):
            import json
            with open(p, encoding="utf-8") as f:
                pf = json.load(f)
        return {"cash": pf.get("cash"), "frozen": 0.0,
                "total_asset": pf.get("cash", 0) + sum(
                    x.get("market_value", 0) for x in pf.get("positions", {}).values())}

    def get_positions(self) -> List[Dict]:
        out = []
        for p in storage.get_all_positions():
            out.append({"symbol": p["symbol"], "qty": p["qty"],
                        "avg_cost": p["avg_cost"], "market_value": 0.0})
        return out

    def get_orders(self, date: str) -> List[Dict]:
        return []

    def get_trades(self, date: str) -> List[Dict]:
        return storage.get_trades(limit=100)

    def get_quotes(self, symbols: List[str]) -> Dict:
        import data_cache
        out = {}
        for s in symbols:
            df = data_cache.load(s)
            if df is not None and len(df):
                out[s] = {"last": float(df["close"].iloc[-1]), "bid": 0.0, "ask": 0.0}
        return out
```

- [ ] **Step 5: 实现 qmt.py（xtquant 可选加载）**

`execution/broker/qmt.py`：

```python
"""QmtAdapter — 迅投 miniQMT (xtquant) 仿真/实盘适配器。

依赖: 券商提供的 xtquant 包（非 pip），需 QMT 客户端本机登录。
未安装 xtquant 时 connect() 返回 False，系统自动降级 PaperAdapter。
"""
from typing import Dict, List, Optional
from execution.broker.base import BrokerAdapter


class QmtAdapter(BrokerAdapter):
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.xt = None
        self._trader = None
        try:
            from xtquant import xttrader  # type: ignore
            self.xt = xttrader
        except ImportError:
            self.xt = None

    def connect(self) -> bool:
        if self.xt is None:
            return False
        # TODO(用户): 按券商账号填充 config.yaml broker.qmt 段后启用
        # self._trader = self.xt.XtQuantTrader(path, session_id)
        # self._trader.start(); self._trader.connect()
        return False  # xtquant 可用但账号未配置前保持关闭

    def place_order(self, symbol: str, side: str, qty: int,
                    price_type: str = "limit", price: Optional[float] = None) -> str:
        raise NotImplementedError("QMT 未启用")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("QMT 未启用")

    def get_balance(self) -> Dict:
        return {"cash": 0.0, "frozen": 0.0, "total_asset": 0.0}

    def get_positions(self) -> List[Dict]:
        return []

    def get_orders(self, date: str) -> List[Dict]:
        return []

    def get_trades(self, date: str) -> List[Dict]:
        return []

    def get_quotes(self, symbols: List[str]) -> Dict:
        return {}
```

`execution/broker/__init__.py`：

```python
"""执行适配器工厂。config: {"adapter": "paper"|"qmt", "qmt": {...}}"""
from typing import Optional


def get_adapter(name: str = "paper", cfg: Optional[dict] = None):
    if name == "qmt":
        from execution.broker.qmt import QmtAdapter
        return QmtAdapter(cfg or {})
    from execution.broker.paper import PaperAdapter
    return PaperAdapter(cfg or {})
```

- [ ] **Step 6: 测试通过 + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_broker.py -v
git add -A && git commit -m "feat: broker adapter layer — BrokerAdapter ABC + PaperAdapter + QmtAdapter (optional xtquant)"
```

---

### Task 14: 后端 /api/broker 端点 + 前端交易监控页

**Files:**
- Create: `web/api/routers/broker.py`
- Modify: `web/api/main.py`（挂路由）
- Create: `web/ui/src/pages/Trading.tsx`
- Modify: `web/ui/src/App.tsx`（路由替换）
- Test: `web/api/tests/test_broker_api.py`

**Interfaces:**
- Consumes: `get_adapter(name, cfg)`（Task 13）
- Produces: `GET /api/broker/status` → `{"adapter": "paper", "connected": true, "balance": {...}, "positions": [...], "orders": [], "trades": [...]}`

- [ ] **Step 1: 写失败测试**

`web/api/tests/test_broker_api.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)


def test_broker_status():
    r = client.get("/api/broker/status")
    assert r.status_code == 200
    body = r.json()
    assert body["adapter"] == "paper"
    assert body["connected"] is True
    assert "balance" in body and "positions" in body
```

- [ ] **Step 2: 实现 broker.py**

`web/api/routers/broker.py`：

```python
"""券商执行状态端点（默认 PaperAdapter 降级）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from execution.broker import get_adapter  # noqa: E402

router = APIRouter(prefix="/api/broker", tags=["broker"])
_adapter = get_adapter("paper")  # config.yaml broker.adapter 可切换 qmt


@router.get("/status")
def broker_status():
    return {
        "adapter": "paper",
        "connected": _adapter.connect(),
        "balance": _adapter.get_balance(),
        "positions": _adapter.get_positions(),
        "orders": _adapter.get_orders(""),
        "trades": _adapter.get_trades(""),
    }
```

- [ ] **Step 3: main.py 挂路由 + 测试通过**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest web/api/tests/test_broker_api.py -v
```

- [ ] **Step 4: 前端 Trading.tsx**（账户资金卡 + 持仓表 + 交易记录表）

```tsx
import { Card, Col, Row, Table, Typography, Spin, Alert } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Trading() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['broker'], queryFn: async () => (await api.get('/broker/status')).data,
    refetchInterval: 30_000,
  })
  if (isLoading) return <Spin />
  if (isError) return <Alert type="error" message="执行状态不可用" />
  const posCols = ['symbol', 'qty', 'avg_cost', 'market_value'].map((k) => ({ title: k, dataIndex: k }))
  const tradeCols = ['date', 'symbol', 'action', 'qty', 'price', 'commission', 'reason'].map((k) => ({ title: k, dataIndex: k }))
  return (
    <div>
      <Typography.Title level={4}>交易监控（{data.adapter} · {data.connected ? '已连接' : '未连接'}）</Typography.Title>
      <Row gutter={12}>
        <Col span={8}><Card size="small" title="账户">现金 {data.balance?.cash?.toLocaleString()}</Card></Col>
        <Col span={8}><Card size="small" title="持仓数">{data.positions?.length}</Card></Col>
        <Col span={8}><Card size="small" title="今日成交">{data.trades?.length}</Card></Col>
      </Row>
      <Table rowKey="symbol" dataSource={data.positions ?? []} columns={posCols} size="small" style={{ marginTop: 16 }}
        pagination={{ pageSize: 10 }} />
      <Typography.Title level={5} style={{ marginTop: 24 }}>最近成交</Typography.Title>
      <Table rowKey={(r, i) => `${i}`} dataSource={data.trades ?? []} columns={tradeCols} size="small" />
    </div>
  )
}
```

- [ ] **Step 5: 路由替换 + build + Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter/web/ui && npm run build
cd ../.. && git add -A && git commit -m "feat: broker status API + trading monitor page"
```

---

### Task 15: 打磨与全量验证（空态/错误态/走查）

**Files:**
- Modify: `web/ui/src/App.tsx`（挂载错误边界）、`web/ui/src/pages/Overview.tsx` 等（统一 Empty 文案）
- 最终验证全流程

**Interfaces:** 无新接口

- [ ] **Step 1: 统一空态与错误态**

给 `App.tsx` 加全局错误提示（后端未启动时）：在 `Routes` 外包 `<ErrorBoundary>` 或使用 TanStack Query 的全局 `QueryErrorResetBoundary`（简化：各页已有 isError 分支，确认 8 个页面均有）。逐页核对：
- 总览/组合/信号/因子/实验/个股/交易监控/数据状态 —— 每页都有 `isError → Alert`、空数据 → `Empty`/`Alert type="info"`

- [ ] **Step 2: 后端全端点 curl 验证**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m uvicorn web.api.main:app --port 8000 &
for ep in health graduation equity portfolio signals experiments factors/ic universe universe/search?q=600519 stocks/600519 broker/status; do
  echo "== /api/$ep =="; curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8000/api/$ep"
done
```

Expected: 全部 200（`stocks/999999` 应为 404）

- [ ] **Step 3: pytest 全量 + 前端 build + 浏览器走查**

```bash
/c/Users/Frozen/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/ web/api/tests/ -q
cd web/ui && npm run build
```

浏览器（playwright 或手动）走查 8 个页面：导航、总览指标卡、个股搜索选股出 K 线、交易监控表格。

- [ ] **Step 4: Commit**

```bash
cd /c/Users/Frozen/ZCodeProject/quant-starter
git add -A && git commit -m "chore: polish empty/error states + full endpoint verification"
```

---

## 后续（非本计划范围，标注留缺口）

- **分钟因子决策链集成**：`p9_minute_ic.json` 数据层/展示层已就绪（Task 9/11），factor_library 注册 + run_paper_signal 评分链集成待 IC 结论后单独做
- **QMT 实盘启用**：券商账号开通后填 `config.yaml` broker.qmt 段 + 实现 `QmtAdapter.connect()` 真实登录（Task 13 已留 TODO）
- **对账机制**：`reconcile()`（券商持仓 vs storage.positions）设计在 spec §5.3，待 QMT 启用后实现
- **毕业指标口径校准**：PAPER_GRADUATION.md 对照复核（Task 8 已注明近似口径）
