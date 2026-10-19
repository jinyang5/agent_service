# Ch04 混合检索 + 重排 + 评估体系 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch03 的 dense 单路检索升级为「Milvus 原生 BM25 + dense 混合召回(RRF)+ 本地 bge-reranker-v2-m3 精排 + Query 理解 + 两道确定性证据闸 + 可点引用」,并交付一套 100 题五桶、四策略、分桶落盘的 RAG 评估体系。

**Architecture:** `work/ch04/` 从 `work/ch03/` 复制,保持单一 agent tool-calling 入口与事务型工具不变。`query_faq` 内部升级为 RAG 管线:口语改写 + 同义词扩展(检索侧)→ category 过滤 → `hybrid_search`(dense Top-50 + BM25 Top-50,RRF k=60)→ 本地重排 Top-10 → 机械闸(不落池)→ 自评闸(only `useful=false` 落 `low_confidence_questions(source='self_check')`)→ 首尾组装 → 编号证据 + citations。`query_faq` 命中任一闸时在 `ToolRun` 上带 `evidence_insufficient` 信号,`run_agent_turn` / `stream_agent_turn` 检测到信号后跳过收敛模型调用,直接输出固定拒答话术。评估脚本复用同一个 `search_knowledge(strategy=...)`,检索段零 LLM、生成段 DeepSeek 裁判 `temperature=0`。

**Tech Stack:** Windows 11 + Git Bash、uv、GNU Make 4.4.1、FastAPI、SQLAlchemy(asyncmy)、MySQL 8(宿主 3307)、Milvus Standalone v2.6.22、pymilvus 3.0.2、FlagEmbedding(BGE-M3 + bge-reranker-v2-m3,进程内 CPU)、LangChain OpenAI 兼容客户端(DeepSeek 上游)。

**Spec:** `E:\mewhelp\docs\superpowers\specs\2026-10-19-ch04-hybrid-retrieval-rerank-eval-design.md`(权威;与教程范本冲突时以本 spec 为准)

## Global Constraints

- **平台与目录**:Windows 11 + Git Bash;项目用 **uv**;`make` 已安装(GNU Make 4.4.1,已在 PATH);所有工作目录一律是 `work/ch04/`;命令必须能在本机 Git Bash 直接执行,不写 macOS/Linux 专有命令。
- **代码来源**:`work/ch04/` 自 `work/ch03/` 拷贝。**不拷 `.venv`**;拷贝后删 `__pycache__`、`.pytest_cache`、`log/`、`torchinductor_*`、`pytest-of-*`;用 `uv sync` 重建环境。
- **MySQL**:容器 `mewhelp-mysql`,宿主端口 **3307**,库 `mewhelp`(root/root);测试库命名 **`mewhelp_ch04_test`**,三处一起改:`.env`(本地,不入库)/`app/config.py` 默认值/`tests/conftest.py`(它从 settings 派生库名,改注释并验证)。
- **Milvus**:Standalone **v2.6.22**,`http://localhost:19530`;pymilvus **3.0.2**。`milvus_client` 的集合主键 = MySQL `knowledge_chunks.id`(`auto_id=False`),重跑 upsert 幂等。
- **基础设施容器名是固定的**(`mewhelp-mysql` / `mewhelp-milvus` 等):本机已有容器在跑时**直接复用**;不要在 `work/ch04` 再起一份 compose(会与 ch03 项目同名冲突)。需要重启时从 `work/ch03` 执行 `make milvus-up` / `docker compose up -d mysql`。
- **HF 缓存**:`HF_HOME=E:\model`,`HF_HUB_OFFLINE=1`。Makefile 顶部 `export HF_HOME ?= E:/model` / `export HF_HUB_OFFLINE ?= 1`;直接 `uv run` 时先在 shell 里 `export HF_HOME='E:\model' HF_HUB_OFFLINE=1`。
- **嵌入/重排都是本地进程内**:嵌入沿用 `BAAI/bge-m3`;重排 `BAAI/bge-reranker-v2-m3`(首次需下载 ~2.3GB,本机 `E:\model` 已有缓存)。**下载与 `compute_score` 必须包在 `asyncio.to_thread` 里**。
- **固定选型(不自换)**:Milvus 2.5+ 原生 BM25 + `hybrid_search` RRF;`bge-reranker-v2-m3`;`BAAI/bge-m3`。走不通 → **停下问用户**,只在选型内换法(见 Task 1 红线)。
- **本章不做**:指代消解、多轮改写、`/rag-eval` 页面、LangGraph / Langfuse。
- **落池口径**:只有自评闸 `useful=false` 写 `low_confidence_questions(source='self_check')`;**机械闸只拒答不落池**。`retrieval_low_conf` / `user_feedback` 是 DDL 预留枚举,本章不写。
- **拒答必须确定性**:`query_faq` 命中任一闸时返回 `signal="evidence_insufficient"` 并落到 `ToolRun.signal`;`run_agent_turn` / `stream_agent_turn` 检测到信号后**跳过收敛模型调用**,直接输出固定拒答话术。验收 4 不依赖模型是否听话。
- **结构化输出**:`method="function_calling"` + `extra_body={"thinking": {"type": "disabled"}}`,字段保持扁平(标量 + 标量 list;不用嵌套对象数组)。DeepSeek 上游不支持 json_schema 与强制 tool_choice(ch03 R11 结论)。
- **评估集**:`tests/data/eval_ch04.jsonl`,**100 题**:A_policy / B_model / C_colloquial / E_multi 各 20(可作答)+ D_offtopic 20(库外)。字段:`id` / `bucket` / `query` / `expect_section`(A/B/C;别名数组,任一命中)/ `expect_sections_all`(E;组数组,组内任一命中)/ `expect_points`(A/B/C)/ `difficulty`。`--limit` **按桶**抽样。
- **TDD 边界**:确定性代码(检索/schema/重排/citations/落池/报告指标)走 TDD;纯 Prompt/数据任务(Query 改写、自评、裁判、评估集)用标注样例验证;前端两条走 Vibe Coding(**不套 TDD/不派 code review,按用户描述直接改,用浏览器验收**)。
- **提交策略**:每个任务结束只 `git commit`,**不 push**;遵循仓库记忆里的 git 远端策略。`.env` 已被 `.gitignore` 挡住,绝不提交密钥。

## Review Focus

1. **`query_faq` 被工具默认 5s 超时截断**。RAG 管线含改写、嵌入、混合检索、CPU 重排、自评,5s 必然不够;合理预期是单次 `query_faq` 能完整跑完。**Task 11 的 `test_query_faq_uses_long_tool_timeout` 钉住**(`registry.TIMEOUTS["query_faq"]=60.0`);Task 1 的 50 对重排耗时给出预算。
2. **拒答不彻底:模型仍被调用或机械闸误落池**。合理预期是命中 `evidence_insufficient` 后模型一次都不再调用,且机械闸只拒答不写池。**Task 11 的 `test_refusal_skips_convergence_run`(同时断言 self_check 落池)/ `test_refusal_skips_convergence_stream` / `test_mechanical_gate_does_not_pool` 钉住**。
3. **引用编号与证据顺序错位**。首尾组装后 `[n]` 必须与 SSE `citations` 的 `n`/`chunk_id` 一一对应;合理预期是点开 `[2]` 看到的就是第 2 条证据。**Task 7 的 `test_build_citations_numbers_and_maps`、Task 10 的 `test_query_faq_sufficient_numbers_after_head_tail`、Task 11 的 `test_citations_event_before_deltas` 钉住**。
4. **重排分数语义/空输入/阻塞事件循环**。`normalize=True` 必须落在 0~1,`top_n` 必须截断,空 docs 必须安全,`compute_score` 必须在 `asyncio.to_thread` 里。**Task 5 的 `test_rerank_orders_and_truncates` / `test_rerank_empty_docs` / `test_rerank_uses_to_thread` 钉住**;Task 1 用真实 50 对实测。
5. **评估集 ground truth 漂移**。`expect_section` 必须命中真实 `section_path`,`expect_points` 必须逐字(去空白)出现在目标小节,E 组必须真的跨小节,D 问题必须真的无答案。**Task 12 的 `scripts/build_eval_set.py`(逐条对 `data/kb` 切块校验)+ Task 18 的 `test_eval_set_structure` + Task 15 的真实 KB 跑分(B 桶 BM25、D 桶拒答)钉住**。

---

## 文件布局

**改(自 ch03 拷贝后):**
- `app/config.py` — ch04 参数(test DB 名、rerank model/device、recall/rerank top_k、min score)
- `app/core/llm.py` — `get_chat_model(..., temperature=0.3)`
- `app/kb/milvus_client.py` — 新 schema(dense + text/chinese analyzer + sparse/BM25 Function + 标量)、dense/bm25/hybrid、filter
- `app/kb/dualwrite.py` — 写 `dense/text/section_path/content_type/category`
- `app/core/retrieval.py` — 四策略 + 首尾组装
- `app/core/prompts.py` — RAG 回答/自评/改写/裁判 prompt + 负面知识 + 固定拒答话术
- `app/tools/business.py` — `query_faq` 升级为 RAG 管线
- `app/tools/infra.py` / `app/tools/registry.py` — `ToolRun.citations/signal`、`query_faq` 长超时
- `app/core/agent.py` — 确定性拒答 + 自评落池 + citations 事件
- `app/api/chat.py` — SSE `citations` 事件转发
- `app/api/kb.py` — `/api/kb/search` 展示最终分
- `app/db/models.py` / `app/db/repository.py` — 两张新表 ORM + 写入
- `tests/conftest.py` — 加 ch04 DDL + 清表清单
- `tests/test_milvus_client.py` / `tests/test_dualwrite.py` / `tests/test_retrieval.py` / `tests/test_tools_infra.py` / `tests/test_chat_api.py` / `tests/test_llm.py` / `tests/test_prompts.py` — 随接口迁移
- `app/static/index.html` — 可点引用 + 👍/👎(Vibe Coding)
- `Makefile` — smoke/eval/check/verify/kb-revectorize
- `README.md` — ch04 用法与验收

**新:**
- `app/core/rerank.py`、`app/core/query_understanding.py`、`app/core/selfcheck.py`、`app/core/citations.py`、`app/core/judge.py`、`app/core/eval_metrics.py`
- `scripts/smoke_ch04.py`、`scripts/build_eval_set.py`、`scripts/check_judge.py`、`scripts/eval_ch04.py`、`scripts/verify_acceptance.py`
- `tests/data/eval_ch04.jsonl`、`tests/data/query_rewrite_samples.jsonl`、`tests/data/judge_samples.jsonl`
- `tests/test_rerank.py`、`tests/test_query_understanding.py`、`tests/test_selfcheck.py`、`tests/test_citations.py`、`tests/test_low_confidence.py`、`tests/test_faith_cases.py`、`tests/test_tool_faq_rag.py`(由 `test_tool_faq_vector.py` 改名)、`tests/test_agent_citations.py`、`tests/test_judge.py`、`tests/test_eval_metrics.py`、`tests/test_ch04_acceptance.py`
- `data/ch04/reports/rag_eval.txt`、`data/ch04/reports/rag_eval.json`(评估产物)

---

## Task 1: Fail-fast 冒烟风险闸(Milvus BM25 + 本地重排,GO/NO-GO)

**Files:**
- Create: `work/ch04/scripts/smoke_ch04.py`

**Interfaces:**
- Produces: 真实 Milvus 临时集合上的 BM25(chinese analyzer)/ `hybrid_search(RRFRanker)` 通过证据;本地 `bge-reranker-v2-m3` 加载、`compute_score(pairs, normalize=True)`、50 对 CPU 耗时;**GO/NO-GO 结论**。任一不通 → 停止后续任务并问用户。

**为什么排最前:** 这是 spec §8.3 的红线。Milvus 原生 BM25、chinese analyzer、pymilvus 3.0.2 的 `Function`/`AnnSearchRequest`/`RRFRanker`,以及本地重排模型,是本章仅有的两个高不确定外部依赖;先证伪,避免写了十个任务才发现选型跑不通。

- [ ] **Step 1: 起 Milvus + MySQL**

在 `work/ch03` 下起基础设施(此时 ch04 还没拷贝):

```bash
cd /e/mewhelp/work/ch03
make milvus-up
docker compose up -d mysql
curl -sf http://localhost:9091/healthz && echo "Milvus OK"
docker exec -i mewhelp-mysql mysqladmin ping -h localhost -proot
```

Expected: `Milvus OK`;MySQL 打印 `mysqld is alive`。

- [ ] **Step 2: 写冒烟脚本**

`work/ch04/scripts/smoke_ch04.py`(standalone:不 import `app.*`,这样在 ch04 工作区建好前也能跑):

```python
"""ch04 fail-fast 冒烟:Milvus 原生 BM25 + chinese analyzer + hybrid_search;
本地 bge-reranker-v2-m3 加载 + compute_score(normalize=True)+ CPU 50 对耗时。

运行(在 work/ch04 下,用 ch03 的 venv;首次下载模型时把 HF_HUB_OFFLINE 设为 0):
  HF_HOME='E:\\model' HF_HUB_OFFLINE=1 uv run --project ../ch03 python scripts/smoke_ch04.py

GO/NO-GO:任一断言失败即停,不要继续后面的任务;只在固定选型内换法,并记录到 dev-notes。
"""
import asyncio
import os
import time
import uuid

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

URI = os.environ.get("MILVUS_URI", "http://localhost:19530")


def smoke_milvus() -> None:
    import pymilvus

    print(f"pymilvus {pymilvus.__version__}")
    client = MilvusClient(uri=URI)
    coll = f"smoke_ch04_{uuid.uuid4().hex[:8]}"
    try:
        try:
            print(f"milvus server {client.get_server_version()}")
        except Exception:
            pass
        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=8)
        schema.add_field(
            "text", DataType.VARCHAR, max_length=2048,
            enable_analyzer=True, analyzer_params={"type": "chinese"},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="bm25_fn", input_field_names=["text"],
            output_field_names="sparse", function_type=FunctionType.BM25,
        ))
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX",
                               metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                               metric_type="BM25",
                               params={"bm25_k1": 1.2, "bm25_b": 0.75})
        client.create_collection(coll, schema=schema, index_params=index_params)
        client.insert(coll, [
            {"id": 1, "dense": [1.0, 0, 0, 0, 0, 0, 0, 0],
             "text": "智能猫砂盆 MH-LP100 废砂盒每 5 至 7 天倾倒一次"},
            {"id": 2, "dense": [0, 1.0, 0, 0, 0, 0, 0, 0],
             "text": "自动饮水机 MH-W40 滤芯每 3 周更换一次"},
            {"id": 3, "dense": [0, 0, 1.0, 0, 0, 0, 0, 0],
             "text": "满 99 元包邮,未满收取 10 元运费"},
        ])
        client.flush(coll)
        client.load_collection(coll)

        res = client.search(
            coll, data=["MH-LP100 废砂盒"], anns_field="sparse", limit=3,
            output_fields=["text"], search_params={"metric_type": "BM25"},
        )
        ids = [h["id"] for h in res[0]]
        print("BM25:", [(h["id"], round(float(h["distance"]), 4)) for h in res[0]])
        assert ids and ids[0] == 1, f"BM25 型号词未命中: {ids}"

        dense_req = AnnSearchRequest(
            data=[[1.0, 0, 0, 0, 0, 0, 0, 0]], anns_field="dense",
            param={"metric_type": "COSINE"}, limit=3,
        )
        bm25_req = AnnSearchRequest(
            data=["MH-W40 滤芯"], anns_field="sparse",
            param={"metric_type": "BM25"}, limit=3,
        )
        hybrid = client.hybrid_search(
            coll, [dense_req, bm25_req], RRFRanker(k=60), limit=3,
            output_fields=["text"],
        )
        print("hybrid:", [(h["id"], round(float(h["distance"]), 4)) for h in hybrid[0]])
        assert hybrid and len(hybrid[0]) >= 1, "hybrid_search 无结果"
        print("GO: Milvus BM25 + chinese analyzer + hybrid_search 通")
    finally:
        if client.has_collection(coll):
            client.drop_collection(coll)


def smoke_reranker() -> float:
    from FlagEmbedding import FlagReranker

    model_name = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    use_fp16 = os.environ.get("RERANK_DEVICE", "cpu") != "cpu"
    print(f"HF_HOME={os.environ.get('HF_HOME')} HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    t0 = time.perf_counter()
    reranker = FlagReranker(model_name, use_fp16=use_fp16)
    load_s = time.perf_counter() - t0

    query = "自动饮水机 MH-W40 的滤芯多久换一次"
    docs = [
        "MH-W40 建议每 3 周更换一次滤芯",
        "MH-W20 建议每 2 周更换一次滤芯",
        "满 99 元包邮,未满收取 10 元运费",
    ] + [f"无关文本 {i}" for i in range(47)]
    pairs = [[query, d] for d in docs]
    t0 = time.perf_counter()
    scores = reranker.compute_score(pairs, normalize=True)
    elapsed = time.perf_counter() - t0
    if not isinstance(scores, list):
        scores = [scores]
    assert len(scores) == 50, f"应返回 50 个分,实际 {len(scores)}"
    assert all(0.0 <= float(s) <= 1.0 for s in scores), "normalize=True 应落在 0~1"
    assert scores[0] > scores[1] > scores[2], f"相关度顺序不对: {scores[:3]}"
    print(f"GO: reranker 加载 {load_s:.1f}s;50 对 CPU 打分 {elapsed:.1f}s;"
          f"top3={[round(float(s), 4) for s in scores[:3]]}")
    return elapsed


async def main() -> None:
    await asyncio.to_thread(smoke_milvus)
    elapsed = await asyncio.to_thread(smoke_reranker)
    if elapsed > 60:
        print(f"WARN: 50 对重排耗时 {elapsed:.1f}s,评估全量会明显变慢;先用 --limit 迭代")
    print("GO: 两条链路都通")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: 跑冒烟**

```bash
cd /e/mewhelp/work/ch04
export HF_HOME='E:\model'
export HF_HUB_OFFLINE=1
uv run --project ../ch03 python scripts/smoke_ch04.py
```

Expected: 打印 `GO: Milvus ...`、`GO: reranker 加载 ...;50 对 CPU 打分 ...`、`GO: 两条链路都通`。
本机已实测:BM25 正确命中 `MH-LP100`;`hybrid_search` 返回 RRF 分;pymilvus 3.0.2;reranker 3 对约 0.26s(50 对预计 2~5s)。

- [ ] **Step 4: 处理 NO-GO(红线)**

- Milvus 报 schema/Function/analyzer 错误:确认镜像确实是 `milvusdb/milvus:v2.6.22`、pymilvus 是 3.0.2;只在 2.6.x 内换 tag。仍不通 → **停,问用户**,不换回 Lite、不弃 BM25。
- reranker 加载失败:若 `E:\model` 无缓存,首次用 `HF_HUB_OFFLINE=0` 下载(~2.3GB),完成后再设回 1;直连不稳时按用户既有约定处理,不换模型。仍不通 → **停,问用户**。
- 把实际版本、耗时、报错原文记到 `dev-notes/ch04.md` 顶部「⚠️ 红线卡点」。`dev-notes/` 已被 gitignore,不要提交。

- [ ] **Step 5: 提交**

```bash
cd /e/mewhelp
git add work/ch04/scripts/smoke_ch04.py
git commit -m "feat(ch04): fail-fast 冒烟 — Milvus BM25/hybrid + 本地 reranker 实测通过"
```

---

## Task 2: 工作区搭建(ch03 → ch04,测试库改名,139 测试基线)

**Files:**
- Create: `work/ch04/`(自 `work/ch03/` 拷贝,排除 `.venv`/缓存/临时目录)
- Modify: `work/ch04/pyproject.toml`(name = `ch04`)
- Modify: `work/ch04/.env`(本地,不入库)、`work/ch04/.env.example`、`work/ch04/app/config.py`、`work/ch04/tests/conftest.py`
- Modify: `work/ch04/Makefile`(`smoke-ch04` 目标 + HF 环境变量)

**Interfaces:**
- Consumes: Task 1 已创建 `work/ch04/scripts/smoke_ch04.py`(拷贝时不能被覆盖或删除)。
- Produces: `work/ch04/` 完整工作区;`settings.test_database_url` 指向 `mewhelp_ch04_test`;`uv sync` 后的 `.venv`;**139 个 ch03 测试全绿**的基线。

- [ ] **Step 1: 确认分支与干净状态**

```bash
cd /e/mewhelp
git branch --show-current
git status --short
```

Expected: 分支 `ch04-hybrid-retrieval-rerank-eval`。若有未提交改动,先按用户要求处理;本计划默认从干净基线开始。

- [ ] **Step 2: 拷贝并清理**

```bash
cd /e/mewhelp
mkdir -p work/ch04
tar -C work/ch03 \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='log' \
  --exclude='torchinductor_*' \
  --exclude='pytest-of-*' \
  -cf - . | tar -C work/ch04 -xf -

# 显式清理(防止 tar 的 exclude 没覆盖到)
find work/ch04 -type d -name '__pycache__' -prune -exec rm -rf {} +
rm -rf work/ch04/.venv work/ch04/.pytest_cache work/ch04/log
find work/ch04 -maxdepth 1 -type d \( -name 'torchinductor_*' -o -name 'pytest-of-*' \) -exec rm -rf {} +
# ch03 的实验残留,不进 ch04
rm -f work/ch04/models.json work/ch04/rr.out

ls work/ch04
test -f work/ch04/scripts/smoke_ch04.py && echo "Task 1 smoke survives"
```

Expected: 看到 `app/ tests/ scripts/ Makefile pyproject.toml uv.lock README.md docker-compose.yml .env .env.example`;`Task 1 smoke survives`。

- [ ] **Step 3: 改项目名与测试库名(三处)**

1. `work/ch04/pyproject.toml`:`name = "ch03"` → `name = "ch04"`。
2. `work/ch04/app/config.py` 默认值:
   ```python
   test_database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp_ch04_test"
   ```
3. `work/ch04/.env.example`:
   ```
   TEST_DATABASE_URL=mysql+asyncmy://root:root@localhost:3307/mewhelp_ch04_test
   ```
4. `work/ch04/.env`(本地真实文件,不入库):补上
   ```
   TEST_DATABASE_URL=mysql+asyncmy://root:root@localhost:3307/mewhelp_ch04_test
   ```
   若文件里已有旧的 `mewhelp_ch03_test`,替换掉。
5. `work/ch04/tests/conftest.py`:它用 `_DB_NAME = settings.test_database_url.rsplit("/", 1)[-1]`,**没有硬编码库名**;把注释里的 `mewhelp_ch03_test` 改成 `mewhelp_ch04_test`,并确认 Step 5 打印的是新库名。

- [ ] **Step 4: 同步环境**

```bash
cd /e/mewhelp/work/ch04
uv sync
uv run python -c "from app.config import settings; print(settings.test_database_url)"
```

Expected: 最后一行以 `.../mewhelp_ch04_test` 结尾。若 `uv lock` 因项目改名更新了 `uv.lock`,把 `uv.lock` 一起提交。

- [ ] **Step 5: 跑 ch03 基线(139 测试)**

```bash
cd /e/mewhelp/work/ch04
# 先确认基础容器已在跑(本机通常已由 ch03 起好);不要在这里重复 compose up
docker ps --filter name=mewhelp-mysql --format '{{.Names}} {{.Status}}'
docker ps --filter name=mewhelp-milvus --format '{{.Names}} {{.Status}}'
# 若没起:从 ch03 目录起(容器名固定,只能有一份)
# cd /e/mewhelp/work/ch03 && make milvus-up && docker compose up -d mysql
uv run pytest -q
```

Expected: **139 passed**(或 Milvus 未起时相关测试 skip;本任务要求 Milvus 起着,所以应全 passed)。若失败,先修环境/路径,不要继续。

- [ ] **Step 6: Makefile 加 HF 环境变量 + smoke 目标**

`work/ch04/Makefile` 顶部 `export PYTHONUTF8 = 1` 之后追加:

```makefile
# ch04:模型全在本地 E:\model;所有 target 与 make dev 共用
export HF_HOME ?= E:/model
export HF_HUB_OFFLINE ?= 1
```

`.PHONY` 行追加 `smoke-ch04`,并在 `test` 目标前加:

```makefile
smoke-ch04:
	PYTHONPATH=. uv run python scripts/smoke_ch04.py
```

运行:

```bash
cd /e/mewhelp/work/ch04
make smoke-ch04
```

Expected: 复现 Task 1 的两个 GO。

- [ ] **Step 7: 提交**

```bash
cd /e/mewhelp
git add work/ch04
git status --short
git commit -m "chore(ch04): 从 ch03 拷贝工作区 + 测试库切换 mewhelp_ch04_test + 139 测试基线"
```

注意:`work/ch04/.env` 被 gitignore,不会进提交;`work/ch04/.venv` 也不会。`git status` 里不应出现 `__pycache__` / `pytest-of-*` / `torchinductor_*`。

---

## Task 3: 配置与 LLM 温度参数

**Files:**
- Modify: `work/ch04/app/config.py`
- Modify: `work/ch04/app/core/llm.py`
- Test: `work/ch04/tests/test_config.py`(追加)、`work/ch04/tests/test_llm.py`(追加)

**Interfaces:**
- Produces:
  - `settings.rerank_model: str = "BAAI/bge-reranker-v2-m3"`
  - `settings.rerank_device: str = "cpu"`
  - `settings.recall_top_k: int = 50`
  - `settings.rerank_top_k: int = 10`
  - `settings.rerank_min_score: float = 0.3`
  - `get_chat_model(streaming: bool = False, temperature: float = 0.3) -> ChatOpenAI`(既有调用不传 temperature,行为不变;评估裁判显式传 `temperature=0`)

- [ ] **Step 1: 写失败测试**

`tests/test_config.py` 追加:

```python
def test_ch04_settings_defaults(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert s.test_database_url.endswith("/mewhelp_ch04_test")
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_device == "cpu"
    assert s.recall_top_k == 50
    assert s.rerank_top_k == 10
    assert s.rerank_min_score == 0.3
```

`tests/test_llm.py` 追加:

```python
def test_factory_temperature_override():
    assert get_chat_model(temperature=0).temperature == 0
    assert get_chat_model().temperature == 0.3
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_config.py::test_ch04_settings_defaults tests/test_llm.py::test_factory_temperature_override -q
```

Expected: FAIL(配置项不存在;`temperature` 参数不存在)。

- [ ] **Step 3: 实现**

`app/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 账号三件套故意不给默认值:缺了启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000

    database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp_ch04_test"

    embed_model: str = "BAAI/bge-m3"      # 本地模型名;首次加载从 HF 下 ~2GB
    embed_device: str = "cpu"             # cpu / cuda:0;非 cpu 才用 fp16 + devices
    milvus_uri: str = "http://localhost:19530"
    retrieval_top_k: int = 3              # ch03 /kb 兼容项;ch04 主链路用 rerank_top_k
    retrieval_min_score: float = 0.4      # ch03 兼容项;ch04 用 rerank_min_score

    # ch04 混合检索 + 本地重排
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cpu"
    recall_top_k: int = 50                # dense / BM25 各召回 Top-50
    rerank_top_k: int = 10                # 精排出 Top-10
    rerank_min_score: float = 0.3         # 机械闸:最高 rerank 分低于此 → 证据不足(不落池)


settings = Settings()
```

`app/core/llm.py`:

```python
from langchain_openai import ChatOpenAI

from app.config import settings


def get_chat_model(streaming: bool = False, temperature: float = 0.3) -> ChatOpenAI:
    """直连聊天上游,说 OpenAI 协议。地址、模型名、密钥全从 .env 来,换上游不动这里。
    评估裁判显式传 temperature=0,消除同一输入重放的抖动。"""
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=temperature,
    )
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_config.py tests/test_llm.py -q
```

Expected: passed。

- [ ] **Step 5: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/config.py work/ch04/app/core/llm.py work/ch04/tests/test_config.py work/ch04/tests/test_llm.py
git commit -m "feat(ch04): config 加 rerank/召回参数 + get_chat_model 支持 temperature"
```

---

## Task 4: 存储层迁移 — Milvus 新 schema + 双写元数据 + dense 检索兼容

**Files:**
- Modify: `work/ch04/app/kb/milvus_client.py`(全文替换)
- Modify: `work/ch04/app/kb/dualwrite.py`(向量化写入新字段)
- Modify: `work/ch04/app/core/retrieval.py`(临时保持 dense 单路,签名兼容,供 Task 7 再重写)
- Test: `work/ch04/tests/test_milvus_client.py`(全文替换)、`work/ch04/tests/test_dualwrite.py`(追加一条真实集合断言)、`work/ch04/tests/test_retrieval.py`(fixture 改新字段)

**Interfaces:**
- Consumes: `settings.milvus_uri`、`documents.Chunk`、`repository.list_pending_chunks/mark_chunk_vectorized`、`embeddings.embed_texts/embed_query`。
- Produces(后续任务严格按这些签名用):
  - `COLLECTION = "knowledge"`;`DIM = 1024`
  - `get_client(uri: str | None = None) -> MilvusClient`
  - `ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None`
  - `upsert_vectors(client: MilvusClient, rows: list[dict], collection: str = COLLECTION) -> None`;row = `{"id", "dense", "text", "question", "answer", "section_path", "content_type", "category"}`
  - `flush(client, collection=COLLECTION) -> None`
  - `dense_search(client, vector: list[float], top_k: int, category: str | None = None, collection=COLLECTION) -> list[dict]`
  - `bm25_search(client, text: str, top_k: int, category: str | None = None, collection=COLLECTION) -> list[dict]`
  - `hybrid_search(client, vector: list[float], text: str, top_k: int, recall: int | None = None, category: str | None = None, collection=COLLECTION) -> list[dict]`
  - `count(client, collection=COLLECTION) -> int`;`drop(client, collection=COLLECTION) -> None`
  - 每个 hit dict:`{"id": int, "score": float, "question": str, "answer": str, "section_path": str, "content_type": str, "category": str}`。`score` 对 dense 是 COSINE 相似度(越大越像),对 BM25 是 BM25 分,对 hybrid 是 RRF 融合分。
  - `dualwrite.vectorize_pending(client=None, batch_size=64, collection=COLLECTION) -> int`
  - **临时**:`retrieval.search_knowledge(query, top_k=None, min_score=None, client=None, collection=COLLECTION) -> list[dict]`(dense 单路,Task 7 重写为四策略;本任务先让旧 API 契约继续可用)

- [ ] **Step 1: 写失败测试(真实 Milvus 临时集合)**

`tests/test_milvus_client.py` 全文替换:

```python
import uuid

import pytest

from app.kb import milvus_client as mc

URI = "http://localhost:19530"


@pytest.fixture()
def milvus():
    """真实 Standalone 上的独立测试集合;连不上就 skip(先 make milvus-up)。"""
    try:
        c = mc.get_client(uri=URI)
        c.list_collections()
    except Exception as e:
        pytest.skip(f"Milvus 不可用,先 make milvus-up:{e}")
    coll = f"knowledge_test_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(c, collection=coll)
    yield c, coll
    mc.drop(c, collection=coll)


def _vec(seed: float) -> list[float]:
    v = [0.0] * mc.DIM
    v[0] = seed
    v[1] = 1.0 - seed
    return v


def _rows() -> list[dict]:
    return [
        {"id": 1, "dense": _vec(1.0), "text": "运费 满 99 元包邮 未满收取 10 元运费",
         "question": "运费怎么算", "answer": "满 99 元包邮,未满收取 10 元运费",
         "section_path": "商品与购物 FAQ / 运费怎么算", "content_type": "faq",
         "category": "商品与购物 FAQ"},
        {"id": 2, "dense": _vec(0.0), "text": "智能猫砂盆 MH-LP100 废砂盒每 5 至 7 天倾倒一次",
         "question": "智能猫砂盆 Pro(型号 MH-LP100)",
         "answer": "废砂盒每 5 至 7 天倾倒一次",
         "section_path": "商品规格手册 / 智能猫砂盆 Pro(型号 MH-LP100)",
         "content_type": "spec", "category": "商品规格手册"},
    ]


def test_upsert_then_dense_search_returns_fields(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.dense_search(c, _vec(0.98), top_k=1, collection=coll)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "运费怎么算"
    assert hits[0]["section_path"] == "商品与购物 FAQ / 运费怎么算"
    assert hits[0]["content_type"] == "faq"
    assert hits[0]["category"] == "商品与购物 FAQ"
    assert isinstance(hits[0]["score"], float)


def test_bm25_hits_model_number(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.bm25_search(c, "MH-LP100 废砂盒", top_k=2, collection=coll)
    assert hits and hits[0]["id"] == 2
    assert "MH-LP100" in hits[0]["section_path"]


def test_hybrid_returns_scored_hits(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.hybrid_search(c, _vec(1.0), "MH-LP100", top_k=2, recall=10,
                            collection=coll)
    assert hits
    assert {"id", "score", "question", "answer", "section_path",
            "content_type", "category"} <= hits[0].keys()


def test_category_filter(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.bm25_search(c, "运费 型号", top_k=5, category="商品与购物 FAQ",
                          collection=coll)
    assert hits
    assert all(h["category"] == "商品与购物 FAQ" for h in hits)


def test_upsert_is_idempotent_by_pk(milvus):
    c, coll = milvus
    row = _rows()[0]
    mc.upsert_vectors(c, [row], collection=coll)
    mc.upsert_vectors(c, [row], collection=coll)
    mc.flush(c, collection=coll)
    assert mc.count(c, collection=coll) == 1


def test_upsert_truncates_overlong_text(milvus):
    c, coll = milvus
    long_answer = "满 99 包邮。" * 2000       # 远超 MAX_ANSWER(8192)
    mc.upsert_vectors(c, [{
        "id": 7, "dense": _vec(1.0), "text": "运费", "question": "运费怎么算",
        "answer": long_answer, "section_path": "运费", "content_type": "faq",
        "category": "运费",
    }], collection=coll)
    mc.flush(c, collection=coll)
    assert mc.count(c, collection=coll) == 1
    hits = mc.dense_search(c, _vec(1.0), top_k=1, collection=coll)
    assert len(hits[0]["answer"]) <= mc.MAX_ANSWER


def test_identical_vector_scores_near_one(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, [_rows()[0]], collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.dense_search(c, _vec(1.0), top_k=1, collection=coll)
    assert hits[0]["score"] > 0.99
```

`tests/test_dualwrite.py` 追加(保留原有两条):

```python
async def test_vectorize_writes_text_and_metadata(db_session_factory, milvus, monkeypatch):
    c, coll = milvus

    async def fake_embed(texts):
        return [[0.05] * mc.DIM for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", fake_embed)
    await dualwrite.write_pending([
        Chunk(category="运费", questions="运费怎么算", answer="满 99 元包邮",
              section_path="商品与购物 FAQ / 运费怎么算", content_type="faq"),
    ])
    done = await dualwrite.vectorize_pending(c, collection=coll)
    assert done == 1
    hits = mc.bm25_search(c, "运费 包邮", top_k=1, collection=coll)
    assert hits and hits[0]["section_path"] == "商品与购物 FAQ / 运费怎么算"
    assert hits[0]["category"] == "运费"
```

`tests/test_retrieval.py` 的 fixture 改为新 row 结构(保留两条测试的语义):

```python
import uuid

import pytest

from app.core import retrieval
from app.kb import milvus_client as mc


@pytest.fixture()
def milvus():
    try:
        c = mc.get_client()
        c.list_collections()
    except Exception as e:
        pytest.skip(f"Milvus 不可用,先 make milvus-up:{e}")
    coll = f"knowledge_test_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(c, collection=coll)
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (mc.DIM - 3)
    mc.upsert_vectors(c, [
        {"id": 1, "dense": v_hit, "text": "运费 满 99 包邮",
         "question": "运费怎么算", "answer": "满 99 包邮",
         "section_path": "运费", "content_type": "faq", "category": "运费"},
        {"id": 2, "dense": v_other, "text": "发货时效 48 小时",
         "question": "发货时效", "answer": "48 小时",
         "section_path": "发货", "content_type": "faq", "category": "发货"},
    ], collection=coll)
    mc.flush(c, collection=coll)
    yield c, coll
    mc.drop(c, collection=coll)


async def _coro(v):
    return v


async def test_search_returns_top_hits_above_threshold(monkeypatch, milvus):
    c, coll = milvus
    qv = [1.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3)
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", lambda q: _coro(qv))
    hits = await retrieval.search_knowledge("邮费是多少", top_k=1, min_score=0.5,
                                            client=c, collection=coll)
    assert hits and hits[0]["question"] == "运费怎么算"


async def test_below_threshold_filtered(monkeypatch, milvus):
    c, coll = milvus
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query",
                        lambda q: _coro([1.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3)))
    hits = await retrieval.search_knowledge("邮费", top_k=2, min_score=0.999999,
                                            client=c, collection=coll)
    assert hits == [] or all(h["score"] >= 0.999999 for h in hits)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_milvus_client.py tests/test_dualwrite.py tests/test_retrieval.py -q
```

Expected: FAIL(`ensure_collection` 仍是旧 schema / 无 `dense_search`/`bm25_search`/`hybrid_search`)。

- [ ] **Step 3: 实现 `milvus_client.py`(全文替换)**

```python
"""Milvus Standalone(HTTP 19530)客户端:pymilvus MilvusClient 高层接口。

ch04 schema = dense(COSINE)+ text(chinese analyzer,BM25 Function 输入)+ sparse(BM25 输出)
+ question/answer/section_path/content_type/category 标量。
主键 id = MySQL knowledge_chunks.id(auto_id=False),重跑按 id upsert,天然幂等。

COSINE 相似度:Standalone 下 search 回的 distance 即相似度(越大越像),阈值直接比 score。
BM25 的 distance 是 BM25 分;hybrid 的 distance 是 RRF 融合分。三种 score 不可跨路比较。
"""
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024
MAX_TEXT = 16384
MAX_QUESTION = 2048
MAX_ANSWER = 8192
MAX_SECTION = 512
MAX_CONTENT_TYPE = 32
MAX_CATEGORY = 255

_CLIENT: MilvusClient | None = None

_OUTPUT = ["question", "answer", "section_path", "content_type", "category"]


def get_client(uri: str | None = None) -> MilvusClient:
    """默认返回进程内单例(连 Standalone);传 uri 返回独立客户端(测试/临时)。"""
    global _CLIENT
    if uri is not None:
        return MilvusClient(uri=uri)
    if _CLIENT is None:
        _CLIENT = MilvusClient(uri=settings.milvus_uri)
    return _CLIENT


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """幂等建集合:dense + text(chinese analyzer)+ sparse(BM25 Function)+ 标量字段。
    已存在则只确保 load。"""
    if client.has_collection(collection):
        client.load_collection(collection)
        return
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=MAX_TEXT,
        enable_analyzer=True, analyzer_params={"type": "chinese"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("question", DataType.VARCHAR, max_length=MAX_QUESTION)
    schema.add_field("answer", DataType.VARCHAR, max_length=MAX_ANSWER)
    schema.add_field("section_path", DataType.VARCHAR, max_length=MAX_SECTION)
    schema.add_field("content_type", DataType.VARCHAR, max_length=MAX_CONTENT_TYPE)
    schema.add_field("category", DataType.VARCHAR, max_length=MAX_CATEGORY)
    schema.add_function(Function(
        name="bm25_fn", input_field_names=["text"],
        output_field_names="sparse", function_type=FunctionType.BM25,
    ))
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX",
                           metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                           metric_type="BM25",
                           params={"bm25_k1": 1.2, "bm25_b": 0.75})
    client.create_collection(collection, schema=schema, index_params=index_params)
    client.load_collection(collection)


def _fit(text: str, limit: int) -> str:
    """按 UTF-8 字节数裁剪:Milvus VARCHAR max_length 计字节(中文一字 3 字节)。"""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore")


def upsert_vectors(client: MilvusClient, rows: list[dict],
                   collection: str = COLLECTION) -> None:
    """按主键 upsert。所有 VARCHAR 字段先按字节裁剪,免得整批被 Milvus 拒。"""
    if not rows:
        return
    fitted = []
    for r in rows:
        fitted.append({
            "id": r["id"],
            "dense": r["dense"],
            "text": _fit(r["text"], MAX_TEXT),
            "question": _fit(r["question"], MAX_QUESTION),
            "answer": _fit(r["answer"], MAX_ANSWER),
            "section_path": _fit(r.get("section_path") or "", MAX_SECTION),
            "content_type": _fit(r.get("content_type") or "", MAX_CONTENT_TYPE),
            "category": _fit(r.get("category") or "", MAX_CATEGORY),
        })
    client.upsert(collection, fitted)


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    """刷盘:新 upsert 的数据须 flush 后才稳定可被 BM25/检索命中。"""
    client.flush(collection)


def _category_expr(category: str | None) -> str:
    if not category:
        return ""
    safe = category.replace("\\", "\\\\").replace('"', '\\"')
    return f'category == "{safe}"'


def _hit(h: dict) -> dict:
    e = h["entity"]
    return {
        "id": h["id"],
        "score": float(h["distance"]),
        "question": e["question"],
        "answer": e["answer"],
        "section_path": e["section_path"],
        "content_type": e["content_type"],
        "category": e["category"],
    }


def dense_search(client: MilvusClient, vector: list[float], top_k: int,
                 category: str | None = None,
                 collection: str = COLLECTION) -> list[dict]:
    res = client.search(
        collection, data=[vector], anns_field="dense", limit=top_k,
        output_fields=_OUTPUT, search_params={"metric_type": "COSINE"},
        filter=_category_expr(category),
    )
    return [_hit(h) for h in res[0]]


def bm25_search(client: MilvusClient, text: str, top_k: int,
                category: str | None = None,
                collection: str = COLLECTION) -> list[dict]:
    res = client.search(
        collection, data=[text], anns_field="sparse", limit=top_k,
        output_fields=_OUTPUT, search_params={"metric_type": "BM25"},
        filter=_category_expr(category),
    )
    return [_hit(h) for h in res[0]]


def hybrid_search(client: MilvusClient, vector: list[float], text: str, top_k: int,
                  recall: int | None = None, category: str | None = None,
                  collection: str = COLLECTION) -> list[dict]:
    """dense + BM25 各召回 recall 条,RRF(k=60)融合后取 top_k。"""
    recall = recall or settings.recall_top_k
    expr = _category_expr(category)
    dense_req = AnnSearchRequest(
        data=[vector], anns_field="dense",
        param={"metric_type": "COSINE"}, limit=recall, expr=expr,
    )
    bm25_req = AnnSearchRequest(
        data=[text], anns_field="sparse",
        param={"metric_type": "BM25"}, limit=recall, expr=expr,
    )
    res = client.hybrid_search(
        collection, [dense_req, bm25_req], RRFRanker(k=60),
        limit=top_k, output_fields=_OUTPUT,
    )
    return [_hit(h) for h in res[0]]


def count(client: MilvusClient, collection: str = COLLECTION) -> int:
    return client.query(collection, filter="id >= 0",
                        output_fields=["count(*)"])[0]["count(*)"]


def drop(client: MilvusClient, collection: str = COLLECTION) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)
```

- [ ] **Step 4: 实现 `dualwrite.py`(全文替换)**

```python
from app.core import embeddings
from app.db import repository
from app.kb import milvus_client
from app.kb.documents import Chunk


async def write_pending(chunks: list[Chunk]) -> list[int]:
    """按顺序把一份文档的 chunks 写入 MySQL(status=pending),并在同文档内连 prev/next 指针。"""
    ids: list[int] = []
    for c in chunks:
        cid = await repository.insert_knowledge_chunk(
            c.category, c.questions, c.answer,
            section_path=c.section_path, content_type=c.content_type,
            is_key_clause=c.is_key_clause,
        )
        ids.append(cid)
    for i, cid in enumerate(ids):
        prev_id = ids[i - 1] if i > 0 else None
        next_id = ids[i + 1] if i < len(ids) - 1 else None
        await repository.set_chunk_neighbors(cid, prev_id, next_id)
    return ids


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def vectorize_pending(client=None, batch_size: int = 64,
                            collection: str = milvus_client.COLLECTION) -> int:
    """幂等可重跑:取 pending → 拼 category+questions+answer → 嵌入 + BM25 text
    → Milvus upsert(PK=id)→ flush → 回填 vector_id、status=done。
    崩在任意批,重跑只捡剩余 pending(Milvus 按 id upsert 无重)。"""
    pending = await repository.list_pending_chunks()
    done = 0
    c = client or milvus_client.get_client()
    milvus_client.ensure_collection(c, collection=collection)
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        rows = [
            {
                "id": r.id, "dense": v, "text": t,
                "question": r.questions, "answer": r.answer,
                "section_path": r.section_path or "",
                "content_type": r.content_type or "",
                "category": r.category or "",
            }
            for r, v, t in zip(batch, vectors, texts)
        ]
        milvus_client.upsert_vectors(c, rows, collection=collection)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    if done:
        milvus_client.flush(c, collection=collection)
    return done
```

- [ ] **Step 5: 实现 `retrieval.py` 的临时 dense 兼容层**

```python
from app.config import settings
from app.core import embeddings
from app.kb import milvus_client


async def search_knowledge(
    query: str, top_k: int | None = None, min_score: float | None = None,
    client=None, collection: str = milvus_client.COLLECTION,
) -> list[dict]:
    """临时:dense 单路迁移版。Task 7 会重写为四策略(strategy 参数)。
    Standalone COSINE 的 score 即相似度(越大越像),故直接 score >= min_score。"""
    top_k = top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if min_score is None else min_score
    vector = await embeddings.embed_query(query)
    c = client or milvus_client.get_client()
    milvus_client.ensure_collection(c, collection=collection)
    hits = milvus_client.dense_search(c, vector, top_k, collection=collection)
    return [h for h in hits if h["score"] >= min_score]
```

- [ ] **Step 6: 跑测试确认通过 + 全量回归**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_milvus_client.py tests/test_dualwrite.py tests/test_retrieval.py -q
uv run pytest -q
```

Expected: 目标测试通过;全量 139 tests 仍绿(新增测试后总数略增)。

- [ ] **Step 7: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/kb/milvus_client.py work/ch04/app/kb/dualwrite.py work/ch04/app/core/retrieval.py \
  work/ch04/tests/test_milvus_client.py work/ch04/tests/test_dualwrite.py work/ch04/tests/test_retrieval.py
git commit -m "feat(ch04): Milvus 新 schema(BM25/chinese analyzer)+ 双写元数据 + dense 迁移"
```

---

## Task 5: 本地重排客户端(FlagReranker + asyncio.to_thread)

**Files:**
- Create: `work/ch04/app/core/rerank.py`
- Test: `work/ch04/tests/test_rerank.py`

**Interfaces:**
- Consumes: `settings.rerank_model`、`settings.rerank_device`。
- Produces:
  - `_model() -> FlagReranker`(惰性单例,测试可 monkeypatch)
  - `_score_pairs(query: str, docs: list[str]) -> list[float]`(同步;调 `compute_score(pairs, normalize=True)`,把单 float 归一成 list)
  - `async rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]`(返回原始索引 + 分,按分降序,截断 top_n;空 docs → `[]`)

- [ ] **Step 1: 写失败测试**

`tests/test_rerank.py`:

```python
import pytest

from app.core import rerank as rr


@pytest.mark.asyncio
async def test_rerank_orders_and_truncates(monkeypatch):
    monkeypatch.setattr(rr, "_score_pairs", lambda q, docs: [0.1, 0.9, 0.5])
    out = await rr.rerank("q", ["a", "b", "c"], top_n=2)
    assert out == [(1, 0.9), (2, 0.5)]


@pytest.mark.asyncio
async def test_rerank_empty_docs():
    assert await rr.rerank("q", []) == []


def test_score_pairs_normalizes_single_float(monkeypatch):
    class FakeModel:
        def compute_score(self, pairs, normalize=True):
            assert normalize is True
            assert pairs == [["q", "a"]]
            return 0.7

    monkeypatch.setattr(rr, "_model", lambda: FakeModel())
    assert rr._score_pairs("q", ["a"]) == [0.7]


@pytest.mark.asyncio
async def test_rerank_uses_to_thread(monkeypatch):
    calls = {}

    async def fake_to_thread(fn, *args):
        calls["fn"] = fn
        calls["args"] = args
        return fn(*args)

    monkeypatch.setattr(rr.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(rr, "_score_pairs", lambda q, docs: [0.2, 0.8])
    out = await rr.rerank("q", ["a", "b"])
    assert calls["fn"] is rr._score_pairs
    assert calls["args"] == ("q", ["a", "b"])
    assert out == [(1, 0.8), (0, 0.2)]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_rerank.py -q
```

Expected: FAIL(模块不存在)。

- [ ] **Step 3: 实现**

`app/core/rerank.py`:

```python
"""本地重排:进程内 FlagReranker(BAAI/bge-reranker-v2-m3),CPU 推理。

模型只在首次调用时加载(惰性单例);加载与 compute_score 都是阻塞的,
统一丢进 asyncio.to_thread,别卡住事件循环。
"""
import asyncio

from app.config import settings

_MODEL = None


def _model():
    """惰性单例。供测试 monkeypatch 成带 compute_score 的假对象。"""
    global _MODEL
    if _MODEL is None:
        from FlagEmbedding import FlagReranker

        _MODEL = FlagReranker(
            settings.rerank_model,
            use_fp16=settings.rerank_device != "cpu",
        )
    return _MODEL


def _score_pairs(query: str, docs: list[str]) -> list[float]:
    pairs = [[query, d] for d in docs]
    scores = _model().compute_score(pairs, normalize=True)
    if isinstance(scores, (int, float)):
        return [float(scores)]
    return [float(s) for s in scores]


async def rerank(query: str, docs: list[str],
                 top_n: int | None = None) -> list[tuple[int, float]]:
    """返回 [(原始索引, 相关分)] 按分降序,截断 top_n。空 docs 直接返回 []。
    normalize=True 的分落在 0~1,与 settings.rerank_min_score 同一个量纲。"""
    if not docs:
        return []
    scores = await asyncio.to_thread(_score_pairs, query, docs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n] if top_n else ranked
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_rerank.py -q
```

Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/core/rerank.py work/ch04/tests/test_rerank.py
git commit -m "feat(ch04): 本地 FlagReranker 重排客户端(to_thread + normalize)"
```

---

## Task 6: Query 理解(口语归一 + 同义词扩展,标注样例验证)

**Files:**
- Modify: `work/ch04/app/core/prompts.py`(追加 `QUERY_REWRITE_SYSTEM` / `QUERY_REWRITE_PROMPT`)
- Create: `work/ch04/app/core/query_understanding.py`
- Create: `work/ch04/tests/data/query_rewrite_samples.jsonl`
- Test: `work/ch04/tests/test_query_understanding.py`

**Interfaces:**
- Consumes: `get_chat_model()`、`QUERY_REWRITE_PROMPT`。
- Produces:
  - `class _Rewrite(BaseModel): standard: str; expanded: list[str]`
  - `def _chain(model=None) -> Runnable`
  - `async def understand(query: str, model=None) -> dict` → `{"standard": str, "expanded": list[str]}`。`standard` 为空时回退原 query;`expanded` 去空白、去空串。

**边界:** 改写只作用于检索侧;`standard` 用于 dense,`standard + expanded` 只用于 BM25。不改入库文本。

- [ ] **Step 1: 写失败单元测试**

`tests/test_query_understanding.py`:

```python
import pytest

from app.core import query_understanding as qu


@pytest.mark.asyncio
async def test_understand_parses_flat_fields(monkeypatch):
    class Fake:
        standard = "运费怎么算"
        expanded = ["邮费", "快递费"]

    class Chain:
        async def ainvoke(self, payload):
            assert payload == {"query": "那个邮费多少钱啊"}
            return Fake()

    monkeypatch.setattr(qu, "_chain", lambda model=None: Chain())
    out = await qu.understand("那个邮费多少钱啊")
    assert out == {"standard": "运费怎么算", "expanded": ["邮费", "快递费"]}


@pytest.mark.asyncio
async def test_understand_falls_back_to_raw_query(monkeypatch):
    class Fake:
        standard = ""
        expanded = ["", "  "]

    class Chain:
        async def ainvoke(self, payload):
            return Fake()

    monkeypatch.setattr(qu, "_chain", lambda model=None: Chain())
    out = await qu.understand("坏了咋整")
    assert out == {"standard": "坏了咋整", "expanded": []}
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_query_understanding.py -q
```

Expected: FAIL(模块不存在)。

- [ ] **Step 3: 实现 prompt + 模块**

`app/core/prompts.py` 追加:

```python
QUERY_REWRITE_SYSTEM = """你是电商客服检索前的 Query 归一化器。把用户口语、模糊、带情绪的问法改写成简洁标准的问法,并给出检索侧同义词/近义扩展词。

- standard:一句话标准问法,去口语和情绪,保留关键实体(型号、品类、政策词)。
- expanded:3-6 个与问题相关的同义词/近义词/别称(如「邮费↔运费」「多久到↔时效」),只列词,不含原词。
- 不臆造原问题没有的实体或型号;不确定时宁可少扩展。
- 扩展词只用于关键词召回,不改变 standard 的含义。
只输出 standard 与 expanded 两个字段,不要输出别的。"""

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QUERY_REWRITE_SYSTEM), ("human", "用户问法:{query}")]
)
```

`app/core/query_understanding.py`:

```python
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


def _chain(model=None):
    m = model or get_chat_model()
    # ch03 R11:思考模式下 function_calling 强制 tool_choice 会 400;
    # 关掉 thinking 再走 function_calling 是唯一实测可走的扁平结构化输出路径。
    return QUERY_REWRITE_PROMPT | m.with_structured_output(
        _Rewrite, method="function_calling",
        extra_body={"thinking": {"type": "disabled"}},
    )


async def understand(query: str, model=None) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str]),避开嵌套对象数组 502。"""
    r: _Rewrite = await _chain(model).ainvoke({"query": query})
    standard = (r.standard or "").strip() or query
    expanded = [w.strip() for w in (r.expanded or []) if w and w.strip()]
    return {"standard": standard, "expanded": expanded}
```

- [ ] **Step 4: 跑单元测试确认通过**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_query_understanding.py -q
```

Expected: 2 passed。

- [ ] **Step 5: 造标注样例并跑真实模型验证**

`tests/data/query_rewrite_samples.jsonl`:

```jsonl
{"query":"那个邮费到底要给多少钱啊急","expect_any":["运费","邮费"]}
{"query":"东西啥时候能到我手上","expect_any":["时效","多久","配送"]}
{"query":"Pro款猫砂盆能自动铲屎不","expect_any":["Pro","猫砂盆"]}
{"query":"不想要了能退不","expect_any":["退货","退款","无理由"]}
{"query":"坏了咋整","expect_any":["维修","售后","质量"]}
{"query":"MH-W40滤芯多久换","expect_any":["MH-W40","滤芯"]}
```

运行(真实 DeepSeek 上游,需要 `.env`):

```bash
cd /e/mewhelp/work/ch04
PYTHONPATH=. uv run python -c "
import asyncio, json
from app.core.query_understanding import understand
async def main():
    ok = tot = 0
    for ln in open('tests/data/query_rewrite_samples.jsonl', encoding='utf-8'):
        s = json.loads(ln); tot += 1
        r = await understand(s['query'])
        blob = r['standard'] + ' ' + ' '.join(r['expanded'])
        hit = any(w in blob for w in s['expect_any'])
        print('OK' if hit else 'MISS', s['query'], '->', r)
        ok += hit
    print(f'{ok}/{tot}')
    assert ok >= 5, '改写标注命中低于 5/6,先调 prompt'
asyncio.run(main())
"
```

Expected: `≥6/6` 命中(至少 5/6 才继续)。低于则调 `QUERY_REWRITE_SYSTEM` 后重跑。

- [ ] **Step 6: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/core/prompts.py work/ch04/app/core/query_understanding.py \
  work/ch04/tests/test_query_understanding.py work/ch04/tests/data/query_rewrite_samples.jsonl
git commit -m "feat(ch04): Query 理解(口语归一 + 同义词扩展,标注样例验证)"
```

---

## Task 7: 检索核心(四策略 + 首尾组装)+ 引用映射

**Files:**
- Modify: `work/ch04/app/core/retrieval.py`(全文替换)
- Create: `work/ch04/app/core/citations.py`
- Modify: `work/ch04/app/api/kb.py`(`/api/kb/search` 展示最终分与 section_path)
- Test: `work/ch04/tests/test_retrieval.py`(全文替换为 mock 单测)、`work/ch04/tests/test_citations.py`

**Interfaces:**
- Consumes: `embeddings.embed_query`、`milvus_client.{dense_search,bm25_search,hybrid_search,ensure_collection,get_client}`、`rerank.rerank`、`settings.{recall_top_k,rerank_top_k}`。
- Produces:
  - `STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")`
  - `arrange_head_tail(items: list) -> list`
  - `async search_knowledge(query: str, strategy: str = "hybrid_rerank", top_k: int | None = None, category: str | None = None, client=None, collection: str = milvus_client.COLLECTION, bm25_text: str | None = None) -> list[dict]`
    - `vector`: `embed_query(query)` → `dense_search`
    - `bm25`: `bm25_search(bm25_text or query)`(不调嵌入)
    - `hybrid`: `embed_query(query)` + `hybrid_search(query, bm25_text or query)`
    - `hybrid_rerank`: 在 hybrid 基础上 `rerank(query, [question + " " + answer])`,给每条 hit 加 `rerank_score`
    - 未知 strategy → `ValueError`
  - `citations.build_citations(hits: list[dict]) -> list[dict]` → `[{"n": 1..n, "chunk_id": int, "section_path": str, "question": str, "answer": str}]`
  - `citations.build_evidence(citations: list[dict]) -> str` → `"[1] question: answer\n[2] ..."`

**说明:** 这里是**确定性检索核心**,不含 LLM;评估脚本直接复用。Query 理解在 `query_faq` 里做,不进入 `search_knowledge`——否则检索段评估会依赖 LLM,违反 spec §5「检索段与证据覆盖度不依赖 LLM」。

- [ ] **Step 1: 写失败测试**

`tests/test_retrieval.py` 全文替换:

```python
import pytest

from app.core import retrieval


def _hit(i: int, section: str, score: float = 0.5, rerank_score: float | None = None):
    h = {"id": i, "score": score, "question": f"q{i}", "answer": f"a{i}",
         "section_path": section, "content_type": "faq", "category": "c"}
    if rerank_score is not None:
        h["rerank_score"] = rerank_score
    return h


def test_arrange_head_tail():
    out = retrieval.arrange_head_tail(["a", "b", "c", "d", "e"])
    assert out[0] == "a" and out[-1] == "b"
    assert out[1:-1] == ["c", "d", "e"]


def test_arrange_head_tail_short():
    assert retrieval.arrange_head_tail(["a"]) == ["a"]
    assert retrieval.arrange_head_tail(["a", "b"]) == ["a", "b"]


async def test_strategy_vector(monkeypatch):
    async def fake_embed(q):
        return [0.1] * 1024

    captured = {}

    def fake_dense(client, vec, top_k, category=None, collection="knowledge"):
        captured.update(vec=vec, top_k=top_k, category=category)
        return [_hit(1, "运费")]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.dense_search", fake_dense)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="vector", top_k=3,
                                           category="运费", client=object())
    assert out[0]["id"] == 1
    assert captured["top_k"] == 3 and captured["category"] == "运费"


async def test_strategy_bm25_uses_bm25_text(monkeypatch):
    captured = {}

    def fake_bm25(client, text, top_k, category=None, collection="knowledge"):
        captured["text"] = text
        return [_hit(2, "型号")]

    monkeypatch.setattr("app.kb.milvus_client.bm25_search", fake_bm25)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("Pro 型号", strategy="bm25",
                                           bm25_text="Pro 型号 猫砂盆",
                                           client=object())
    assert out[0]["id"] == 2
    assert captured["text"] == "Pro 型号 猫砂盆"


async def test_strategy_hybrid(monkeypatch):
    async def fake_embed(q):
        return [0.1] * 1024

    captured = {}

    def fake_hybrid(client, vec, text, top_k, recall=None, category=None,
                    collection="knowledge"):
        captured.update(vec=vec, text=text, top_k=top_k, recall=recall)
        return [_hit(1, "运费", 0.9)]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="hybrid", top_k=5,
                                           client=object())
    assert out[0]["id"] == 1
    assert captured["text"] == "运费" and captured["recall"] == 50


async def test_strategy_hybrid_rerank(monkeypatch):
    async def fake_embed(q):
        return [0.1] * 1024

    def fake_hybrid(client, vec, text, top_k, recall=None, category=None,
                    collection="knowledge"):
        return [_hit(1, "运费", 0.5), _hit(2, "型号", 0.4)]

    async def fake_rerank(q, docs, top_n=None):
        assert len(docs) == 2
        return [(1, 0.95), (0, 0.2)]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("Pro 型号", strategy="hybrid_rerank",
                                           client=object())
    assert out[0]["id"] == 2 and out[0]["rerank_score"] == 0.95
    assert out[1]["id"] == 1


async def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        await retrieval.search_knowledge("x", strategy="nope")
```

`tests/test_citations.py`:

```python
from app.core import citations


def _hit(i, section):
    return {"id": i, "question": f"q{i}", "answer": f"a{i}",
            "section_path": section, "content_type": "faq", "category": "c",
            "score": 0.5}


def test_build_citations_numbers_and_maps():
    out = citations.build_citations([_hit(5, "运费"), _hit(9, "退货")])
    assert [c["n"] for c in out] == [1, 2]
    assert out[0]["chunk_id"] == 5 and out[1]["chunk_id"] == 9
    assert out[0]["section_path"] == "运费"


def test_build_citations_empty():
    assert citations.build_citations([]) == []


def test_build_evidence_format():
    cites = citations.build_citations([_hit(1, "运费"), _hit(2, "退货")])
    assert citations.build_evidence(cites) == "[1] q1: a1\n[2] q2: a2"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_retrieval.py tests/test_citations.py -q
```

Expected: FAIL(`strategy` 参数不存在;`citations` 模块不存在)。

- [ ] **Step 3: 实现 `retrieval.py`(全文替换)**

```python
from app.config import settings
from app.core import embeddings, rerank
from app.kb import milvus_client

STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def arrange_head_tail(items: list) -> list:
    """最相关放首、次相关放尾,其余按序居中(缓解 lost-in-the-middle)。"""
    if len(items) <= 2:
        return list(items)
    return [items[0], *items[2:], items[1]]


async def search_knowledge(
    query: str,
    strategy: str = "hybrid_rerank",
    top_k: int | None = None,
    category: str | None = None,
    client=None,
    collection: str = milvus_client.COLLECTION,
    bm25_text: str | None = None,
) -> list[dict]:
    """确定性检索核心(无 LLM):vector | bm25 | hybrid | hybrid_rerank。

    - query:标准问法,dense 路用。
    - bm25_text:关键词路文本(可为 standard + 同义词扩展);缺省用 query。
    - 返回最终排序的 hit;hybrid_rerank 额外带 rerank_score。
    - 不做首尾组装,调用侧按需 arrange_head_tail。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知检索策略:{strategy}")
    top_k = top_k or settings.rerank_top_k
    c = client or milvus_client.get_client()
    milvus_client.ensure_collection(c, collection=collection)

    if strategy == "vector":
        vec = await embeddings.embed_query(query)
        return milvus_client.dense_search(c, vec, top_k, category=category,
                                          collection=collection)
    if strategy == "bm25":
        return milvus_client.bm25_search(c, bm25_text or query, top_k,
                                         category=category, collection=collection)

    vec = await embeddings.embed_query(query)
    hits = milvus_client.hybrid_search(
        c, vec, bm25_text or query, top_k, recall=settings.recall_top_k,
        category=category, collection=collection,
    )
    if strategy == "hybrid":
        return hits

    docs = [f"{h['question']} {h['answer']}" for h in hits]
    ranked = await rerank.rerank(query, docs, top_n=top_k)
    out = []
    for idx, score in ranked:
        h = dict(hits[idx])
        h["rerank_score"] = score
        out.append(h)
    return out
```

`app/core/citations.py`:

```python
"""编号证据 ↔ chunk 元数据。n 从 1 开始,与证据文本、SSE citations 一致。"""


def build_citations(hits: list[dict]) -> list[dict]:
    return [
        {
            "n": i + 1,
            "chunk_id": h["id"],
            "section_path": h.get("section_path") or "",
            "question": h.get("question") or "",
            "answer": h.get("answer") or "",
        }
        for i, h in enumerate(hits)
    ]


def build_evidence(citations: list[dict]) -> str:
    return "\n".join(
        f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations
    )
```

`app/api/kb.py` 的 `/api/kb/search` 改为:

```python
@router.post("/search")
async def search(body: SearchIn) -> dict:
    """检索自测(默认 hybrid_rerank):换个说法问一句,看召回的是不是该召回的那块。"""
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="问一句话再检索")
    try:
        hits = await retrieval.search_knowledge(q, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"检索失败({type(e).__name__}: {e});Milvus 要在线") from e
    return {"q": q, "top_k": body.top_k,
            "hits": [{"id": h.get("id"), "question": h.get("question"),
                      "answer": h.get("answer"), "section_path": h.get("section_path"),
                      "score": h.get("rerank_score", h.get("score"))} for h in hits]}
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_retrieval.py tests/test_citations.py -q
uv run pytest -q
```

Expected: 目标测试通过;全量绿。

- [ ] **Step 5: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/core/retrieval.py work/ch04/app/core/citations.py \
  work/ch04/app/api/kb.py work/ch04/tests/test_retrieval.py work/ch04/tests/test_citations.py
git commit -m "feat(ch04): 四策略检索核心 + 首尾组装 + 编号引用映射"
```

---

## Task 8: 数据模型与仓储(low_confidence_questions + faith_cases)

**Files:**
- Modify: `work/ch04/app/db/models.py`(追加两个 ORM 模型)
- Modify: `work/ch04/app/db/repository.py`(追加写入/查询)
- Modify: `work/ch04/tests/conftest.py`(ch04 DDL + 清表清单)
- Test: `work/ch04/tests/test_low_confidence.py`、`work/ch04/tests/test_faith_cases.py`

**Interfaces:**
- Produces:
  - `repository.insert_low_confidence(conversation_id: int | None, raw_question: str, source: str, reason: str | None) -> int`
  - `repository.upsert_faith_case(eval_id: str, bucket: str, query: str, strategy: str, answer: str, reason: str, citations: list[dict] | None, judge_model: str | None) -> dict`
    - 新题:`{"id": int, "seen_count": 1, "relapsed": False, "status": "未解决"}`
    - 复发(原 status 非「未解决」):`seen_count+1`,status 退回「未解决」,`relapsed=True`,`resolved_at`/`resolution` 保留(DDL 注释:复发后仍保留 resolved_at)
  - `repository.list_faith_cases(limit: int = 100) -> list[FaithCase]`
  - ORM:`LowConfidenceQuestion`、`FaithCase`

- [ ] **Step 1: conftest 扩 DDL 与清表清单**

`tests/conftest.py`:

```python
_DDL_FILES = [
    pathlib.Path(__file__).resolve().parents[3] / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parents[3] / "sql" / "ch03-ddl.sql",
    pathlib.Path(__file__).resolve().parents[3] / "sql" / "ch04-ddl.sql",
]
# 删除顺序:先子表后父表;knowledge_chunks 自引用 FK 靠 FOREIGN_KEY_CHECKS=0 兜
_TABLES = ["faith_cases", "low_confidence_questions", "messages", "tickets",
           "conversations", "faq", "qa_extraction_staging", "knowledge_chunks"]
```

（`_DB_NAME` 仍从 `settings.test_database_url` 派生;确认注释已写 `mewhelp_ch04_test`。）

- [ ] **Step 2: 写失败测试**

`tests/test_low_confidence.py`:

```python
from sqlalchemy import select

from app.db import repository
from app.db.models import LowConfidenceQuestion


async def test_insert_low_confidence(db_session_factory):
    cid = await repository.create_conversation("u1")
    lid = await repository.insert_low_confidence(
        cid, "邮费到底多少啊", "self_check", "证据不足")
    assert lid > 0
    async with db_session_factory() as s:
        row = (await s.execute(
            select(LowConfidenceQuestion).where(LowConfidenceQuestion.id == lid)
        )).scalar_one()
    assert row.raw_question == "邮费到底多少啊"
    assert row.source == "self_check"
    assert row.reason == "证据不足"
    assert row.conversation_id == cid


async def test_insert_low_confidence_without_conversation(db_session_factory):
    lid = await repository.insert_low_confidence(
        None, "库外问题", "retrieval_low_conf", None)
    assert lid > 0
```

`tests/test_faith_cases.py`:

```python
from datetime import datetime

from sqlalchemy import select

from app.db import repository
from app.db.models import FaithCase

CITES = [{"n": 1, "chunk_id": 5, "section_path": "运费",
          "question": "运费怎么算", "answer": "满 99 包邮"}]


async def test_upsert_new_row(db_session_factory):
    r = await repository.upsert_faith_case(
        "A01", "A_policy", "满多少包邮", "hybrid_rerank",
        "答案", "编造了数值", CITES, "deepseek-v4-flash")
    assert r["seen_count"] == 1
    assert r["relapsed"] is False
    assert r["status"] == "未解决"
    async with db_session_factory() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == "A01")
        )).scalar_one()
    assert row.citations[0]["chunk_id"] == 5
    assert row.answer == "答案"


async def test_recurrence_updates_one_row(db_session_factory):
    first = await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a1", "r1", CITES, "m")
    second = await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a2", "r2", CITES, "m")
    assert second["id"] == first["id"]
    assert second["seen_count"] == 2
    assert second["relapsed"] is False
    rows = await repository.list_faith_cases()
    assert len(rows) == 1 and rows[0].answer == "a2"


async def test_resolved_then_recurrence_resets_status(db_session_factory):
    r = await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a1", "r1", CITES, "m")
    async with db_session_factory() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.id == r["id"])
        )).scalar_one()
        row.status = "已解决"
        row.resolution = "补充了知识"
        row.resolved_at = datetime.now()
        await s.commit()

    again = await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a2", "r2", CITES, "m")
    assert again["relapsed"] is True
    assert again["status"] == "未解决"
    assert again["seen_count"] == 2
    async with db_session_factory() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.id == r["id"])
        )).scalar_one()
    assert row.resolved_at is not None          # 复发后仍保留,用来标「复发」
    assert row.resolution == "补充了知识"
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_low_confidence.py tests/test_faith_cases.py -q
```

Expected: FAIL(表/模型/方法不存在)。

- [ ] **Step 4: 加 ORM 模型**

`app/db/models.py` 追加:

```python
class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=True)
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback"))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FaithCase(Base):
    __tablename__ = "faith_cases"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决"), server_default="未解决")
    seen_count: Mapped[int] = mapped_column(Integer, server_default="1")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
```

- [ ] **Step 5: 加 repository 方法**

`app/db/repository.py` 顶部 import 加 `FaithCase, LowConfidenceQuestion`,文件末尾追加:

```python
async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None
) -> int:
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


async def upsert_faith_case(
    eval_id: str, bucket: str, query: str, strategy: str, answer: str,
    reason: str, citations: list[dict] | None, judge_model: str | None,
) -> dict:
    """一题一行:新题插入;复发只更新并 seen_count+1,已处置的退回「未解决」。"""
    async with db.async_session() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id)
        )).scalar_one_or_none()
        now = datetime.now()
        if row is None:
            row = FaithCase(
                eval_id=eval_id, bucket=bucket, query=query, strategy=strategy,
                answer=answer, reason=reason, citations=citations,
                judge_model=judge_model, seen_count=1,
                first_seen_at=now, last_seen_at=now,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return {"id": row.id, "seen_count": row.seen_count,
                    "relapsed": False, "status": row.status}

        relapsed = row.status != "未解决"
        row.bucket = bucket
        row.query = query
        row.strategy = strategy
        row.answer = answer
        row.reason = reason
        row.citations = citations
        row.judge_model = judge_model
        row.seen_count = (row.seen_count or 0) + 1
        row.last_seen_at = now
        if relapsed:
            row.status = "未解决"
        await s.commit()
        await s.refresh(row)
        return {"id": row.id, "seen_count": row.seen_count,
                "relapsed": relapsed, "status": row.status}


async def list_faith_cases(limit: int = 100) -> list[FaithCase]:
    async with db.async_session() as s:
        result = await s.execute(
            select(FaithCase)
            .order_by(FaithCase.last_seen_at.desc(), FaithCase.id.desc())
            .limit(limit)
        )
        return list(result.scalars())
```

- [ ] **Step 6: 把权威 DDL 落到真实库 `mewhelp`**

```bash
cd /e/mewhelp/work/ch04
docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SHOW TABLES LIKE 'low_confidence_questions'; SHOW TABLES LIKE 'faith_cases';"
docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < ../../sql/ch04-ddl.sql
docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SHOW TABLES LIKE 'low_confidence_questions'; SHOW TABLES LIKE 'faith_cases';"
```

Expected: 第二次 SHOW 两张表都在。若之前已建过,`CREATE TABLE` 会报 `already exists`——这是幂等边界;不要改权威 DDL,先确认表结构一致即可。

- [ ] **Step 7: 跑测试确认通过 + 全量回归**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_low_confidence.py tests/test_faith_cases.py -q
uv run pytest -q
```

Expected: 5 passed;全量绿。

- [ ] **Step 8: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/db/models.py work/ch04/app/db/repository.py work/ch04/tests/conftest.py \
  work/ch04/tests/test_low_confidence.py work/ch04/tests/test_faith_cases.py
git commit -m "feat(ch04): low_confidence_questions + faith_cases 模型/仓储 + conftest 扩表"
```

---

## Task 9: RAG 生成/自评 Prompt + 负面知识 + 固定拒答话术

**Files:**
- Modify: `work/ch04/app/core/prompts.py`
- Create: `work/ch04/app/core/selfcheck.py`
- Test: `work/ch04/tests/test_selfcheck.py`、`work/ch04/tests/test_prompts.py`(追加)

**Interfaces:**
- Produces:
  - `prompts.RAG_ANSWER_SYSTEM` / `RAG_ANSWER_PROMPT`(评测生成段用)
  - `prompts.SELF_CHECK_SYSTEM` / `SELF_CHECK_PROMPT`
  - `prompts.EVIDENCE_REFUSAL`(固定拒答话术)
  - `selfcheck.check_sufficient(query: str, evidence_texts: list[str], model=None) -> dict` → `{"useful": bool, "reason": str}`

- [ ] **Step 1: 写失败测试**

`tests/test_selfcheck.py`:

```python
import pytest

from app.core import selfcheck


@pytest.mark.asyncio
async def test_check_sufficient_parses_flat(monkeypatch):
    class Fake:
        useful = False
        reason = "证据只讲运费,未涉及型号"

    class Chain:
        async def ainvoke(self, payload):
            assert "[1] 满 99 包邮" in payload["evidence"]
            assert "[2] 七天无理由" in payload["evidence"]
            return Fake()

    monkeypatch.setattr(selfcheck, "_chain", lambda model=None: Chain())
    out = await selfcheck.check_sufficient(
        "Pro 型号功能", ["满 99 包邮", "七天无理由"])
    assert out == {"useful": False, "reason": "证据只讲运费,未涉及型号"}


@pytest.mark.asyncio
async def test_check_sufficient_empty_evidence(monkeypatch):
    class Fake:
        useful = False
        reason = "无证据"

    class Chain:
        async def ainvoke(self, payload):
            assert payload["evidence"] == "(无证据)"
            return Fake()

    monkeypatch.setattr(selfcheck, "_chain", lambda model=None: Chain())
    assert (await selfcheck.check_sufficient("q", []))["useful"] is False
```

`tests/test_prompts.py` 追加:

```python
def test_rag_answer_prompt_has_citations_and_negative_knowledge():
    from app.core.prompts import RAG_ANSWER_SYSTEM

    for kw in ("[n]", "[1][2]", "证据", "以平台售后规则为准", "不要编造"):
        assert kw in RAG_ANSWER_SYSTEM


def test_evidence_refusal_is_fixed():
    from app.core.prompts import EVIDENCE_REFUSAL

    assert "没有查到" in EVIDENCE_REFUSAL
    assert "人工客服" in EVIDENCE_REFUSAL
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_selfcheck.py tests/test_prompts.py -q
```

Expected: FAIL(模块/常量不存在)。

- [ ] **Step 3: 实现 prompts**

`app/core/prompts.py` 追加:

```python
EVIDENCE_REFUSAL = (
    "抱歉,我在知识库里暂时没有查到能可靠回答这个问题的依据。"
    "建议您换个说法再问一次,或通过「喵喵优选」App 的「我的」-「联系客服」转人工"
    "(每日 9:00 至 22:00)。售后时效以平台售后规则为准。"
)

RAG_ANSWER_SYSTEM = """你是「喵喵优选」电商平台的智能客服「小喵」。下面提供了带编号的知识证据,请严格依据证据回答用户问题。

## 引用规则
- 答案里每个关键结论后标注来源编号,如「满 99 元包邮[1]」;多个来源写成 [1][2],不要写 [1,2]。
- 编号对应证据的序号;只使用提供的证据作答,不要编造证据之外的信息。

## 拒答规则
- 若证据不足以回答用户问题,明确告知「暂时没有查到相关信息」并引导用户联系人工客服,不要硬编答案。

## 禁止承诺(负面知识,必须遵守)
- 不承诺具体到账时间、到货/配送时间、维修时长等时效;统一表述「以平台实际处理为准」。
- 不承诺赔偿金额或赔付时效;退款政策统一「以平台售后规则为准」。
- 不臆造订单、物流、库存、价格;无权限转接/提交工单时引导用户走 App 人工客服入口。
- 语气亲切专业、简洁,中文作答。"""

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", RAG_ANSWER_SYSTEM),
     ("human", "用户问题:{query}\n\n知识证据:\n{evidence}")]
)

SELF_CHECK_SYSTEM = """你是检索质量评审员。给定用户问题和检索到的知识证据,判断这些证据是否足以准确回答该问题。

- useful=true:证据包含回答该问题所需的关键信息。
- useful=false:证据与问题无关、或缺少关键信息、或只能部分回答核心诉求。
- reason:一句话说明判断依据。

严格只看证据是否够答,不要脑补证据外的知识。"""

SELF_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SELF_CHECK_SYSTEM),
     ("human", "用户问题:{query}\n\n检索证据:\n{evidence}")]
)
```

同时把 `AGENT_SYSTEM` 全文替换为下面这版(保留原有工具原则,新增「知识库回答规则」):

```python
AGENT_SYSTEM = """你是「喵喵优选」电商平台的智能客服「小喵」。你可以调用工具查询真实数据来回答用户。

## 工具使用原则
- 需要订单/商品/物流的具体信息时,调用对应工具查询(query_order / query_product / query_logistics),不要臆造数据。
- 用户咨询政策、规则、操作流程等通用问题时,用 query_faq 按关键词检索常见问答。
- 用户明确要求人工介入、投诉、或问题无法自助解决时,用 create_ticket 建人工工单(工单关联的会话号由系统填写,你不要编造)。
- 能直接回答的闲聊或超出电商客服范围的问题,礼貌回应或引导回购物话题,不必调用工具。
- 拿到工具结果后,用简洁、亲切、专业的中文组织回答;工具查不到时如实告知并给出下一步建议,不要编造。
- 退款/售后时效统一表述为「以平台售后规则为准」,不承诺无法保证的赔偿。

## 知识库回答规则
- 用户咨询政策、规则、操作流程、商品规格等通用问题时,用 query_faq 检索知识库。
- query_faq 返回的 evidence 是带 [n] 编号的知识证据;回答里的关键结论后必须标注来源编号,多个来源写成 [1][2],不要写 [1,2]。
- 只依据 evidence 作答,不要编造证据之外的信息;证据不足时如实说明。
- 如果工具返回 sufficient=false 或 signal=evidence_insufficient,必须直接拒答,不要用自身知识补全。
- 不承诺具体到账、到货、配送、维修时效;退款政策统一表述为「以平台售后规则为准」。"""
```

- [ ] **Step 4: 实现 selfcheck**

`app/core/selfcheck.py`:

```python
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import SELF_CHECK_PROMPT


class _Check(BaseModel):
    useful: bool = Field(description="证据是否足以回答")
    reason: str = Field(default="", description="判断依据")


def _chain(model=None):
    m = model or get_chat_model()
    return SELF_CHECK_PROMPT | m.with_structured_output(
        _Check, method="function_calling",
        extra_body={"thinking": {"type": "disabled"}},
    )


async def check_sufficient(query: str, evidence_texts: list[str],
                           model=None) -> dict:
    """生成前证据充分性自评。扁平字段 useful/reason(避开嵌套对象数组 502)。"""
    evidence = "\n".join(
        f"[{i + 1}] {t}" for i, t in enumerate(evidence_texts)
    ) or "(无证据)"
    r: _Check = await _chain(model).ainvoke({"query": query, "evidence": evidence})
    return {"useful": bool(r.useful), "reason": r.reason or ""}
```

- [ ] **Step 5: 跑测试确认通过 + 全量回归**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_selfcheck.py tests/test_prompts.py -q
uv run pytest -q
```

Expected: 目标测试通过;全量绿。

- [ ] **Step 6: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/core/prompts.py work/ch04/app/core/selfcheck.py \
  work/ch04/tests/test_selfcheck.py work/ch04/tests/test_prompts.py
git commit -m "feat(ch04): RAG 生成/自评 prompt + 负面知识 + 固定拒答话术"
```

---

## Task 10: query_faq 升级为 RAG 管线(两闸 + 编号证据 + 信号)

**Files:**
- Modify: `work/ch04/app/tools/business.py`(替换 `FaqInput` + `query_faq`)
- Rename: `work/ch04/tests/test_tool_faq_vector.py` → `work/ch04/tests/test_tool_faq_rag.py`(全文替换)

**Interfaces:**
- Consumes: `query_understanding.understand`、`retrieval.search_knowledge`、`retrieval.arrange_head_tail`、`selfcheck.check_sufficient`、`citations.build_citations/build_evidence`、`settings.rerank_min_score`。
- Produces:
  - `class FaqInput(BaseModel): keyword: str; category: str | None = None`
  - `async query_faq(keyword: str, category: str | None = None) -> dict`
    - 足:`{"sufficient": True, "evidence": "[1] ...\n[2] ...", "citations": [{"n", "chunk_id", "section_path", "question", "answer"}, ...]}`
    - 不足:`{"sufficient": False, "signal": "evidence_insufficient", "source": "retrieval_low_conf" | "self_check", "reason": str, "citations": []}`
  - **管线顺序**:改写 → `search_knowledge(strategy="hybrid_rerank", bm25_text=standard+expanded)` → 机械闸(top rerank 分) → 自评闸 → 首尾组装 → 编号。
  - **鲁棒性决定**:改写失败回退原 query;自评调用异常按「证据可用」fail-open 并记 warning(异常不是 `useful=false`,不落池)。

- [ ] **Step 1: 改名并写失败测试**

```bash
cd /e/mewhelp
git mv work/ch04/tests/test_tool_faq_vector.py work/ch04/tests/test_tool_faq_rag.py
```

`tests/test_tool_faq_rag.py` 全文:

```python
import pytest

from app.tools import business


def _hit(i, section, rerank=0.9, category="c"):
    return {"id": i, "score": 0.5, "rerank_score": rerank,
            "question": f"q{i}", "answer": f"a{i}",
            "section_path": section, "content_type": "faq", "category": category}


@pytest.fixture(autouse=True)
def _default_understanding(monkeypatch):
    async def fake_understand(q, model=None):
        return {"standard": q, "expanded": ["运费"]}

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)


async def test_query_faq_sufficient_numbers_after_head_tail(monkeypatch):
    async def fake_search(query, **kw):
        assert kw["strategy"] == "hybrid_rerank"
        assert kw["bm25_text"] == "邮费多少 运费"
        return [_hit(1, "运费政策"), _hit(2, "退货政策", 0.7),
                _hit(3, "会员政策", 0.6)]

    async def fake_check(q, texts, model=None):
        return {"useful": True, "reason": "够"}

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)

    out = await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert out["sufficient"] is True
    # 首尾组装:1 首、2 尾、3 居中 → 编号顺序 1,3,2
    assert [c["chunk_id"] for c in out["citations"]] == [1, 3, 2]
    assert [c["n"] for c in out["citations"]] == [1, 2, 3]
    assert out["evidence"].startswith("[1] q1: a1")
    assert "[2] q3: a3" in out["evidence"]
    assert "[3] q2: a2" in out["evidence"]


async def test_query_faq_understanding_falls_back(monkeypatch):
    async def boom(q, model=None):
        raise RuntimeError("上游挂了")

    async def fake_search(query, **kw):
        assert query == "原话问题"
        return [_hit(1, "p")]

    async def fake_check(q, texts, model=None):
        return {"useful": True, "reason": "够"}

    monkeypatch.setattr("app.core.query_understanding.understand", boom)
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "原话问题"})
    assert out["sufficient"] is True


async def test_query_faq_no_hits_mechanical_gate(monkeypatch):
    async def fake_search(query, **kw):
        return []

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    out = await business.query_faq.ainvoke({"keyword": "火星车"})
    assert out["sufficient"] is False
    assert out["signal"] == "evidence_insufficient"
    assert out["source"] == "retrieval_low_conf"
    assert out["citations"] == []


async def test_query_faq_low_score_mechanical_gate(monkeypatch):
    async def fake_search(query, **kw):
        return [_hit(1, "p", rerank=0.1)]

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    out = await business.query_faq.ainvoke({"keyword": "火星车"})
    assert out["source"] == "retrieval_low_conf"
    assert "top=0.100" in out["reason"]


async def test_query_faq_self_check_gate(monkeypatch):
    async def fake_search(query, **kw):
        return [_hit(1, "运费", rerank=0.8)]

    async def fake_check(q, texts, model=None):
        return {"useful": False, "reason": "问型号但证据只讲运费"}

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "Pro 型号能自动铲屎吗"})
    assert out["sufficient"] is False
    assert out["source"] == "self_check"
    assert out["signal"] == "evidence_insufficient"
    assert "型号" in out["reason"]


async def test_query_faq_self_check_exception_fails_open(monkeypatch):
    async def fake_search(query, **kw):
        return [_hit(1, "运费", rerank=0.8)]

    async def boom(q, texts, model=None):
        raise RuntimeError("自评上游挂了")

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", boom)
    out = await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert out["sufficient"] is True


async def test_query_faq_passes_category(monkeypatch):
    captured = {}

    async def fake_search(query, **kw):
        captured.update(kw)
        return [_hit(1, "商品规格手册 / MH-W40")]

    async def fake_check(q, texts, model=None):
        return {"useful": True, "reason": "够"}

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    await business.query_faq.ainvoke(
        {"keyword": "MH-W40 滤芯", "category": "商品规格手册"})
    assert captured["category"] == "商品规格手册"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_tool_faq_rag.py -q
```

Expected: FAIL(`query_faq` 仍返回旧 `{"hits": ...}`)。

- [ ] **Step 3: 实现 query_faq**

`app/tools/business.py` 顶部 import 改为:

```python
import logging
import random
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from app.config import settings
from app.core import citations, query_understanding, retrieval, selfcheck
from app.db import repository

logger = logging.getLogger(__name__)
```

替换 `FaqInput` + `query_faq`:

```python
class FaqInput(BaseModel):
    keyword: str = Field(description="用户咨询的政策/规则/操作/商品规格类问题(可用原话)")
    category: str | None = Field(
        default=None,
        description="可选:按 category 过滤。一般留空;只有用户明确说品类时才填,如『商品规格手册』『退货退款政策』",
    )


@tool(args_schema=FaqInput)
async def query_faq(keyword: str, category: str | None = None) -> dict:
    """查询常见问题/政策/商品规格知识库(混合检索 + 重排)。

    返回带 [n] 编号的证据供作答引用。证据不足时返回 sufficient=False 与
    signal=evidence_insufficient,调用方必须直接拒答,不要用自身知识编造。
    """
    try:
        u = await query_understanding.understand(keyword)
    except Exception:
        logger.warning("Query 改写失败,回退原话", exc_info=True)
        u = {"standard": keyword, "expanded": []}

    standard = u["standard"] or keyword
    bm25_text = standard + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        standard, strategy="hybrid_rerank", category=category, bm25_text=bm25_text)

    # ⑤ 机械闸:无命中 or 最高 rerank 分低于阈值 → 拒答,不落池
    top_score = float(hits[0].get("rerank_score", 0.0)) if hits else 0.0
    if not hits or top_score < settings.rerank_min_score:
        return {
            "sufficient": False,
            "signal": "evidence_insufficient",
            "source": "retrieval_low_conf",
            "reason": f"检索证据不足(top={top_score:.3f})",
            "citations": [],
        }

    # ⑥ 自评闸:生成前判断证据够不够;只有 useful=false 才落池(由 agent 写)
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    try:
        chk = await selfcheck.check_sufficient(standard, ev_texts)
    except Exception:
        # 自评上游异常不是「judged false」:fail-open 继续,避免误把异常当拒答/入池
        logger.warning("证据自评失败,按证据可用继续", exc_info=True)
        chk = {"useful": True, "reason": "self_check_unavailable"}
    if not chk["useful"]:
        return {
            "sufficient": False,
            "signal": "evidence_insufficient",
            "source": "self_check",
            "reason": chk["reason"],
            "citations": [],
        }

    # ⑦ 首尾组装 + ⑧ 编号证据
    arranged = retrieval.arrange_head_tail(hits)
    cites = citations.build_citations(arranged)
    return {
        "sufficient": True,
        "evidence": citations.build_evidence(cites),
        "citations": cites,
    }
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_tool_faq_rag.py -q
uv run pytest -q
```

Expected: 7 passed;全量绿。

- [ ] **Step 5: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/tools/business.py work/ch04/tests/test_tool_faq_rag.py work/ch04/tests/test_tool_faq_vector.py
git commit -m "feat(ch04): query_faq 升级为 RAG 管线(改写→混合→重排→两闸→编号证据)"
```

---

## Task 11: ToolRun 信号 + agent 确定性拒答 + 自评落池 + SSE citations

**Files:**
- Modify: `work/ch04/app/tools/registry.py`(加 `TIMEOUTS`)
- Modify: `work/ch04/app/tools/infra.py`(`ToolRun.citations/signal` + 提取 + 长超时)
- Modify: `work/ch04/app/core/agent.py`(拒答/落池/citations)
- Modify: `work/ch04/app/api/chat.py`(citations SSE)
- Test: `work/ch04/tests/test_tools_infra.py`(追加)、`work/ch04/tests/test_agent_citations.py`(新建)、`work/ch04/tests/test_chat_api.py`(追加)

**Interfaces:**
- Consumes: `query_faq` 返回的 `sufficient/signal/source/reason/citations`;`repository.insert_low_confidence`。
- Produces:
  - `registry.TIMEOUTS: dict[str, float] = {"query_faq": 60.0}`
  - `ToolRun(tool_call_id, name, ok, tool_message, citations: list[dict] = [], signal: str | None = None)`
  - `execute_tool_call` 从结果 dict 里提取 `citations` 并**从回灌给模型的 content 中移除**;`signal` 保留在 content 里。
  - `agent.AgentResult(..., citations: list[dict] = [])`
  - `agent._faq_payload(run) -> dict | None`、`_faq_signal(runs) -> ToolRun | None`、`_faq_citations(runs) -> list[dict]`
  - `stream_agent_turn` 事件新增 `{"type": "citations", "items": [...]}`,在工具帧之后、正文 delta 之前。
  - `/api/chat` 转发 `{"event": "citations", "items": [...]}`。
  - 命中信号 → 固定 `prompts.EVIDENCE_REFUSAL`,**不调用收敛模型**;`source="self_check"` 时写 `low_confidence_questions`。

- [ ] **Step 1: 写失败测试**

`tests/test_tools_infra.py` 追加:

```python
async def test_tool_run_extracts_citations_and_signal(monkeypatch):
    async def fake(_args):
        return {
            "sufficient": False, "signal": "evidence_insufficient",
            "source": "self_check", "reason": "不够",
            "citations": [{"n": 1, "chunk_id": 5}],
        }

    fake_tool = type("T", (), {"name": "query_faq",
                               "ainvoke": staticmethod(fake)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake_tool)
    run = await infra.execute_tool_call(
        {"name": "query_faq", "args": {"keyword": "x"}, "id": "c1"},
        conversation_id=1)
    assert run.ok is True
    assert run.signal == "evidence_insufficient"
    assert run.citations == [{"n": 1, "chunk_id": 5}]
    # citations 是给 SSE 的元数据,不再重复灌进模型上下文
    assert "citations" not in run.tool_message.content


async def test_query_faq_uses_long_tool_timeout(monkeypatch):
    async def slow(_args):
        await asyncio.sleep(0.2)
        return {"sufficient": True, "evidence": "x", "citations": []}

    fake_tool = type("T", (), {"name": "query_faq",
                               "ainvoke": staticmethod(slow)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake_tool)
    # 传 0.05 会被 TIMEOUTS["query_faq"]=60 覆盖,慢工具仍成功
    run = await infra.execute_tool_call(
        {"name": "query_faq", "args": {"keyword": "x"}, "id": "c1"},
        conversation_id=1, timeout=0.05)
    assert run.ok is True
```

`tests/test_agent_citations.py`:

```python
import json

from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.core import agent
from app.core.prompts import EVIDENCE_REFUSAL
from app.db.models import LowConfidenceQuestion
from tests.test_agent_orchestration import FakeModel


def _install_fake_faq(monkeypatch, payload):
    async def fake(_args):
        return payload

    fake_tool = type("T", (), {"name": "query_faq",
                               "ainvoke": staticmethod(fake)})()
    monkeypatch.setattr("app.tools.registry.get_tool",
                        lambda name: fake_tool if name == "query_faq" else None)


def _faq_call():
    return AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "q"}, "id": "c1"}])


class _FakeRun:
    def __init__(self, name, payload):
        self.name = name
        self.ok = True
        self.signal = payload.get("signal")
        self.citations = payload.get("citations", [])
        self.tool_call_id = "c1"
        self.tool_message = type("M", (), {
            "content": json.dumps(payload, ensure_ascii=False)})()


def test_faq_payload_and_signal_helpers():
    ok = _FakeRun("query_faq", {"sufficient": True, "citations": [{"n": 1}]})
    bad = _FakeRun("query_faq", {"sufficient": False,
                                 "signal": "evidence_insufficient",
                                 "source": "self_check", "reason": "不够"})
    assert agent._faq_payload(ok)["sufficient"] is True
    assert agent._faq_signal([ok]) is None
    assert agent._faq_signal([bad]) is bad
    assert agent._faq_citations([ok]) == [{"n": 1}]


async def test_refusal_skips_convergence_run(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq(monkeypatch, {
        "sufficient": False, "signal": "evidence_insufficient",
        "source": "self_check", "reason": "证据不够", "citations": []})
    # 只给 turn1 脚本;若 agent 仍尝试收敛,FakeModel 会 IndexError,测试即失败
    model = FakeModel([_faq_call()])
    res = await agent.run_agent_turn("u1", "Pro 型号能自动铲屎吗", None, model=model)
    assert res.answer == EVIDENCE_REFUSAL
    assert model.bind_calls == 1
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert len(rows) == 1
    assert rows[0].source == "self_check"
    assert rows[0].raw_question == "Pro 型号能自动铲屎吗"
    assert rows[0].reason == "证据不够"


async def test_refusal_skips_convergence_stream(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq(monkeypatch, {
        "sufficient": False, "signal": "evidence_insufficient",
        "source": "self_check", "reason": "证据不够", "citations": []})
    model = FakeModel([_faq_call()])
    events = [ev async for ev in agent.stream_agent_turn(
        "u1", "问题的原话", None, model=model)]
    assert [e["type"] for e in events] == ["tool", "delta", "done"]
    assert events[1]["text"] == EVIDENCE_REFUSAL
    assert model.astream_messages == []          # 收敛模型一次都没调


async def test_mechanical_gate_does_not_pool(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq(monkeypatch, {
        "sufficient": False, "signal": "evidence_insufficient",
        "source": "retrieval_low_conf", "reason": "top=0.010", "citations": []})
    model = FakeModel([_faq_call()])
    res = await agent.run_agent_turn("u1", "你们卖火星车吗", None, model=model)
    assert res.answer == EVIDENCE_REFUSAL
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert rows == []                            # 机械闸只拒答,不落池


async def test_citations_event_before_deltas(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq(monkeypatch, {
        "sufficient": True, "evidence": "[1] q: a",
        "citations": [{"n": 1, "chunk_id": 5, "section_path": "运费",
                       "question": "q", "answer": "a"}]})
    model = FakeModel([_faq_call()], stream_tokens=["答案", "。"])
    events = [ev async for ev in agent.stream_agent_turn(
        "u1", "邮费多少", None, model=model)]
    types = [e["type"] for e in events]
    assert types == ["tool", "citations", "delta", "delta", "done"]
    assert events[1]["items"][0]["chunk_id"] == 5
```

`tests/test_chat_api.py` 的 `_parse_sse` 全文替换为下面这版(只多了 citations 收集):

```python
def _parse_sse(text: str) -> dict:
    tools: list[str] = []
    deltas: list[str] = []
    errors: list[str] = []
    citations: list[list] = []
    done: int | None = None
    saw_done_marker = False
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if data == "[DONE]":
            saw_done_marker = True
            continue
        if event == "error":
            errors.append(json.loads(data)["message"])
            continue
        payload = json.loads(data)
        if payload.get("event") == "tool":
            tools.append(payload["name"])
        elif payload.get("event") == "citations":
            citations.append(payload["items"])
        elif payload.get("event") == "done":
            done = payload["conversation_id"]
        elif "delta" in payload:
            deltas.append(payload["delta"])
    return {"tools": tools, "deltas": deltas, "done": done,
            "errors": errors, "done_marker": saw_done_marker,
            "citations": citations}
```

追加测试:

```python
async def test_chat_forwards_citations_event(db_session_factory, db_clean, monkeypatch):
    async def fake(_args):
        return {"sufficient": True, "evidence": "[1] q: a",
                "citations": [{"n": 1, "chunk_id": 5, "section_path": "运费",
                               "question": "q", "answer": "a"}]}

    fake_tool = type("T", (), {"name": "query_faq",
                               "ainvoke": staticmethod(fake)})()
    monkeypatch.setattr("app.tools.registry.get_tool", lambda name: fake_tool)
    first = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "q"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["答案"])
    out = await _stream_chat(model, {"user_id": "u1", "message": "邮费多少"})
    assert out["citations"][0][0]["chunk_id"] == 5
    assert "".join(out["deltas"]) == "答案"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_tools_infra.py tests/test_agent_citations.py tests/test_chat_api.py -q
```

Expected: FAIL(`ToolRun` 无新字段;agent 不识别信号;SSE 无 citations)。

- [ ] **Step 3: 实现 registry 超时映射**

`app/tools/registry.py` 追加:

```python
# query_faq 内含改写/嵌入/混合检索/CPU 重排/自评,5s 默认超时不够
TIMEOUTS: dict[str, float] = {"query_faq": 60.0}
```

- [ ] **Step 4: 实现 infra**

`app/tools/infra.py` 的 `ToolRun` 改为:

```python
from dataclasses import dataclass, field


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage
    citations: list[dict] = field(default_factory=list)
    signal: str | None = None
```

`execute_tool_call` 成功分支改为:

```python
        try:
            eff_timeout = registry.TIMEOUTS.get(name, timeout)
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=eff_timeout)
            citations: list[dict] = []
            signal: str | None = None
            if isinstance(result, dict):
                payload = dict(result)
                citations = payload.pop("citations", []) or []
                raw_signal = payload.get("signal")
                signal = raw_signal if isinstance(raw_signal, str) else None
                result = payload
            content = json.dumps(result, ensure_ascii=False, default=str)
            return ToolRun(
                tool_call_id=tc_id,
                name=name,
                ok=True,
                tool_message=ToolMessage(content=content, tool_call_id=tc_id, name=name),
                citations=citations,
                signal=signal,
            )
```

- [ ] **Step 5: 实现 agent**

`app/core/agent.py` 顶部 import 加 `json`,并把 `prompts` import 改为:

```python
from app.core.prompts import AGENT_SYSTEM, CONVERGE_NUDGE, EVIDENCE_REFUSAL
```

`AgentResult` 加默认字段:

```python
@dataclass
class AgentResult:
    conversation_id: int
    answer: str
    tool_calls: list[dict]
    tool_runs: list[ToolRun]
    citations: list[dict] = field(default_factory=list)
```

（`field` 从 `dataclasses` import。）

新增辅助:

```python
def _faq_payload(run: ToolRun) -> dict | None:
    if getattr(run, "name", None) != "query_faq" or not getattr(run, "ok", False):
        return None
    try:
        data = json.loads(run.tool_message.content)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _faq_signal(runs: list[ToolRun]) -> ToolRun | None:
    for r in runs:
        if getattr(r, "signal", None) == "evidence_insufficient":
            return r
    return None


def _faq_citations(runs: list[ToolRun]) -> list[dict]:
    """只取第一个有 citations 的 query_faq 结果,避免多次调用编号重复。"""
    for r in runs:
        if getattr(r, "name", None) == "query_faq" and getattr(r, "citations", None):
            return r.citations
    return []


async def _record_low_confidence(conversation_id: int, message: str,
                                 run: ToolRun) -> None:
    data = _faq_payload(run) or {}
    if data.get("source") != "self_check":
        return
    await repository.insert_low_confidence(
        conversation_id, message, "self_check", data.get("reason") or None)
```

`run_agent_turn` 在 `_prepare_turn` 之后、收敛之前插入:

```python
    signal = _faq_signal(runs)
    if signal is not None:
        await _record_low_confidence(conversation_id, message, signal)
        answer = EVIDENCE_REFUSAL
        await repository.append_message(conversation_id, "assistant", content=answer)
        return AgentResult(conversation_id, answer, ai.tool_calls, runs)
```

收敛成功返回改为:

```python
    return AgentResult(conversation_id, answer, ai.tool_calls, runs,
                       citations=_faq_citations(runs))
```

`stream_agent_turn` 在推完 tool 帧之后插入:

```python
    signal = _faq_signal(runs)
    if signal is not None:
        await _record_low_confidence(conversation_id, message, signal)
        yield {"type": "delta", "text": EVIDENCE_REFUSAL}
        await repository.append_message(
            conversation_id, "assistant", content=EVIDENCE_REFUSAL)
        yield {"type": "done", "conversation_id": conversation_id}
        return

    citations = _faq_citations(runs)
    if citations:
        yield {"type": "citations", "items": citations}
```

（后续收敛流式逻辑不变。）

- [ ] **Step 6: 实现 chat SSE 转发**

`app/api/chat.py` 的事件分支加:

```python
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
```

- [ ] **Step 7: 跑测试确认通过 + 全量回归**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_tools_infra.py tests/test_agent_citations.py tests/test_chat_api.py -q
uv run pytest -q
```

Expected: 目标测试通过;全量绿。

- [ ] **Step 8: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/tools/registry.py work/ch04/app/tools/infra.py work/ch04/app/core/agent.py \
  work/ch04/app/api/chat.py work/ch04/tests/test_tools_infra.py \
  work/ch04/tests/test_agent_citations.py work/ch04/tests/test_chat_api.py
git commit -m "feat(ch04): ToolRun 信号/citations + agent 确定性拒答 + 自评落池 + SSE citations"
```

---

## Task 12: 评估集 100 题 + `build_eval_set.py` 校验器

**Files:**
- Create: `work/ch04/tests/data/eval_ch04.jsonl`
- Create: `work/ch04/scripts/build_eval_set.py`
- Modify: `work/ch04/Makefile`(`eval-check`)

**Interfaces:**
- Consumes: `data/kb/*.md`、`app.kb.documents.build_chunks`、`app.kb.sources.SOURCE_TYPES`。
- Produces: 100 行 JSONL(A/B/C/E 各 20 + D 20);`make eval-check` 退出码 0/1。

**数据任务:非 TDD。** 校验器是确定性脚本,负责把 ground truth 钉在真实文档上。

- [ ] **Step 1: 写评估集**

`tests/data/eval_ch04.jsonl` 全文(每行一题;D 的 `expect_section`/`expect_points` 为空数组):

```jsonl
{"id":"A01","bucket":"A_policy","query":"满多少钱可以包邮","expect_section":["运费怎么算","运费与包邮"],"expect_points":["满99元包邮"],"difficulty":1}
{"id":"A02","bucket":"A_policy","query":"无理由退货的运费由谁承担","expect_section":["退换货运费承担"],"expect_points":["买家承担"],"difficulty":1}
{"id":"A03","bucket":"A_policy","query":"因为质量问题退货,运费谁出","expect_section":["退换货运费承担"],"expect_points":["平台承担"],"difficulty":1}
{"id":"A04","bucket":"A_policy","query":"现货商品付款后多久发货","expect_section":["发货时效多久"],"expect_points":["48小时内发货"],"difficulty":1}
{"id":"A05","bucket":"A_policy","query":"怎么在App里申请退款","expect_section":["如何申请退款"],"expect_points":["申请退款","原路退回"],"difficulty":1}
{"id":"A06","bucket":"A_policy","query":"换货流程是什么","expect_section":["换货流程怎么走"],"expect_points":["7天内","寄回","补发新品"],"difficulty":1}
{"id":"A07","bucket":"A_policy","query":"电子发票多久能开出来","expect_section":["发票如何开具","开票时效"],"expect_points":["3个工作日内"],"difficulty":1}
{"id":"A08","bucket":"A_policy","query":"下单后怎么开发票","expect_section":["发票如何开具","可开票类型"],"expect_points":["申请开票","电子发票"],"difficulty":1}
{"id":"A09","bucket":"A_policy","query":"平台支持哪些支付方式","expect_section":["支持哪些支付方式"],"expect_points":["支付宝","微信支付"],"difficulty":1}
{"id":"A10","bucket":"A_policy","query":"会员等级按什么标准升级","expect_section":["会员等级"],"expect_points":["1000元","5000元"],"difficulty":2}
{"id":"A11","bucket":"A_policy","query":"积分怎么抵扣现金","expect_section":["积分怎么用"],"expect_points":["100积分抵1元"],"difficulty":1}
{"id":"A12","bucket":"A_policy","query":"优惠券在结算时怎么使用","expect_section":["优惠券和满减怎么用"],"expect_points":["结算页","限用一张店铺券"],"difficulty":1}
{"id":"A13","bucket":"A_policy","query":"哪些地区属于偏远地区,配送有什么限制","expect_section":["配送范围与偏远地区"],"expect_points":["新疆","西藏","内蒙古","港澳台及海外暂不支持"],"difficulty":2}
{"id":"A14","bucket":"A_policy","query":"订单还没发货可以取消吗","expect_section":["订单可以修改或取消吗"],"expect_points":["未发货前","自助取消"],"difficulty":1}
{"id":"A15","bucket":"A_policy","query":"商品缺货后多久能补货","expect_section":["商品缺货多久补货"],"expect_points":["3至7天","到货通知"],"difficulty":1}
{"id":"A16","bucket":"A_policy","query":"价格保护怎么申请,差价退到哪里","expect_section":["价格保护怎么申请"],"expect_points":["7天内","申请价保","原支付渠道"],"difficulty":2}
{"id":"A17","bucket":"A_policy","query":"无理由退货的7天从哪天开始算","expect_section":["适用范围"],"expect_points":["自签收之日起7天内"],"difficulty":2}
{"id":"A18","bucket":"A_policy","query":"收到包裹时发现运输破损怎么办","expect_section":["运输破损处理"],"expect_points":["48小时内","开箱视频"],"difficulty":1}
{"id":"A19","bucket":"A_policy","query":"哪些商品不支持无理由退货","expect_section":["不支持退货的商品"],"expect_points":["定制类","生鲜类","已拆封的猫砂"],"difficulty":1}
{"id":"A20","bucket":"A_policy","query":"电子类商品保修多久","expect_section":["质量问题与保修","保修说明"],"expect_points":["12个月"],"difficulty":1}
{"id":"B01","bucket":"B_model","query":"MH-LP100 的废砂盒多久倒一次","expect_section":["MH-LP100"],"expect_points":["5至7天"],"difficulty":1}
{"id":"B02","bucket":"B_model","query":"MH-LP50 支持App远程监控吗","expect_section":["MH-LP50"],"expect_points":["不含App联网功能"],"difficulty":1}
{"id":"B03","bucket":"B_model","query":"MH-LP200 适合体重多大的猫","expect_section":["MH-LP200"],"expect_points":["2kg至12kg"],"difficulty":1}
{"id":"B04","bucket":"B_model","query":"MH-W20 的滤芯多久换一次","expect_section":["MH-W20"],"expect_points":["每2周"],"difficulty":1}
{"id":"B05","bucket":"B_model","query":"MH-W40 的滤芯多久换一次","expect_section":["MH-W40"],"expect_points":["每3周"],"difficulty":1}
{"id":"B06","bucket":"B_model","query":"MH-W60 的水箱容量是多少","expect_section":["MH-W60"],"expect_points":["6L"],"difficulty":1}
{"id":"B07","bucket":"B_model","query":"MH-W40 有紫外杀菌功能吗","expect_section":["MH-W40"],"expect_points":["UV紫外杀菌"],"difficulty":1}
{"id":"B08","bucket":"B_model","query":"MH-HP12 的恒温温度是多少","expect_section":["MH-HP12"],"expect_points":["38℃"],"difficulty":1}
{"id":"B09","bucket":"B_model","query":"MH-HP20 有哪几档温度","expect_section":["MH-HP20"],"expect_points":["30℃","38℃","45℃"],"difficulty":2}
{"id":"B10","bucket":"B_model","query":"MH-FD10 断电后还能维持多久投喂","expect_section":["MH-FD10"],"expect_points":["约24小时"],"difficulty":1}
{"id":"B11","bucket":"B_model","query":"MH-FD30 断电后还能维持多久投喂","expect_section":["MH-FD30"],"expect_points":["约48小时"],"difficulty":1}
{"id":"B12","bucket":"B_model","query":"MH-FD30 支持视频回看吗,需要什么网络","expect_section":["MH-FD30"],"expect_points":["支持视频回看","2.4GWi-Fi"],"difficulty":2}
{"id":"B13","bucket":"B_model","query":"MH-CAM1 的分辨率是多少","expect_section":["MH-CAM1"],"expect_points":["2K分辨率"],"difficulty":1}
{"id":"B14","bucket":"B_model","query":"MH-CAM1 的水平旋转和俯仰角度是多少","expect_section":["MH-CAM1"],"expect_points":["355°水平旋转","90°俯仰"],"difficulty":2}
{"id":"B15","bucket":"B_model","query":"MH-CT30 猫爬架单层承重多少","expect_section":["MH-CT30"],"expect_points":["单层最大10kg"],"difficulty":1}
{"id":"B16","bucket":"B_model","query":"MH-NEST20 适合多重的猫","expect_section":["MH-NEST20"],"expect_points":["8kg以内"],"difficulty":1}
{"id":"B17","bucket":"B_model","query":"MH-NEST20 猫窝保修多久","expect_section":["MH-NEST20"],"expect_points":["12个月"],"difficulty":1}
{"id":"B18","bucket":"B_model","query":"MH-LP200 的猫砂容量是多少","expect_section":["MH-LP200"],"expect_points":["12L"],"difficulty":1}
{"id":"B19","bucket":"B_model","query":"MH-W60 的表面盘是什么材质","expect_section":["MH-W60"],"expect_points":["304不锈钢"],"difficulty":1}
{"id":"B20","bucket":"B_model","query":"MH-LP50 的废砂盒几天倒一次","expect_section":["MH-LP50"],"expect_points":["每4至5天"],"difficulty":2}
{"id":"C01","bucket":"C_colloquial","query":"买满多少就不用掏邮费了呀","expect_section":["运费怎么算","运费与包邮"],"expect_points":["满99元包邮"],"difficulty":1}
{"id":"C02","bucket":"C_colloquial","query":"东西买回来不想要了还能退不","expect_section":["适用范围"],"expect_points":["7天内","不影响二次销售"],"difficulty":1}
{"id":"C03","bucket":"C_colloquial","query":"钱退回去大概要多久啊","expect_section":["退款时效"],"expect_points":["原路退回"],"difficulty":2}
{"id":"C04","bucket":"C_colloquial","query":"发票咋弄啊,电子票行不行","expect_section":["发票如何开具","可开票类型"],"expect_points":["申请开票","电子发票"],"difficulty":1}
{"id":"C05","bucket":"C_colloquial","query":"东西坏了咋整","expect_section":["质量问题与保修","维修寄修流程"],"expect_points":["12个月","免费维修"],"difficulty":2}
{"id":"C06","bucket":"C_colloquial","query":"付完款啥时候给我发","expect_section":["发货时效多久"],"expect_points":["48小时内发货"],"difficulty":1}
{"id":"C07","bucket":"C_colloquial","query":"那个Pro款猫砂盆能自己铲屎不","expect_section":["MH-LP100"],"expect_points":["自动感应清理"],"difficulty":2}
{"id":"C08","bucket":"C_colloquial","query":"Plus饮水机滤芯多久换一次呀","expect_section":["MH-W40"],"expect_points":["每3周"],"difficulty":2}
{"id":"C09","bucket":"C_colloquial","query":"基础版喂食器停电了还能喂吗","expect_section":["MH-FD10"],"expect_points":["约24小时"],"difficulty":2}
{"id":"C10","bucket":"C_colloquial","query":"猫窝能扔洗衣机里洗吗","expect_section":["猫窝是否可以机洗"],"expect_points":["外套可以机洗","垫芯不可机洗"],"difficulty":1}
{"id":"C11","bucket":"C_colloquial","query":"金卡会员有啥好处","expect_section":["会员等级"],"expect_points":["95折","3次免运费"],"difficulty":2}
{"id":"C12","bucket":"C_colloquial","query":"积分能当钱花吗","expect_section":["积分怎么用"],"expect_points":["100积分抵1元"],"difficulty":1}
{"id":"C13","bucket":"C_colloquial","query":"偏远地方发货快不快","expect_section":["运费与包邮"],"expect_points":["顺延2至4天"],"difficulty":2}
{"id":"C14","bucket":"C_colloquial","query":"下单了还能改收货地址不","expect_section":["订单可以修改或取消吗"],"expect_points":["已发货订单无法修改收货信息"],"difficulty":2}
{"id":"C15","bucket":"C_colloquial","query":"价保咋申请啊","expect_section":["价格保护怎么申请"],"expect_points":["申请价保"],"difficulty":1}
{"id":"C16","bucket":"C_colloquial","query":"收到箱子破了咋办","expect_section":["运输破损处理"],"expect_points":["48小时内","开箱视频"],"difficulty":1}
{"id":"C17","bucket":"C_colloquial","query":"啥样的东西不能退","expect_section":["不支持退货的商品"],"expect_points":["定制类","生鲜类"],"difficulty":1}
{"id":"C18","bucket":"C_colloquial","query":"定金交了不想要了能退吗","expect_section":["预售与定金"],"expect_points":["不支持无理由退定金"],"difficulty":1}
{"id":"C19","bucket":"C_colloquial","query":"猫咪有点重,Pro款猫砂盆还能用吗","expect_section":["MH-LP100"],"expect_points":["1.5kg至8kg"],"difficulty":2}
{"id":"C20","bucket":"C_colloquial","query":"换货要自己出邮费吗","expect_section":["换货运费"],"expect_points":["买家承担","平台承担"],"difficulty":2}
{"id":"E01","bucket":"E_multi","query":"无理由退货的运费谁承担,人工客服几点下班","expect_sections_all":[["退换货运费承担","无理由退货"],["人工客服入口与时间"]],"difficulty":2}
{"id":"E02","bucket":"E_multi","query":"金卡会员每月免几次运费,不满99元的运费是多少","expect_sections_all":[["会员运费权益","会员等级"],["运费怎么算","运费与包邮"]],"difficulty":2}
{"id":"E03","bucket":"E_multi","query":"退款多久到账,售后首次响应要多久","expect_sections_all":[["退款时效"],["常见问题处理时限"]],"difficulty":2}
{"id":"E04","bucket":"E_multi","query":"电子发票多久开出来,开好后怎么收到","expect_sections_all":[["开票时效"],["可开票类型","发票如何开具"]],"difficulty":2}
{"id":"E05","bucket":"E_multi","query":"MH-W40 的滤芯多久换,换滤芯服务能用积分兑吗","expect_sections_all":[["MH-W40"],["积分怎么用"]],"difficulty":3}
{"id":"E06","bucket":"E_multi","query":"猫砂盆Pro和Lite的废砂盒容量分别是多少","expect_sections_all":[["MH-LP100"],["MH-LP50"]],"difficulty":2}
{"id":"E07","bucket":"E_multi","query":"自动喂食器断电后能撑多久,基础版和摄像头版分别说","expect_sections_all":[["MH-FD10"],["MH-FD30"]],"difficulty":2}
{"id":"E08","bucket":"E_multi","query":"质量问题换货谁出运费,电子类保修多久","expect_sections_all":[["换货运费","退换货运费承担"],["质量问题与保修","保修说明"]],"difficulty":2}
{"id":"E09","bucket":"E_multi","query":"订单没发货能取消吗,退货后积分会扣回吗","expect_sections_all":[["订单可以修改或取消吗"],["积分怎么攒"]],"difficulty":3}
{"id":"E10","bucket":"E_multi","query":"运输破损怎么处理,需要提供什么凭证","expect_sections_all":[["运输破损处理"],["人工客服入口与时间"]],"difficulty":2}
{"id":"E11","bucket":"E_multi","query":"预售商品什么时候发货,定金能退吗","expect_sections_all":[["预售与定金"],["发货时效多久"]],"difficulty":2}
{"id":"E12","bucket":"E_multi","query":"猫窝怎么清洗,保修多久","expect_sections_all":[["猫窝是否可以机洗"],["MH-NEST20"]],"difficulty":2}
{"id":"E13","bucket":"E_multi","query":"积分怎么攒,积分有效期多久","expect_sections_all":[["积分怎么攒"],["积分怎么用"]],"difficulty":2}
{"id":"E14","bucket":"E_multi","query":"饮水机不出水怎么自查,保修多久","expect_sections_all":[["常见故障自查"],["保修说明"]],"difficulty":2}
{"id":"E15","bucket":"E_multi","query":"摄像头离线怎么办,支持多大的存储卡","expect_sections_all":[["常见故障自查"],["MH-CAM1"]],"difficulty":2}
{"id":"E16","bucket":"E_multi","query":"喂食器卡粮怎么办,粮桶容量多大","expect_sections_all":[["常见故障自查"],["MH-FD10","MH-FD30"]],"difficulty":3}
{"id":"E17","bucket":"E_multi","query":"金卡生日礼盒怎么领,金卡每月免几次运费","expect_sections_all":[["会员等级"],["会员运费权益"]],"difficulty":2}
{"id":"E18","bucket":"E_multi","query":"偏远地区运费怎么算,配送时效会顺延吗","expect_sections_all":[["运费与包邮"],["配送范围与偏远地区"]],"difficulty":2}
{"id":"E19","bucket":"E_multi","query":"纸质发票怎么寄,开票时效多久","expect_sections_all":[["可开票类型"],["开票时效"]],"difficulty":2}
{"id":"E20","bucket":"E_multi","query":"换货可以换不同型号吗,换货运费谁出","expect_sections_all":[["换货条件"],["换货运费"]],"difficulty":2}
{"id":"D01","bucket":"D_offtopic","query":"你们支持货到付款吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D02","bucket":"D_offtopic","query":"可以用数字人民币支付吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D03","bucket":"D_offtopic","query":"有以旧换新活动吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D04","bucket":"D_offtopic","query":"有猫咪寄养服务吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D05","bucket":"D_offtopic","query":"保修可以转让给新主人吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D06","bucket":"D_offtopic","query":"有碎屏险吗","expect_section":[],"expect_points":[],"difficulty":2}
{"id":"D07","bucket":"D_offtopic","query":"支持花呗分期免息吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D08","bucket":"D_offtopic","query":"能帮忙宠物托运吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D09","bucket":"D_offtopic","query":"可以去门店自提吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D10","bucket":"D_offtopic","query":"可以指定顺丰快递发货吗","expect_section":[],"expect_points":[],"difficulty":2}
{"id":"D11","bucket":"D_offtopic","query":"能加急配送吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D12","bucket":"D_offtopic","query":"猫粮的成分表在哪里看","expect_section":[],"expect_points":[],"difficulty":2}
{"id":"D13","bucket":"D_offtopic","query":"你们卖狗粮吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D14","bucket":"D_offtopic","query":"有仓鼠笼吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D15","bucket":"D_offtopic","query":"卖鱼缸和过滤器吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D16","bucket":"D_offtopic","query":"有爬宠加热灯吗","expect_section":[],"expect_points":[],"difficulty":2}
{"id":"D17","bucket":"D_offtopic","query":"提供宠物保险吗","expect_section":[],"expect_points":[],"difficulty":2}
{"id":"D18","bucket":"D_offtopic","query":"积分可以转赠给朋友吗","expect_section":[],"expect_points":[],"difficulty":2}
{"id":"D19","bucket":"D_offtopic","query":"有实体会员卡吗","expect_section":[],"expect_points":[],"difficulty":1}
{"id":"D20","bucket":"D_offtopic","query":"能合并两个订单一起发货吗","expect_section":[],"expect_points":[],"difficulty":2}
```

- [ ] **Step 2: 写校验器**

`scripts/build_eval_set.py`:

```python
"""校验评估集:结构 + ground truth 必须能在 data/kb 的真实切块里落地。
数据任务,非 TDD。运行:make eval-check(或 PYTHONPATH=. uv run python scripts/build_eval_set.py)。
"""
import json
import pathlib
import re
import sys

from app.kb import documents
from app.kb.sources import KB_DIR, SOURCE_TYPES

ROOT = pathlib.Path(__file__).resolve().parents[1]       # work/ch04
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"
BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi", "D_offtopic"]


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def load_chunks() -> list:
    out = []
    for fname, ctype in SOURCE_TYPES.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        out.extend(documents.build_chunks(md, content_type=ctype))
    return out


def main() -> int:
    errors: list[str] = []
    rows = [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    if len(rows) != 100:
        errors.append(f"题数应为 100,实际 {len(rows)}")

    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        errors.append("id 有重复")

    counts = {b: 0 for b in BUCKETS}
    chunks = load_chunks()

    for r in rows:
        rid, bucket = r.get("id"), r.get("bucket")
        if bucket not in counts:
            errors.append(f"{rid}: 未知 bucket {bucket}")
            continue
        counts[bucket] += 1
        if not r.get("query"):
            errors.append(f"{rid}: query 为空")
        if r.get("difficulty") not in (1, 2, 3):
            errors.append(f"{rid}: difficulty 必须是 1/2/3")

        if bucket in ("A_policy", "B_model", "C_colloquial"):
            aliases = r.get("expect_section") or []
            points = r.get("expect_points") or []
            if not aliases:
                errors.append(f"{rid}: 缺 expect_section")
            if not points:
                errors.append(f"{rid}: 缺 expect_points")
            matched = [c for c in chunks
                       if any(a in (c.section_path or "") for a in aliases)]
            if not matched:
                errors.append(f"{rid}: expect_section 在真实 section_path 里没命中: {aliases}")
                continue
            blob = norm("\n".join(c.questions + c.answer for c in matched))
            for p in points:
                if norm(p) not in blob:
                    errors.append(f"{rid}: expect_point 未在目标小节原文中出现: {p!r}")
        elif bucket == "E_multi":
            groups = r.get("expect_sections_all") or []
            if len(groups) < 2:
                errors.append(f"{rid}: E 桶至少两组 expect_sections_all")
            hit_paths: list[set[str]] = []
            for group in groups:
                paths = {c.section_path for c in chunks
                         if any(a in (c.section_path or "") for a in group)}
                if not paths:
                    errors.append(f"{rid}: E 组在真实 section_path 里没命中: {group}")
                hit_paths.append(paths)
            # 至少两组落在不同的 section,证明是真正的多跳
            representatives = []
            for paths in hit_paths:
                if paths:
                    representatives.append(next(iter(paths)))
            if len(set(representatives)) < 2:
                errors.append(f"{rid}: E 题没有跨小节(命中都在同一 section)")
            if r.get("expect_section") or r.get("expect_points"):
                errors.append(f"{rid}: E 桶不应带 expect_section/expect_points")
        else:  # D_offtopic
            if r.get("expect_section") or r.get("expect_points"):
                errors.append(f"{rid}: D 桶不应带 ground truth")
            if r.get("expect_sections_all"):
                errors.append(f"{rid}: D 桶不应带 expect_sections_all")

    for b, n in counts.items():
        if n != 20:
            errors.append(f"{b}: 应为 20 题,实际 {n}")

    print(f"评估集 {len(rows)} 题;分桶 {counts};KB 切块 {len(chunks)} 块")
    if errors:
        print(f"\n❌ {len(errors)} 个错误:")
        for e in errors:
            print("  -", e)
        return 1
    print("✅ 结构、section 命中、points 原文、E 多跳、D 无 ground truth 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Makefile 加 eval-check**

`Makefile` 的 `.PHONY` 加 `eval-check`,并追加:

```makefile
eval-check:
	PYTHONPATH=. uv run python scripts/build_eval_set.py
```

- [ ] **Step 4: 跑校验**

```bash
cd /e/mewhelp/work/ch04
make eval-check
```

Expected: `✅ ... 全部通过`。若报 `expect_point 未在目标小节原文中出现`,按真实原文修正 JSONL(不要改文档)。

- [ ] **Step 5: 提交**

```bash
cd /e/mewhelp
git add work/ch04/tests/data/eval_ch04.jsonl work/ch04/scripts/build_eval_set.py work/ch04/Makefile
git commit -m "feat(ch04): 100 题五桶评估集 + 对真实 KB 切块的 ground-truth 校验器"
```

---

## Task 13: 裁判模块 + 标注样例 + `make judge-check`

**Files:**
- Modify: `work/ch04/app/core/prompts.py`(追加 `ANSWER_JUDGE_SYSTEM/PROMPT`)
- Create: `work/ch04/app/core/judge.py`
- Create: `work/ch04/tests/data/judge_samples.jsonl`
- Create: `work/ch04/scripts/check_judge.py`
- Modify: `work/ch04/Makefile`(`judge-check`)
- Test: `work/ch04/tests/test_judge.py`

**Interfaces:**
- Produces:
  - `class JudgeVerdict(BaseModel): covered: list[int]; faithful: bool; reason: str`
  - `def _chain(model=None) -> Runnable`
  - `async judge_answer(query: str, evidence: str, answer: str, points: list[str], model=None) -> dict` → `{"covered": list[int](1-based,已裁剪到 1..len(points)), "faithful": bool, "reason": str}`

**Prompt 任务:** 单元测试钉解析与裁剪;真实裁判质量用 6 条标注样例验证。

- [ ] **Step 1: 写失败单元测试**

`tests/test_judge.py`:

```python
import pytest

from app.core import judge


@pytest.mark.asyncio
async def test_judge_clamps_and_dedupes_covered(monkeypatch):
    class Fake:
        covered = [1, 3, 99, 1]
        faithful = False
        reason = "第 2 个数字是编的"

    class Chain:
        async def ainvoke(self, payload):
            assert "1. 满99元包邮" in payload["points"]
            assert payload["answer"] == "答案"
            return Fake()

    monkeypatch.setattr(judge, "_chain", lambda model=None: Chain())
    out = await judge.judge_answer(
        "q", "[1] 满 99 包邮", "答案", ["满99元包邮", "10元运费", "每3周"])
    assert out == {"covered": [1, 3], "faithful": False,
                   "reason": "第 2 个数字是编的"}


@pytest.mark.asyncio
async def test_judge_empty_points(monkeypatch):
    class Fake:
        covered = [1]
        faithful = True
        reason = ""

    class Chain:
        async def ainvoke(self, payload):
            return Fake()

    monkeypatch.setattr(judge, "_chain", lambda model=None: Chain())
    out = await judge.judge_answer("q", "e", "a", [])
    assert out["covered"] == []
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_judge.py -q
```

Expected: FAIL(模块不存在)。

- [ ] **Step 3: 写裁判 prompt**

`app/core/prompts.py` 追加:

```python
ANSWER_JUDGE_SYSTEM = """你是 RAG 答案评审员。给定用户问题、检索证据、客服答案和编号的标准要点,完成两件事:

1) covered:答案实际覆盖了哪些要点(只填要点编号,可多选)。换述、合并、同义表达都算覆盖;答案没提到的编号不要填。
2) faithful:答案中的事实性主张是否都能被证据支撑。

## 豁免(以下不算编造,faithful 应为 true)
- 只重复/换述证据内容,或把多块证据合并成一句话;
- 礼貌语、过渡句、引用编号 [n]、风险提示、引导联系人工客服;
- 合理拒答(证据不足时明确说没查到);
- 不确定语气(如「建议以页面为准」)和无害的通用常识(如「请确认收货地址」);
- 对证据的合理概括,不新增具体数字、型号、条件。

## 真编造(faithful 应为 false)
- 凭空出现证据里没有的具体数值、型号、时间、费用、折扣或赠品;
- 把 A 型号/品类的规格、条件安到 B 上(张冠李戴);
- 承诺证据未给出的到账/到货/赔偿时效或金额。

## 边界样例
例1:证据「MH-W40 每 3 周换滤芯」,答案「W40 大概三周换一次滤芯」→ faithful=true(换述)。
例2:证据同上,答案「W40 每 2 周换一次」→ faithful=false(凭空数字,与证据冲突)。
例3:证据「退款原路退回,到账时效以平台规则为准」,答案「一般 3 个工作日到账」→ faithful=false(凭空时效)。
例4:答案「暂时没查到相关信息,建议联系人工客服」→ faithful=true(合理拒答)。
例5:证据两块分别讲「无理由退货 7 天自签收算」和「价保 7 天自下单算」,答案把两者合并但起算点写对 → faithful=true。
例6:证据「退货需商品完好」,答案「退货需商品完好,建议保留包装并尽快联系客服」→ faithful=true(无害提醒,不是新事实)。

严格按证据判断,不要用你自己的知识补全。"""

ANSWER_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_JUDGE_SYSTEM),
     ("human",
      "用户问题:{query}\n\n检索证据:\n{evidence}\n\n客服答案:\n{answer}\n\n"
      "标准要点(编号从 1 开始):\n{points}")]
)
```

- [ ] **Step 4: 实现 judge**

`app/core/judge.py`:

```python
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import ANSWER_JUDGE_PROMPT


class JudgeVerdict(BaseModel):
    covered: list[int] = Field(default_factory=list,
                               description="答案覆盖的要点编号,从 1 开始")
    faithful: bool = Field(description="答案事实是否都能被证据支撑")
    reason: str = Field(default="", description="一句话判断依据")


def _chain(model=None):
    m = model or get_chat_model(temperature=0)
    return ANSWER_JUDGE_PROMPT | m.with_structured_output(
        JudgeVerdict, method="function_calling",
        extra_body={"thinking": {"type": "disabled"}},
    )


async def judge_answer(query: str, evidence: str, answer: str,
                       points: list[str], model=None) -> dict:
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
    r: JudgeVerdict = await _chain(model).ainvoke({
        "query": query, "evidence": evidence, "answer": answer,
        "points": numbered or "(无)",
    })
    valid = sorted({i for i in r.covered if 1 <= i <= len(points)})
    return {"covered": valid, "faithful": bool(r.faithful),
            "reason": r.reason or ""}
```

- [ ] **Step 5: 跑单元测试确认通过**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_judge.py -q
```

Expected: 2 passed。

- [ ] **Step 6: 标注样例 + 真实裁判校验**

`tests/data/judge_samples.jsonl`:

```jsonl
{"query":"MH-W40 滤芯多久换一次","evidence":"[1] MH-W40 建议每 3 周更换一次滤芯","answer":"W40 大概三周换一次滤芯。","points":["每3周"],"expect_faithful":true,"note":"换述"}
{"query":"退款多久到账","evidence":"[1] 退款原路退回,到账时效以平台售后规则为准。","answer":"暂时没查到具体到账时间,建议联系人工客服。","points":["原路退回"],"expect_faithful":true,"note":"合理拒答"}
{"query":"不满99元运费多少","evidence":"[1] 满 99 元包邮,未满收取 10 元运费。","answer":"满 99 元包邮,未满收取 20 元运费。","points":["满99元包邮","10元运费"],"expect_faithful":false,"note":"凭空数字"}
{"query":"MH-W40 滤芯多久换一次","evidence":"[1] MH-W40 每 3 周;[2] MH-W20 每 2 周。","answer":"W40 每 2 周换一次。","points":["每3周"],"expect_faithful":false,"note":"张冠李戴"}
{"query":"退货要注意什么","evidence":"[1] 无理由退货需商品完好、不影响二次销售。","answer":"退货需商品完好,建议保留包装并尽快联系客服。","points":["商品完好"],"expect_faithful":true,"note":"无害提醒"}
{"query":"退货和价保的7天从哪天算","evidence":"[1] 无理由退货 7 天自签收之日起算;[2] 价保 7 天自下单时间起算。","answer":"无理由退货的 7 天从签收算,价保的 7 天从下单算。","points":["自签收之日起","自下单时间起算"],"expect_faithful":true,"note":"跨证据合并"}
```

`scripts/check_judge.py`:

```python
"""用标注样例重放裁判,量真实 DeepSeek 裁判与人工标注的一致率。
运行:make judge-check(需 .env 里的聊天上游)。
"""
import asyncio
import json
import pathlib
import sys

from app.core.judge import judge_answer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests/data/judge_samples.jsonl"


async def main() -> int:
    rows = [json.loads(ln) for ln in SAMPLES.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    ok = 0
    for s in rows:
        verdict = await judge_answer(
            s["query"], s["evidence"], s["answer"], s["points"])
        hit = verdict["faithful"] == s["expect_faithful"]
        ok += hit
        print(f"{'PASS' if hit else 'FAIL'} {s['note']}: "
              f"expected={s['expect_faithful']} got={verdict['faithful']} "
              f"reason={verdict['reason']}")
    print(f"\n{ok}/{len(rows)}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 7: Makefile + 跑真实裁判校验**

`Makefile` `.PHONY` 加 `judge-check`,追加:

```makefile
judge-check:
	PYTHONPATH=. uv run python scripts/check_judge.py
```

```bash
cd /e/mewhelp/work/ch04
make judge-check
```

Expected: `6/6`。若某条判反,先补 few-shot 边界样例或收紧豁免/真编造条款,再重跑;不要在没通过时继续跑全量评估。

- [ ] **Step 8: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/core/prompts.py work/ch04/app/core/judge.py \
  work/ch04/tests/test_judge.py work/ch04/tests/data/judge_samples.jsonl \
  work/ch04/scripts/check_judge.py work/ch04/Makefile
git commit -m "feat(ch04): 忠实度/覆盖度裁判(temperature=0)+ 6 条标注样例回归"
```

---

## Task 14: 评估指标库 + 四策略评估脚本 + 报告落盘

**Files:**
- Create: `work/ch04/app/core/eval_metrics.py`
- Create: `work/ch04/scripts/eval_ch04.py`
- Modify: `work/ch04/Makefile`(`eval-rag`)
- Test: `work/ch04/tests/test_eval_metrics.py`

**Interfaces:**
- Produces:
  - `eval_metrics.norm(s) -> str`(去所有空白)
  - `eval_metrics.section_hit(section_path, aliases) -> bool`
  - `eval_metrics.recall_at_k(sample, hits, k=5) -> float`(A/B/C 0/1;E 按组部分分)
  - `eval_metrics.mrr(sample, hits) -> float`(A/B/C 首个命中名次倒数;E 每组名次倒数取平均,漏组记 0)
  - `eval_metrics.evidence_coverage(sample, hits) -> float`(A/B/C 要点机械匹配比例)
  - `eval_metrics.sample_per_bucket(rows, limit) -> list[dict]`(`limit` 按桶)
  - `scripts/eval_ch04.py` 产物:`data/ch04/reports/rag_eval.txt` + `rag_eval.json`
- 报告 JSON 结构(固定,验收脚本按此读):
  ```json
  {
    "meta": {"generated_at", "eval_set_size", "sampled", "limit", "strategies",
             "buckets", "retrieval_depth", "recall_k", "embed_model",
             "rerank_model", "judge_model", "kb_chunks"},
    "retrieval": {strategy: {bucket: {"recall_at_5": float, "mrr": float, "n": int}}},
    "evidence_coverage": {strategy: {"A_policy": float, "B_model": float,
                                     "C_colloquial": float, "overall": float}},
    "generation": {
      "completed": bool,
      "answer_coverage": {strategy: {"A_policy": float, "B_model": float,
                                     "C_colloquial": float, "overall": float}},
      "faithfulness": {"hybrid_rerank": {bucket: {"rate": float, "n": int}, "overall": {...}}},
      "refusal": {"D_offtopic": {"rate": float, "n": int,
                                 "sources": {...}, "details": [...]}},
      "faithfulness_cases": [...],
      "errors": [...]
    }
  }
  ```

- [ ] **Step 1: 写失败指标测试**

`tests/test_eval_metrics.py`:

```python
from app.core import eval_metrics as m


def _hit(section):
    return {"id": 1, "section_path": section, "question": "q", "answer": "a"}


def test_norm_and_section_hit():
    assert m.norm("满 99 元\n包邮") == "满99元包邮"
    assert m.section_hit("商品规格手册 / MH-W40", ["MH-W40"]) is True
    assert m.section_hit("商品规格手册 / MH-W20", ["MH-W40"]) is False


def test_recall_at_5_single_bucket():
    s = {"bucket": "A_policy", "expect_section": ["运费"]}
    assert m.recall_at_k(s, [_hit("x"), _hit("运费")], k=5) == 1.0
    assert m.recall_at_k(s, [_hit("x")] * 5 + [_hit("运费")], k=5) == 0.0


def test_recall_at_5_e_multi_partial():
    s = {"bucket": "E_multi", "expect_sections_all": [["A"], ["B"]]}
    hits = [_hit("A")] + [_hit("x")] * 5 + [_hit("B")]
    assert m.recall_at_k(s, hits, k=5) == 0.5


def test_mrr_single_and_multi():
    s = {"bucket": "A_policy", "expect_section": ["运费"]}
    assert m.mrr(s, [_hit("x"), _hit("运费")]) == 0.5
    e = {"bucket": "E_multi", "expect_sections_all": [["A"], ["B"]]}
    assert m.mrr(e, [_hit("A"), _hit("x"), _hit("x"), _hit("B")]) == (1 + 1 / 4) / 2
    assert m.mrr(e, [_hit("A"), _hit("x")]) == 0.5


def test_evidence_coverage_mechanical():
    s = {"expect_points": ["满99元包邮", "10元运费"]}
    hits = [{"question": "运费", "answer": "满 99 元包邮"}]
    assert m.evidence_coverage(s, hits) == 0.5


def test_sample_per_bucket_is_per_bucket():
    rows = ([{"id": f"A{i}", "bucket": "A_policy"} for i in range(3)] +
            [{"id": f"D{i}", "bucket": "D_offtopic"} for i in range(3)])
    out = m.sample_per_bucket(rows, limit=2)
    assert [r["id"] for r in out] == ["A0", "A1", "D0", "D1"]
    assert m.sample_per_bucket(rows, limit=None) == rows
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_eval_metrics.py -q
```

Expected: FAIL(模块不存在)。

- [ ] **Step 3: 实现 `eval_metrics.py`**

```python
"""评估指标:确定性、可复现,不依赖 LLM。"""
import re

GRADED_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
ANSWERABLE_BUCKETS = ("A_policy", "B_model", "C_colloquial")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def section_hit(section_path: str, aliases: list[str]) -> bool:
    return any(a in (section_path or "") for a in aliases)


def _single_rank(sample: dict, hits: list[dict]) -> int | None:
    for i, h in enumerate(hits):
        if section_hit(h.get("section_path", ""), sample.get("expect_section") or []):
            return i + 1
    return None


def _group_rank(group: list[str], hits: list[dict]) -> int | None:
    for i, h in enumerate(hits):
        if section_hit(h.get("section_path", ""), group):
            return i + 1
    return None


def recall_at_k(sample: dict, hits: list[dict], k: int = 5) -> float:
    top = hits[:k]
    if sample["bucket"] == "E_multi":
        groups = sample.get("expect_sections_all") or []
        if not groups:
            return 0.0
        hit = sum(1 for g in groups
                  if any(section_hit(h.get("section_path", ""), g) for h in top))
        return hit / len(groups)
    return 1.0 if any(section_hit(h.get("section_path", ""), sample.get("expect_section") or [])
                      for h in top) else 0.0


def mrr(sample: dict, hits: list[dict]) -> float:
    if sample["bucket"] == "E_multi":
        groups = sample.get("expect_sections_all") or []
        if not groups:
            return 0.0
        scores = []
        for g in groups:
            rank = _group_rank(g, hits)
            scores.append(1.0 / rank if rank else 0.0)
        return sum(scores) / len(scores)
    rank = _single_rank(sample, hits)
    return 1.0 / rank if rank else 0.0


def evidence_coverage(sample: dict, hits: list[dict]) -> float:
    points = sample.get("expect_points") or []
    if not points:
        return 0.0
    blob = norm("\n".join((h.get("question") or "") + (h.get("answer") or "")
                          for h in hits))
    return sum(1 for p in points if norm(p) in blob) / len(points)


def sample_per_bucket(rows: list[dict], limit: int | None) -> list[dict]:
    if not limit:
        return list(rows)
    picked, counts = [], {}
    for r in rows:
        b = r["bucket"]
        if counts.get(b, 0) < limit:
            picked.append(r)
            counts[b] = counts.get(b, 0) + 1
    return picked
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /e/mewhelp/work/ch04
uv run pytest tests/test_eval_metrics.py -q
```

Expected: 6 passed。

- [ ] **Step 5: 写评估脚本**

`scripts/eval_ch04.py`:

```python
"""四策略 RAG 评估:检索段 + 证据覆盖度(确定性)+ 生成段(DeepSeek 裁判)。

用法:
  PYTHONPATH=. uv run python scripts/eval_ch04.py                 # 全量 100 题
  PYTHONPATH=. uv run python scripts/eval_ch04.py --limit 5       # 每桶 5 题
  PYTHONPATH=. uv run python scripts/eval_ch04.py --skip-generation
产物:
  data/ch04/reports/rag_eval.txt  人读
  data/ch04/reports/rag_eval.json 机读
"""
import argparse
import asyncio
import datetime as dt
import json
import logging
import pathlib
import sys
from collections import defaultdict

from app.config import settings
from app.core import citations, eval_metrics, retrieval
from app.core.judge import judge_answer
from app.core.llm import get_chat_model
from app.core.prompts import RAG_ANSWER_PROMPT
from app.db import repository
from app.kb import milvus_client
from app.tools.business import query_faq

logger = logging.getLogger("eval_ch04")

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"
OUT_DIR = ROOT.parent.parent / "data/ch04/reports"
OUT_TXT = OUT_DIR / "rag_eval.txt"
OUT_JSON = OUT_DIR / "rag_eval.json"

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
RETRIEVAL_BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi"]
ANSWERABLE = ["A_policy", "B_model", "C_colloquial"]
RETRIEVAL_DEPTH = 10
RECALL_K = 5
RETR_CONCURRENCY = 4
GEN_CONCURRENCY = 2
GEN_TIMEOUT = 60.0
REFUSAL_TIMEOUT = 120.0

_LINES: list[str] = []
_ERRORS: list[str] = []


def _log(line: str = "") -> None:
    print(line, flush=True)
    _LINES.append(line)


def _load(limit: int | None) -> tuple[list[dict], int]:
    rows = [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    return eval_metrics.sample_per_bucket(rows, limit), len(rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _preflight() -> int:
    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    n = milvus_client.count(client)
    if n == 0:
        raise SystemExit("Milvus 集合为空:先 make kb-revectorize && make kb-vectorize")
    return n


async def _retrieve_all(samples: list[dict]) -> dict[tuple[str, str], list[dict]]:
    sem = asyncio.Semaphore(RETR_CONCURRENCY)

    async def one(sample: dict, strategy: str) -> list[dict]:
        async with sem:
            return await retrieval.search_knowledge(
                sample["query"], strategy=strategy, top_k=RETRIEVAL_DEPTH)

    tasks = {}
    for s in samples:
        if s["bucket"] == "D_offtopic":
            continue          # D 只用 query_faq 的线上拒答路径,不做四策略检索
        for st in STRATEGIES:
            tasks[(s["id"], st)] = asyncio.create_task(one(s, st))
    out: dict[tuple[str, str], list[dict]] = {}
    for key, task in tasks.items():
        try:
            out[key] = await task
        except Exception as e:
            out[key] = []
            _ERRORS.append(f"检索失败 {key}: {type(e).__name__}: {e}")
    return out


def _retrieval_reports(samples: list[dict],
                       hits: dict[tuple[str, str], list[dict]]) -> dict:
    buckets = defaultdict(lambda: defaultdict(list))
    for s in samples:
        if s["bucket"] not in RETRIEVAL_BUCKETS:
            continue
        for st in STRATEGIES:
            h = hits[(s["id"], st)]
            buckets[st][s["bucket"]].append(
                (eval_metrics.recall_at_k(s, h, RECALL_K), eval_metrics.mrr(s, h)))
    report = {}
    for st in STRATEGIES:
        report[st] = {}
        for b in RETRIEVAL_BUCKETS:
            vals = buckets[st][b]
            report[st][b] = {
                "recall_at_5": _mean([v[0] for v in vals]),
                "mrr": _mean([v[1] for v in vals]),
                "n": len(vals),
            }
    return report


def _coverage_reports(samples: list[dict],
                      hits: dict[tuple[str, str], list[dict]]) -> dict:
    report = {}
    for st in STRATEGIES:
        per_bucket = defaultdict(list)
        for s in samples:
            if s["bucket"] in ANSWERABLE:
                per_bucket[s["bucket"]].append(
                    eval_metrics.evidence_coverage(s, hits[(s["id"], st)]))
        all_vals = [v for b in ANSWERABLE for v in per_bucket[b]]
        report[st] = {b: _mean(per_bucket[b]) for b in ANSWERABLE}
        report[st]["overall"] = _mean(all_vals)
    return report


def _answer_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))


async def _generation(samples: list[dict],
                      hits: dict[tuple[str, str], list[dict]],
                      skip: bool) -> dict:
    out = {
        "completed": False,
        "answer_coverage": {},
        "faithfulness": {},
        "refusal": {},
        "faithfulness_cases": [],
        "errors": [],
    }
    if skip:
        out["errors"].append("--skip-generation:生成段未跑")
        return out

    answer_model = get_chat_model(temperature=0)
    judge_model = get_chat_model(temperature=0)
    gen_sem = asyncio.Semaphore(GEN_CONCURRENCY)
    answer_chain = RAG_ANSWER_PROMPT | answer_model

    async def gen_one(s: dict, st: str) -> dict | None:
        h = hits[(s["id"], st)]
        cites = citations.build_citations(h)
        evidence = citations.build_evidence(cites)
        async with gen_sem:
            try:
                msg = await asyncio.wait_for(
                    answer_chain.ainvoke({"query": s["query"], "evidence": evidence}),
                    timeout=GEN_TIMEOUT)
                answer = _answer_text(msg)
                verdict = await asyncio.wait_for(
                    judge_answer(s["query"], evidence, answer,
                                 s.get("expect_points") or [], model=judge_model),
                    timeout=GEN_TIMEOUT)
            except Exception as e:
                out["errors"].append(
                    f"生成/裁判失败 {s['id']}/{st}: {type(e).__name__}: {e}")
                return None
        coverage = (len(verdict["covered"]) / len(s["expect_points"])
                    if s.get("expect_points") else None)
        return {"answer": answer, "evidence": evidence, "cites": cites,
                "verdict": verdict, "coverage": coverage}

    async def d_refusal(s: dict) -> dict:
        async with gen_sem:
            try:
                payload = await asyncio.wait_for(
                    query_faq.ainvoke({"keyword": s["query"]}),
                    timeout=REFUSAL_TIMEOUT)
            except Exception as e:
                out["errors"].append(
                    f"D 拒答检查失败 {s['id']}: {type(e).__name__}: {e}")
                return {"id": s["id"], "refused": False, "source": "error",
                        "reason": str(e)}
        return {"id": s["id"], "refused": payload.get("sufficient") is False,
                "source": payload.get("source") or ("answerable" if payload.get("sufficient") else "unknown"),
                "reason": payload.get("reason") or ""}

    tasks = []
    for s in samples:
        if s["bucket"] in ANSWERABLE:
            for st in STRATEGIES:
                tasks.append(gen_one(s, st))
        elif s["bucket"] == "D_offtopic":
            tasks.append(d_refusal(s))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 聚合成四策略 × 桶的答案覆盖度
    per = defaultdict(lambda: defaultdict(list))
    faithful = defaultdict(lambda: defaultdict(list))
    cases, seen = [], set()
    idx = 0
    for s in samples:
        if s["bucket"] in ANSWERABLE:
            for st in STRATEGIES:
                r = results[idx]
                idx += 1
                if isinstance(r, Exception) or r is None:
                    continue
                if r["coverage"] is not None:
                    per[st][s["bucket"]].append(r["coverage"])
                if st == "hybrid_rerank":
                    faithful[s["bucket"]].append(1.0 if r["verdict"]["faithful"] else 0.0)
                    if not r["verdict"]["faithful"] and s["id"] not in seen:
                        seen.add(s["id"])
                        cases.append({
                            "eval_id": s["id"], "bucket": s["bucket"],
                            "query": s["query"], "reason": r["verdict"]["reason"],
                            "answer": r["answer"],
                        })
                        try:
                            await repository.upsert_faith_case(
                                s["id"], s["bucket"], s["query"], "hybrid_rerank",
                                r["answer"], r["verdict"]["reason"],
                                r["cites"], settings.chat_model)
                        except Exception as e:
                            logger.warning("faith_cases 写入失败 %s: %s", s["id"], e)
        elif s["bucket"] == "D_offtopic":
            r = results[idx]
            idx += 1
            if isinstance(r, Exception):
                continue
            out["refusal"].setdefault("details", []).append(r)

    out["answer_coverage"] = {}
    for st in STRATEGIES:
        vals = [v for b in ANSWERABLE for v in per[st][b]]
        out["answer_coverage"][st] = {b: _mean(per[st][b]) for b in ANSWERABLE}
        out["answer_coverage"][st]["overall"] = _mean(vals)

    out["faithfulness"] = {"hybrid_rerank": {}}
    all_f = [v for b in ANSWERABLE for v in faithful[b]]
    for b in ANSWERABLE:
        out["faithfulness"]["hybrid_rerank"][b] = {
            "rate": _mean(faithful[b]), "n": len(faithful[b])}
    out["faithfulness"]["hybrid_rerank"]["overall"] = {
        "rate": _mean(all_f), "n": len(all_f)}
    out["faithfulness_cases"] = cases

    details = out["refusal"].pop("details", [])
    sources = defaultdict(int)
    for d in details:
        sources[d["source"]] += 1
    out["refusal"]["D_offtopic"] = {
        "rate": _mean([1.0 if d["refused"] else 0.0 for d in details]) if details else None,
        "n": len(details),
        "sources": dict(sources),
        "details": details,
    }
    out["completed"] = bool(all_f or per)
    return out


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _write_report(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    lines = ["Ch04 RAG 评估报告", "=" * 72, ""]
    meta = report["meta"]
    lines.append(f"时间:{meta['generated_at']}")
    lines.append(f"题数:{meta['eval_set_size']}(本轮 {meta['sampled']};limit={meta['limit']})")
    lines.append(f"KB 块数:{meta['kb_chunks']};嵌入:{meta['embed_model']};"
                 f"重排:{meta['rerank_model']};裁判:{meta['judge_model']}")
    lines.append("")
    lines.append("## 检索(Recall@5 / MRR)")
    for st in STRATEGIES:
        lines.append(f"- {st}")
        for b in RETRIEVAL_BUCKETS:
            m = report["retrieval"][st][b]
            lines.append(f"    {b:14s} Recall@5={_fmt(m['recall_at_5'])} "
                         f"MRR={_fmt(m['mrr'])} n={m['n']}")
    lines.append("")
    lines.append("## 证据覆盖度(机械匹配 expect_points)")
    for st in STRATEGIES:
        m = report["evidence_coverage"][st]
        lines.append(f"- {st}: " + " ".join(
            f"{b}={_fmt(m[b])}" for b in ANSWERABLE + ("overall",)))
    lines.append("")
    lines.append("## 生成段")
    g = report["generation"]
    if g.get("completed"):
        lines.append("答案覆盖度:")
        for st in STRATEGIES:
            m = g["answer_coverage"][st]
            lines.append(f"- {st}: " + " ".join(
                f"{b}={_fmt(m[b])}" for b in ANSWERABLE + ("overall",)))
        f = g["faithfulness"]["hybrid_rerank"]
        lines.append("Faithfulness(hybrid_rerank): " + " ".join(
            f"{b}={_fmt(f[b]['rate'])}(n={f[b]['n']})"
            for b in ANSWERABLE + ("overall",)))
        r = g["refusal"]["D_offtopic"]
        lines.append(f"D 桶拒答率:{_fmt(r['rate'])} (n={r['n']}, sources={r['sources']})")
        lines.append(f"Faithfulness 个案:{len(g['faithfulness_cases'])} 条")
    else:
        lines.append("本次未完成(--skip-generation 或上游失败);检索段与证据覆盖度仍然有效")
    if g.get("errors"):
        lines.append("")
        lines.append("## 错误")
        lines.extend(f"- {e}" for e in g["errors"])
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="每个桶最多取 N 题(默认全量)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="只跑确定性检索段与证据覆盖度")
    args = parser.parse_args()

    samples, total = _load(args.limit)
    kb_chunks = _preflight()
    _log(f"评估集 {total} 题,本轮 {len(samples)} 题;KB {kb_chunks} 块")

    hits = await _retrieve_all(samples)
    retrieval_report = _retrieval_reports(samples, hits)
    coverage_report = _coverage_reports(samples, hits)
    _log("检索段与证据覆盖度完成(零 LLM)")

    try:
        generation = await _generation(samples, hits, args.skip_generation)
    except Exception as e:
        _ERRORS.append(f"生成段整体失败:{type(e).__name__}: {e}")
        generation = {"completed": False, "answer_coverage": {},
                      "faithfulness": {}, "refusal": {},
                      "faithfulness_cases": [], "errors": _ERRORS}

    report = {
        "meta": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "eval_set_size": total,
            "sampled": len(samples),
            "limit": args.limit,
            "strategies": STRATEGIES,
            "buckets": RETRIEVAL_BUCKETS + ["D_offtopic"],
            "retrieval_depth": RETRIEVAL_DEPTH,
            "recall_k": RECALL_K,
            "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model,
            "judge_model": settings.chat_model,
            "kb_chunks": kb_chunks,
        },
        "retrieval": retrieval_report,
        "evidence_coverage": coverage_report,
        "generation": generation,
    }
    if _ERRORS:
        report["generation"].setdefault("errors", []).extend(_ERRORS)
    _write_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 6: Makefile 加 eval-rag**

`Makefile` `.PHONY` 加 `eval-rag`,追加:

```makefile
eval-rag:
	PYTHONPATH=. uv run python scripts/eval_ch04.py $(ARGS)
```

- [ ] **Step 7: 语法检查**

```bash
cd /e/mewhelp/work/ch04
uv run python scripts/eval_ch04.py --help
```

Expected: 打印 `--limit` / `--skip-generation` 帮助。真实跑在 Task 15(需要先重建 KB 到新 schema)。

- [ ] **Step 8: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/core/eval_metrics.py work/ch04/scripts/eval_ch04.py \
  work/ch04/tests/test_eval_metrics.py work/ch04/Makefile
git commit -m "feat(ch04): 四策略评估脚本(检索/证据覆盖度/生成/报告落盘)"
```

---

## Task 15: 知识库重建(新 schema)+ 四策略实跑 + B 桶 BM25 实证

**Files:**
- Modify: `work/ch04/Makefile`(`kb-revectorize`)
- Modify: `work/ch04/app/config.py`(仅当评估数据要求微调 `rerank_min_score`)
- Generated: `data/ch04/reports/rag_eval.txt`、`rag_eval.json`(仓库根;从 `work/ch04` 看是 `../../data/ch04/reports/`)

**Interfaces:**
- Consumes: Task 4 新 schema、Task 12 评估集、Task 13 裁判、Task 14 评估脚本。
- Produces: 新 `knowledge` 集合(全部 MySQL chunk 重新向量化);`data/ch04/reports/` 两份产物;验收 1/2 的实测数字。

- [ ] **Step 1: Makefile 加 kb-revectorize(保留现有 82 块,不删 MySQL)**

`Makefile` `.PHONY` 加 `kb-revectorize`,追加:

```makefile
# schema 升级专用:保留 knowledge_chunks 原文,只把全部块置回 pending 并 drop 旧集合
kb-revectorize:
	docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "UPDATE knowledge_chunks SET vectorize_status='pending', vector_id=NULL;"
	PYTHONPATH=. uv run python -c "from app.kb import milvus_client as m; m.drop(m.get_client(), 'knowledge')"
	@echo "已清空 Milvus 并把全部 chunk 置回 pending。接着跑: make kb-vectorize"
```

（若想要干净的 52 块语料,也可 `make kb-reset && make kb-build && make kb-vectorize`;本计划默认保留现有 82 块,评估集只依赖 `data/kb` 的 6 份 md,两种都成立。）

- [ ] **Step 2: 重建集合并核对条数**

```bash
cd /e/mewhelp/work/ch04
# 复用 ch03 起好的容器;若没起,从 work/ch03 起(容器名固定,不要在 ch04 重复 compose up)
docker ps --filter name=mewhelp-mysql --format '{{.Names}} {{.Status}}'
docker ps --filter name=mewhelp-milvus --format '{{.Names}} {{.Status}}'
make kb-revectorize
make kb-vectorize
PYTHONPATH=. uv run python -c "from app.kb import milvus_client as m; print('Milvus:', m.count(m.get_client()))"
docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SELECT vectorize_status, COUNT(*) FROM knowledge_chunks GROUP BY vectorize_status;"
```

Expected: `Milvus:` 条数 == MySQL `done` 条数(本机现有 82 块;若重置过则是 52 块)。两者必须相等。

- [ ] **Step 3: 校验评估集 + 确定性冒烟**

```bash
cd /e/mewhelp/work/ch04
make eval-check
make eval-rag ARGS="--limit 5 --skip-generation"
```

Expected: `eval-check` 0 错;评估脚本打印检索段/证据覆盖度,并落 `data/ch04/reports/rag_eval.{txt,json}`。用下面命令确认四策略都在:

```bash
cd /e/mewhelp/work/ch04
PYTHONPATH=. uv run python -c "
import json
p='../../data/ch04/reports/rag_eval.json'
d=json.load(open(p, encoding='utf-8'))
print(d['meta'])
print('strategies:', sorted(d['retrieval']))
print('buckets:', sorted(next(iter(d['retrieval'].values()))))
assert set(d['retrieval']) == {'vector','bm25','hybrid','hybrid_rerank'}
print('OK: 四策略分桶报告结构')
"
```

- [ ] **Step 4: 小样本生成段冒烟**

```bash
cd /e/mewhelp/work/ch04
make eval-rag ARGS="--limit 5"
```

Expected: 生成段完成,打印四策略答案覆盖度、hybrid_rerank Faithfulness、D 拒答率;`generation.completed=true`。若上游失败,看 `generation.errors`,修好上游后重跑。

- [ ] **Step 5: 全量四策略实跑(验收 1)**

```bash
cd /e/mewhelp/work/ch04
make eval-rag
```

Expected: 100 题跑完;`data/ch04/reports/rag_eval.txt` 与 `.json` 落盘。验收 1 的判据:
- 四策略 × A/B/C/E 的 Recall@5/MRR 都有数字;
- 四策略 × A/B/C 的证据覆盖度都有数字;
- 生成段四策略答案覆盖度、hybrid_rerank 三桶 Faithfulness、D 桶拒答率都有数字;
- `hybrid_rerank` 总体 MRR 通常最高;BM25 在 B 桶强、在 C 口语桶偏弱(用数据说话,不硬性断言)。

全量含 CPU 重排与 LLM 调用,耗时可能 20~60 分钟;先用 `--limit` 迭代,最终报告必须来自全量。

- [ ] **Step 6: B 桶 BM25 型号命中实证(验收 2)**

```bash
cd /e/mewhelp/work/ch04
PYTHONPATH=. uv run python -c "
import asyncio
from app.core import retrieval
async def main():
    hits = await retrieval.search_knowledge('MH-W40 滤芯多久换一次', strategy='bm25', top_k=5)
    for h in hits:
        print(h['id'], h['section_path'])
    assert any('MH-W40' in h['section_path'] for h in hits), 'BM25 未命中 MH-W40'
    print('OK: BM25 型号命中')
asyncio.run(main())
"
```

Expected: 至少一条 `section_path` 含 `MH-W40`;打印 `OK: BM25 型号命中`。

- [ ] **Step 7: 查看报告与个案台账**

```bash
cd /e/mewhelp/work/ch04
cat ../../data/ch04/reports/rag_eval.txt
docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SELECT eval_id,bucket,seen_count,status,LEFT(reason,60) AS reason FROM faith_cases ORDER BY last_seen_at DESC LIMIT 10;"
```

Expected: 报告完整;若 Faithfulness 有个案,`faith_cases` 有一题一行记录(重跑同一题只增 `seen_count`,不新增行)。

- [ ] **Step 8: 视数据微调 `rerank_min_score`(可选)**

看 `generation.refusal.D_offtopic.details`:
- 若大量 D 被 `retrieval_low_conf` 拒绝且 `reason` 里的 top 分远低于 0.3,说明机械闸有效;
- 若有 D 被答出,先确认 self-check 是否 fail-open 或证据确实像答案;必要时小幅上调阈值(0.3 → 0.35/0.4),改 `app/config.py`,重跑 `--limit 5`,记录理由到 dev-notes;
- 若正常 A/B/C 被机械闸误杀,下调阈值。没有数据支撑就不改。

- [ ] **Step 9: ch03 检索回归(参考)**

```bash
cd /e/mewhelp/work/ch04
make eval-retrieval
```

Expected: 5 条样例的 Top-1 命中数不低于 ch03 的 3/5;把实际数字记到 dev-notes。

- [ ] **Step 10: 提交**

```bash
cd /e/mewhelp
git add work/ch04/Makefile work/ch04/app/config.py \
  data/ch04/reports/rag_eval.txt data/ch04/reports/rag_eval.json
git status --short
git commit -m "feat(ch04): 新 schema 重建 82 块 + 四策略全量评估报告(验收1/2)"
```

---

## Task 16: 聊天页可点引用角标 + 来源浮层(Vibe Coding)

> **Vibe Coding:不套 TDD、不派 code review。按用户描述直接改,用浏览器验收。** 本任务只做前端;后端 citations 事件已在 Task 11 交付。

**Files:**
- Modify: `work/ch04/app/static/index.html`

**目标效果:** 助手回答里的 `[n]` 渲染成可点角标;点击弹出浮层,显示该引用的 `section_path` + 来源 chunk 原文(question/answer);点外部或 Esc 关闭。数据来自 SSE `{"event":"citations","items":[...]}`。

- [ ] **Step 1: 读现有结构(已确认)**

`app/static/index.html` 的 `inlineMd()`(约 195 行)负责行内 markdown;`send()`(约 378 行)解析 SSE;`addAssistantRow()` 返回 `{bubble, tools, content}`。改动都在这三处附近。

- [ ] **Step 2: 加 CSS**

在 `<style>` 末尾(`#send:disabled` 之后)追加:

```css
  /* 可点引用角标 + 来源浮层 */
  .cite {
    color: #4c6ef5; cursor: pointer; font-size: .76em; vertical-align: super;
    padding: 0 1px; font-weight: 600;
  }
  .cite:hover { text-decoration: underline; }
  .cite-pop {
    position: fixed; z-index: 60; width: 340px; max-width: calc(100vw - 24px);
    background: #fff; border: 2px solid #212529; box-shadow: 5px 5px 0 #212529;
    padding: 12px 14px; font-size: 13px; line-height: 1.55; display: none;
  }
  .cite-pop h4 { font-size: 13px; margin: 0 0 6px; }
  .cite-pop .q { color: #495057; margin-bottom: 4px; }
  .cite-pop .a { color: #212529; white-space: pre-wrap; }
```

- [ ] **Step 3: 接收 citations 事件 + 渲染角标**

在 `inlineMd()` 的链接替换之后、加粗之前插入:

```js
  // 引用角标:[n] → 可点 sup。行内码已被抽走,不会误伤。
  s = s.replace(/\[(\d+)\]/g, function (m, n) {
    return '<sup class="cite" data-n="' + n + '">[' + n + ']</sup>';
  });
```

在 `send()` 里 `let started = false;` 附近加:

```js
  let citations = [];
  function onCitations(items) {
    citations = Array.isArray(items) ? items : [];
    ui.bubble._citations = citations;
  }
```

`onDelta()` 首行加 `citations = ui.bubble._citations || citations;`(保证重渲染拿到本消息的引用)。SSE 分支加:

```js
        } else if (obj.event === 'citations') {
          onCitations(obj.items);
        }
```

- [ ] **Step 4: 点击角标弹浮层**

在 `sendBtn.addEventListener('click', send);` 之前追加:

```js
/* ============================ 引用浮层 ============================ */
let citePop = null;

function ensureCitePop() {
  if (citePop) return citePop;
  citePop = document.createElement('div');
  citePop.className = 'cite-pop';
  document.body.appendChild(citePop);
  document.addEventListener('click', function (e) {
    if (!citePop.contains(e.target) && !e.target.classList.contains('cite')) {
      citePop.style.display = 'none';
    }
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') citePop.style.display = 'none';
  });
  return citePop;
}

messagesEl.addEventListener('click', function (e) {
  const sup = e.target.closest('.cite');
  if (!sup) return;
  const bubble = sup.closest('.bubble');
  const items = (bubble && bubble._citations) || [];
  const n = sup.getAttribute('data-n');
  const item = items.find(function (c) { return String(c.n) === String(n); });
  if (!item) return;

  const pop = ensureCitePop();
  pop.textContent = '';
  const h = document.createElement('h4');
  h.textContent = item.section_path || '来源';
  const q = document.createElement('div');
  q.className = 'q';
  q.textContent = item.question || '';
  const a = document.createElement('div');
  a.className = 'a';
  a.textContent = item.answer || '';
  pop.appendChild(h);
  pop.appendChild(q);
  pop.appendChild(a);

  const r = sup.getBoundingClientRect();
  pop.style.display = 'block';
  pop.style.left = Math.min(r.left, window.innerWidth - 360) + 'px';
  pop.style.top = Math.min(r.bottom + 8, window.innerHeight - 220) + 'px';
});
```

- [ ] **Step 5: 浏览器验收**

```bash
cd /e/mewhelp/work/ch04
make dev
```

浏览器打开 `http://localhost:8000`,问「MH-W40 的滤芯多久换一次」。Expected:
1. 回答里的 `[1]` 是蓝色上标;
2. 点击 `[1]` 弹出浮层,标题是 `商品规格手册 / 自动饮水机 Plus(型号 MH-W40)`,正文是该 chunk 的 question/answer;
3. 点浮层外部或按 Esc 关闭。

截图保存到 `dev-notes/ch04-citation-click.png`(目录已 gitignore)。

- [ ] **Step 6: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/static/index.html
git commit -m "feat(ch04): 聊天页可点引用角标 + 来源浮层(Vibe Coding)"
```

---

## Task 17: 每段回答 👍/👎 一次性锁定(Vibe Coding)

> **Vibe Coding:不套 TDD、不派 code review。按用户描述直接改,用浏览器验收。** 本章只做前端采集,不落库。

**Files:**
- Modify: `work/ch04/app/static/index.html`

**目标效果:** 每个有效助手回答左下角一组 👍/👎;点击即点亮所选、另一个淡出、显示「已反馈」,两键锁定,一次性;纯前端,不请求后端。

- [ ] **Step 1: 加 CSS**

追加:

```css
  /* 每段回答满意度反馈 */
  .feedback { display: flex; align-items: center; gap: 8px; margin-top: 8px;
    font-size: 12px; color: #868e96; }
  .fb-btn { border: 1px solid #ced4da; background: #fff; border-radius: 6px;
    padding: 2px 8px; cursor: pointer; font-size: 13px; transition: all .12s; }
  .fb-btn:hover { border-color: #4c6ef5; }
  .fb-btn.active { border-color: #4c6ef5; background: #edf2ff;
    box-shadow: 2px 2px 0 #4c6ef5; }
  .fb-btn.dim { opacity: .35; }
  .fb-btn:disabled { cursor: default; }
  .fb-note { color: #4c6ef5; }
```

- [ ] **Step 2: 加反馈条函数**

在 `addToolBadge()` 之后追加:

```js
function addFeedbackBar(bubble) {
  if (bubble.querySelector('.feedback')) return;
  const bar = document.createElement('div');
  bar.className = 'feedback';
  const up = document.createElement('button');
  up.type = 'button'; up.className = 'fb-btn'; up.textContent = '👍';
  const down = document.createElement('button');
  down.type = 'button'; down.className = 'fb-btn'; down.textContent = '👎';
  const note = document.createElement('span');
  note.className = 'fb-note';

  let locked = false;
  function choose(btn, other) {
    if (locked) return;
    locked = true;
    btn.classList.add('active');
    other.classList.add('dim');
    up.disabled = true;
    down.disabled = true;
    note.textContent = '已反馈';
  }
  up.addEventListener('click', function () { choose(up, down); });
  down.addEventListener('click', function () { choose(down, up); });

  bar.appendChild(up);
  bar.appendChild(down);
  bar.appendChild(note);
  bubble.appendChild(bar);
}
```

- [ ] **Step 3: 在有效回答结束时挂反馈条**

`send()` 里加 `let failed = false;`;所有错误分支(`!res.ok`、SSE error、catch)置 `failed = true;`。`finally` 改为:

```js
  } finally {
    content.classList.remove('cursor');
    sendBtn.disabled = false;
    inputEl.focus();
    if (started && !failed) addFeedbackBar(ui.bubble);
  }
```

- [ ] **Step 4: 浏览器验收**

```bash
cd /e/mewhelp/work/ch04
make dev
```

浏览器问一个可回答问题(如「满多少钱包邮」),等回答结束:
1. 气泡左下角出现 👍 / 👎;
2. 点 👍 → 👍 点亮(蓝框+硬阴影),👎 淡出,出现「已反馈」;
3. 再点 👎 或重复点 👍 → 无变化(一次性锁定);
4. 刷新页面后状态不保留(本章不落库)。

截图 `dev-notes/ch04-feedback.png`。

- [ ] **Step 5: 提交**

```bash
cd /e/mewhelp
git add work/ch04/app/static/index.html
git commit -m "feat(ch04): 每段回答 👍/👎 一次性反馈(Vibe Coding,不落库)"
```

---

## Task 18: README + 验收程序化脚本 + 验收测试

**Files:**
- Create: `work/ch04/scripts/verify_acceptance.py`
- Create: `work/ch04/tests/test_ch04_acceptance.py`
- Modify: `work/ch04/Makefile`(`verify-ch04`)
- Modify: `work/ch04/app/main.py`(title)
- Modify: `work/ch04/README.md`(全文替换为 ch04 版)

**Interfaces:**
- Consumes: Task 12 评估集、Task 14 报告、Task 15 真实 KB、Task 10 `query_faq` 返回契约。
- Produces: `make verify-ch04` 的 PASS/FAIL;`tests/test_ch04_acceptance.py` 的快速验收测试。

- [ ] **Step 1: 写验收脚本**

`scripts/verify_acceptance.py`:

```python
"""验收四条的程序化检查(1/2/3 服务端/4 拒答)。
浏览器可点引用、落池 UI 由 Task 19 的人工步骤确认。
运行:make verify-ch04(需 Milvus + 已建库 + .env 聊天上游)。
"""
import asyncio
import json
import pathlib
import sys

from app.core import retrieval
from app.kb import milvus_client
from app.tools.business import query_faq

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"
REPORT = ROOT.parent.parent / "data/ch04/reports/rag_eval.json"
BUCKETS = {"A_policy", "B_model", "C_colloquial", "E_multi", "D_offtopic"}
STRATEGIES = {"vector", "bm25", "hybrid", "hybrid_rerank"}


def check_eval_set() -> bool:
    rows = [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    counts = {}
    for r in rows:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    ok = len(rows) == 100 and set(counts) == BUCKETS and all(
        v == 20 for v in counts.values())
    print(f"{'PASS' if ok else 'FAIL'} 验收1a 评估集: {len(rows)} 题 {counts}")
    return ok


def check_report() -> bool:
    if not REPORT.exists():
        print(f"FAIL 验收1b 报告不存在: {REPORT};先跑 make eval-rag")
        return False
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    ok = (set(d.get("retrieval", {})) == STRATEGIES
          and set(d.get("evidence_coverage", {})) == STRATEGIES
          and all(set(d["retrieval"][st]) >= {"A_policy", "B_model",
                                             "C_colloquial", "E_multi"}
                  for st in STRATEGIES))
    print(f"{'PASS' if ok else 'FAIL'} 验收1b 四策略报告: "
          f"strategies={sorted(d.get('retrieval', {}))}")
    return ok


async def check_bm25() -> bool:
    hits = await retrieval.search_knowledge(
        "MH-W40 滤芯多久换一次", strategy="bm25", top_k=5)
    ok = any("MH-W40" in (h.get("section_path") or "") for h in hits)
    print(f"{'PASS' if ok else 'FAIL'} 验收2 BM25 型号命中: "
          f"{[h.get('section_path') for h in hits]}")
    return ok


async def check_citations() -> bool:
    out = await query_faq.ainvoke({"keyword": "MH-W40 的滤芯多久换一次"})
    if not out.get("sufficient"):
        print(f"FAIL 验收3 引用: 可回答问题被判证据不足 {out}")
        return False
    cites = out.get("citations") or []
    ok = bool(cites) and [c["n"] for c in cites] == list(range(1, len(cites) + 1)) \
        and all(c.get("chunk_id") and c.get("section_path") for c in cites)
    print(f"{'PASS' if ok else 'FAIL'} 验收3 引用映射: {len(cites)} 条,首条={cites[:1]}")
    return ok


async def check_refusal() -> bool:
    out = await query_faq.ainvoke({"keyword": "你们卖狗粮吗"})
    ok = out.get("sufficient") is False
    print(f"{'PASS' if ok else 'FAIL'} 验收4 D 桶拒答: "
          f"source={out.get('source')} reason={out.get('reason')}")
    return ok


async def main() -> int:
    try:
        milvus_client.ensure_collection(milvus_client.get_client())
        if milvus_client.count(milvus_client.get_client()) == 0:
            print("FAIL 知识库为空:先 make kb-revectorize && make kb-vectorize")
            return 1
    except Exception as e:
        print(f"FAIL Milvus 不可用: {e}")
        return 1
    results = [check_eval_set(), check_report(), await check_bm25(),
               await check_citations(), await check_refusal()]
    print(f"\n{sum(results)}/{len(results)} 项通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: 写验收测试**

`tests/test_ch04_acceptance.py`:

```python
import json
import pathlib

import pytest

from app.core import retrieval
from app.kb import milvus_client

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"
REPORT = ROOT.parent.parent / "data/ch04/reports/rag_eval.json"
BUCKETS = {"A_policy", "B_model", "C_colloquial", "E_multi", "D_offtopic"}


def test_eval_set_structure():
    rows = [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    assert len(rows) == 100
    counts = {}
    ids = set()
    for r in rows:
        assert r["id"] not in ids
        ids.add(r["id"])
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
        assert r["difficulty"] in (1, 2, 3)
        if r["bucket"] == "D_offtopic":
            assert r["expect_section"] == [] and r["expect_points"] == []
        elif r["bucket"] == "E_multi":
            assert len(r["expect_sections_all"]) >= 2
        else:
            assert r["expect_section"] and r["expect_points"]
    assert counts == {b: 20 for b in BUCKETS}


def test_report_schema_when_present():
    if not REPORT.exists():
        pytest.skip("报告还没跑;先 make eval-rag")
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    assert set(d["retrieval"]) == {"vector", "bm25", "hybrid", "hybrid_rerank"}
    assert set(d["evidence_coverage"]) == {"vector", "bm25", "hybrid", "hybrid_rerank"}
    for st in d["retrieval"]:
        assert {"A_policy", "B_model", "C_colloquial", "E_multi"} <= set(d["retrieval"][st])
    assert "meta" in d and d["meta"]["recall_k"] == 5


async def test_bm25_model_hit_on_real_kb():
    try:
        client = milvus_client.get_client()
        milvus_client.ensure_collection(client)
        if milvus_client.count(client) == 0:
            pytest.skip("知识库为空;先 make kb-vectorize")
    except Exception as e:
        pytest.skip(f"Milvus 不可用:{e}")
    hits = await retrieval.search_knowledge(
        "MH-W40 滤芯多久换一次", strategy="bm25", top_k=5)
    assert any("MH-W40" in h["section_path"] for h in hits)
```

- [ ] **Step 3: Makefile 加 verify-ch04**

`Makefile` `.PHONY` 加 `verify-ch04`,追加:

```makefile
verify-ch04:
	PYTHONPATH=. uv run python scripts/verify_acceptance.py
```

- [ ] **Step 4: 跑验收脚本与测试**

```bash
cd /e/mewhelp/work/ch04
make verify-ch04
uv run pytest tests/test_ch04_acceptance.py -q
```

Expected: `verify-ch04` 五项的 PASS(1a/1b/2/3/4);验收测试 passed(报告存在则检查 schema,KB 为空则 skip)。

- [ ] **Step 5: 更新 main.py 标题**

`app/main.py`:

```python
app = FastAPI(title="MewHelp Ch04", version="0.4.0")
```

- [ ] **Step 6: 全文更新 README**

`README.md` 替换为:

```markdown
# MewHelp — Ch04 混合检索 + 重排 + 评估

在 ch03 的知识库上把检索做上一个台阶:Milvus 原生 BM25 + dense 混合召回(RRF k=60)、
本地 `bge-reranker-v2-m3` 精排、Query 理解(口语归一 + 同义词扩展)、元数据过滤、
两道确定性证据闸(机械闸 + 自评闸)、可点引用,以及 100 题五桶四策略评估。

## 启动

    cd work/ch04
    cp .env.example .env                    # 填 CHAT_MODEL / CHAT_API_KEY(真实 key)
    docker compose up -d mysql              # 宿主 3307 → 容器 3306(复用 mewhelp-mysql)
    make milvus-up                          # etcd + minio + milvus-standalone(v2.6.22)
    make dev                                # Windows 上必须这么起,原因见 ch03 README

页面:http://localhost:8000 聊天页;`/kb` 知识库录入页;`/admin` 后台首页。
模型缓存:`HF_HOME=E:\model`、`HF_HUB_OFFLINE=1`;Makefile 已 export,直接 `uv run` 时请自行 export。

## 检索管线

    query_faq:
      ① Query 理解   口语→标准问法 + 同义词扩展(只作用于检索侧)
      ② 元数据过滤   category(Milvus filter)
      ③ 混合召回     dense Top-50 + BM25 Top-50 → hybrid_search(RRFRanker(k=60))
      ④ 重排         bge-reranker-v2-m3 精排 Top-10(本地 CPU,asyncio.to_thread)
      ⑤ 机械闸       最高 rerank 分 < rerank_min_score 或无命中 → 拒答(不落池)
      ⑥ 自评闸       useful=false → 拒答 + 落池(source=self_check)
      ⑦ 首尾组装     第 1 条放首、第 2 条放尾、其余居中
      ⑧ 编号证据 + citations(ToolRun 暂存 → SSE citations 事件)

`query_faq` 命中任一闸时返回 `signal=evidence_insufficient`;agent 检测到后**跳过收敛模型调用**,
直接输出固定拒答话术。机械闸只拒答不落池;只有自评 `useful=false` 写
`low_confidence_questions(source='self_check')`。

## 建库与 schema 迁移

    make kb-preview        # 材料清单 + 切块预览(不写库)
    make kb-build          # data/kb/*.md → MySQL knowledge_chunks(pending)
    make kb-vectorize      # pending → BGE-M3 + BM25 text → Milvus upsert
    make kb-revectorize    # ch04 schema 迁移:全部 chunk 置回 pending + drop 旧集合
    make kb-reset          # 清空 MySQL 两表 + drop 集合(慎用)

## 评估

    make eval-check                         # 评估集结构 + ground truth 对真实 KB 切块
    make eval-rag ARGS="--limit 5"          # 每桶 5 题快速跑
    make eval-rag                           # 100 题全量(四策略 + 生成段)
    make eval-rag ARGS="--skip-generation"  # 只跑确定性检索段/证据覆盖度
    make judge-check                        # 6 条标注样例重放裁判
    make verify-ch04                        # 验收四条程序化检查

产物:`data/ch04/reports/rag_eval.txt`(人读)+ `rag_eval.json`(机读)。

指标:
- 检索:A/B/C/E 桶 Recall@5、MRR(按 `expect_section` 命中 `section_path`;E 按组部分分)
- 证据覆盖度:四策略 Top-10 机械匹配 `expect_points`(零 LLM)
- 生成:四策略答案覆盖度、A/B/C Faithfulness(hybrid_rerank)、D 桶拒答率
- 个案:Faithfulness 判出的编造写入 `faith_cases`,一题一行,复发只累加 `seen_count`

## 验收四条

1. **四策略报告能跑出数字**:`make eval-rag` → `data/ch04/reports/rag_eval.{txt,json}`;
   `make verify-ch04` 的 1a/1b 通过。
2. **型号问题 BM25 能命中**:`make verify-ch04` 的 2 通过;B 桶检索指标可佐证。
3. **引用编号可定位原文**:聊天页回答带 `[n]`,点击显示 `section_path` + 原文;
   `make verify-ch04` 的 3 检查 citations 映射。截图 `dev-notes/ch04-citation-click.png`。
4. **库外问题拒答且进低置信池**:问「你们卖狗粮吗」→ 固定拒答;
   `low_confidence_questions` 出现 `source=self_check` 的记录;
   `make verify-ch04` 的 4 检查拒答。机械闸触发的拒答**不会**落池。
   若该问被机械闸先拦住,改用「猫砂盆的滤芯多久换一次」(库内无答案但能过分数闸)验证自评落池。

每段回答左下角的 👍/👎 是一次性前端反馈,本章只采集、不落库(留给后续数据飞轮章)。

## 测试

    uv run pytest -q          # 单测;Milvus 没起时真集合那几条自动 skip
    make eval-retrieval       # ch03 检索回归线(需建库)
```

- [ ] **Step 7: 提交**

```bash
cd /e/mewhelp
git add work/ch04/README.md work/ch04/app/main.py work/ch04/Makefile \
  work/ch04/scripts/verify_acceptance.py work/ch04/tests/test_ch04_acceptance.py
git commit -m "docs(ch04): README 校准 + 验收四条程序化脚本/测试"
```

---

## Task 19: 最终验收(浏览器 + 全量回归)+ dev-notes 写法

**Files:**
- Modify (local, gitignored): `dev-notes/ch04.md`
- No code changes expected;若验收发现缺陷,回到对应任务修复并补测试。

- [ ] **Step 1: 全量单测**

```bash
cd /e/mewhelp/work/ch04
# 复用已在运行的容器;没起时从 work/ch03 起
docker ps --filter name=mewhelp-mysql --format '{{.Names}} {{.Status}}'
docker ps --filter name=mewhelp-milvus --format '{{.Names}} {{.Status}}'
uv run pytest -q
```

Expected: 全绿(139 基线 + ch04 新增;Milvus 真集合测试不 skip)。任何红都先修,不要用 `-k` 跳过。

- [ ] **Step 2: 起服务做浏览器验收**

```bash
cd /e/mewhelp/work/ch04
make dev
```

另一个终端/浏览器:
1. **验收 3(引用可点)**:问「MH-W40 的滤芯多久换一次」→ 回答带 `[1]` → 点击 → 浮层显示 `商品规格手册 / 自动饮水机 Plus(型号 MH-W40)` + 原文。截图 `dev-notes/ch04-acceptance3.png`。
2. **验收 4(拒答 + 落池)**:问「你们卖狗粮吗」→ 固定拒答话术。查库:
   ```bash
   docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SELECT id,source,LEFT(raw_question,30) AS q,LEFT(reason,40) AS r FROM low_confidence_questions ORDER BY id DESC LIMIT 5;"
   ```
   Expected: 新记录 `source=self_check`。若该问先被机械闸拦住(无记录),改用「猫砂盆的滤芯多久换一次」重试;这仍是「库内没有答案」的问题,应过分数闸后由自评闸拒答并落池。截图 `dev-notes/ch04-acceptance4.png`。
3. **反馈**:任一回答点 👍 → 点亮 + 「已反馈」+ 两键锁定。截图 `dev-notes/ch04-feedback.png`(Task 17 已拍可复用)。
4. **验收 2(浏览器侧)**:问「MH-W40 滤芯多久换一次」,回答应准确;BM25 数据侧由 `make verify-ch04` 覆盖。

- [ ] **Step 3: 复跑验收脚本**

```bash
cd /e/mewhelp/work/ch04
make verify-ch04
```

Expected: `5/5 项通过`(1a/1b/2/3/4)。

- [ ] **Step 4: 写 `dev-notes/ch04.md`(本地,gitignored,不提交)**

必须包含以下小节,内容用实际命令输出/数字,不要写「正常」「已通过」这种空话:

```markdown
# Ch04 开发笔记

## 1. 冒烟实测(YYYY-MM-DD)
- Milvus 镜像/tag:
- pymilvus 版本:
- BM25 top hits(MH-LP100/MH-W40):
- hybrid_search RRF 输出:
- reranker 加载耗时:
- 50 对 CPU 打分耗时:
- HF_HOME / HF_HUB_OFFLINE:

## 2. 与 spec 的偏差与决定
- COSINE/BM25/RRF 三种 score 的语义:
- Query 改写失败回退原话;自评异常 fail-open(不落池)的原因:
- `rerank_min_score` 最终值与依据:
- 其它(如有):

## 3. 建库与检索
- MySQL chunks 数 / Milvus 条数 / 是否一致:
- `make eval-retrieval` 结果(ch03 基线 3/5):
- B 桶 BM25 命中样例:

## 4. 评估首轮结果
- 四策略 × 桶 Recall@5/MRR 表:
- 证据覆盖度:
- 生成段答案覆盖度 / Faithfulness / D 拒答率:
- `generation.errors`(如有):
- `faith_cases` 条数与复发:

## 5. 验收四条
- 1:命令 + 报告路径 + 关键数字
- 2:命令 + 输出
- 3:命令 + 截图路径 + 点击结果
- 4:命令 + SQL 输出 + source 分布
- 每段回答 👍/👎:截图路径

## 6. 已知问题 / 未做
- 指代消解、多轮改写、/rag-eval 页面、LangGraph/Langfuse 均未做(本章边界)
- 裁判自评偏袒:同模型生成/裁判,已用 temperature=0 与明确判据缓解
- 其它:

## 7. 个案与复发
- `faith_cases` 查询 SQL:
- 本轮判出的个案与处置状态:
```

注意:
- 不写任何 API key / token;引用 `.env` 只写变量名。
- 红线卡点如果有,放文件最顶部「⚠️ 红线卡点」,不要埋在中间。
- `dev-notes/` 已被 `.gitignore` 挡住,`git add` 不会带上;不要强行 `-f`。

- [ ] **Step 5: 最终检查与提交**

```bash
cd /e/mewhelp
git status --short
git log --oneline -8
```

Expected: 工作区干净(除 `.claude/` 与 gitignored 的 dev-notes/报告日志);最近提交覆盖 Task 1~18。若有本任务修复,单独提交:

```bash
git add <修复文件>
git commit -m "fix(ch04): 验收阶段修复 — <一句话>"
```

**不要 push。**

---

## Self-Review(写完计划后的自查)

### 1. Spec 覆盖

- §0 目标/不做 → Global Constraints + 各任务边界;前端两条任务标 Vibe Coding。
- §1 环境差异(本地嵌入/本地重排/Standalone/只自评落池/不做评估页/100 题/DeepSeek 裁判)→ Global Constraints + Task 1/3/5/8/10/11/12/14。
- §2.1 集合 schema(BM25 Function/chinese analyzer/双索引/标量)→ Task 4。
- §2.2 在线主链路 ①~⑧ → Task 6(理解)、7(检索/组装)、9(自评 prompt)、10(管线)、11(拒答/引用)。
- §3 组件表(milvus/rerank/retrieval/query understanding/citations)→ Task 4/5/7/6 + Task 7 citations。
- §4 生成质量控制(引用/自评/机械闸/负面知识/确定性拒答/落池口径)→ Task 9/10/11。
- §5 评估(100 题五桶、Recall@5/MRR、证据覆盖度、生成三段、四策略、裁判 temperature=0、健壮性、--limit、报告、faith_cases)→ Task 12/13/14/15。
- §6 数据模型(low_confidence_questions/faith_cases)→ Task 8。
- §7 前端(可点引用/👍👎 一次性/不落库)→ Task 16/17。
- §8 测试策略/TDD/标注验证/冒烟/验收映射 → 各 TDD 任务 + Task 1/6/13 + Task 18/19。
- §9 风险闸(本地重排/集合重建/pymilvus 3.0.2/裁判偏袒)→ Task 1/15 + dev-notes 要求。
- §10 文件布局 → 文件布局表 + 各任务 Files。

### 2. 占位符扫描

检查结果:无 `TBD` / `TODO` / `implement later` / 「类似 Task N」/「加适当错误处理」;每个代码步骤都有真实代码,每个数据步骤都有真实 JSONL/样例,每个前端步骤都有真实 JS/CSS。

### 3. 前后签名一致性

- `search_knowledge(query, strategy, top_k, category, client, collection, bm25_text)` 在 Task 7 定义,Task 10/14/18 全部按此调用。
- hit dict 字段 `id/score/question/answer/section_path/content_type/category`(+`rerank_score`)在 Task 4/7/10/14 一致。
- `query_faq` 返回 `sufficient/signal/source/reason/evidence/citations` 在 Task 10/11/18 一致。
- `ToolRun.citations/signal` 在 Task 11 的 infra/agent 测试一致。
- `insert_low_confidence(conversation_id, raw_question, source, reason)` 在 Task 8/11 一致。
- `upsert_faith_case(eval_id, bucket, query, strategy, answer, reason, citations, judge_model)` 在 Task 8/14 一致。
- `judge_answer(query, evidence, answer, points, model=None)` 在 Task 13/14 一致。
- citations item `{n, chunk_id, section_path, question, answer}` 在 Task 7/10/11/16 一致。

### 4. Review Focus 落点

1. query_faq 超时 → Task 11 `test_query_faq_uses_long_tool_timeout`。
2. 拒答不彻底/误落池 → Task 11 四个拒答/落池测试。
3. 引用编号错位 → Task 7 `test_build_citations_numbers_and_maps`、Task 10 首尾组装断言、Task 11 `test_citations_event_before_deltas`。
4. 重排语义/空输入/to_thread → Task 5 四个测试 + Task 1 真实 50 对。
5. 评估集 ground truth 漂移 → Task 12 `build_eval_set.py` + Task 18 `test_eval_set_structure` + Task 15 真实 KB 跑分。

---

## 追加:Task 20/21 RAG 评估页(终审后,用户要求)

原计划「本章不做 `/rag-eval` 页面、只出报告文件」已由用户要求覆盖,追加两条任务:

- **Task 20(后端)**:`app/api/rageval.py` 只读
  `data/ch04/reports/rag_eval.json` / `rag_eval.txt`;
  `GET /api/rag-eval/overview` 原样透传检索/覆盖度/生成段,只做四桶 MRR 算术平均;
  `GET /api/rag-eval/faith-cases` 与状态流转;`eval-rag` 加入后台作业白名单
  (`/api/jobs/eval-rag`,含 stop)。产物缺失/形状漂移返回 `present=false`,不 500。
- **Task 21(前端,Vibe Coding)**:`/rag-eval` 页面展示报告头、检索质量、证据覆盖度、
  生成段、D 桶拒答、错误与编造个案台账,支持个案状态处置和重跑/停止;后台首页
  加入口。人工验收:页面可打开、报告可读、重跑作业可取日志。

边界不变:评估页只读产物、不重算指标;验收四条与 Task 18 的
`scripts/verify_acceptance.py` 仍覆盖报告/BM25/引用/拒答落池。
