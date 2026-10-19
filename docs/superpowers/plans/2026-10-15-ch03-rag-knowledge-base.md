# Ch03 知识库向量语义检索(RAG 基础) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 的内部实现从关键词查表升级为 **进程内 BGE-M3 向量语义检索**,建起 MySQL 权威源 + **Milvus Standalone** 双写的知识库,入参出参契约不变;建库这条链落一页后台 `/kb`,录入、切块预览、补向量、检索自测都在浏览器里做。

**Architecture:** 两条离线链路(文档结构感知切分 / 历史对话挖 QA)把知识写进 MySQL `knowledge_chunks`(status=pending),一个幂等可重跑的 vectorize job 取 pending → **进程内 BGE-M3 嵌入(dense, dim 1024)** → Milvus `knowledge` 集合(按 id upsert)→ 回填 `vector_id`、status=done。在线 `query_faq` 把问题向量化后到 Milvus 取 Top-K,直接从 Milvus 取回 question/answer。录入页 `/kb` 复用同一套切块与双写:手工贴的正文走 `/api/kb/ingest`(先写 MySQL 再向量化),离线那几步走 `/api/jobs` 的 make 白名单;几块后台共用一份导航,聚合首页在 `/admin`。

**Tech Stack:** Python 3.12 · FastAPI · SQLAlchemy 2.0 async(asyncmy)· MySQL 8(Docker,端口 **3307**)· **Milvus Standalone(Docker, 19530)** · LangChain 1.x(text-splitters)· **FlagEmbedding 进程内 BGE-M3(本地 dense, dim 1024)** · pymilvus。**不接 SiliconFlow、无 `EMBED_API_KEY`**。

**Spec:** [docs/superpowers/specs/2026-10-15-ch03-rag-knowledge-base-design.md](../specs/2026-10-15-ch03-rag-knowledge-base-design.md)

> **本章代码落点**:`work/ch03/`(自 `work/ch02/` 拷贝起步)。**下列所有命令均在 `work/ch03/` 目录下执行**(uv、pytest、python 脚本、docker compose、make、git 相对路径皆以 `work/ch03/` 为当前目录;仓库根的 `sql/` 与 `data/kb/` 用 `../../sql/`、`../../data/kb/` 引用)。
> 与教程范本的三处差异(定稿):**① 嵌入走进程内 BGE-M3(非 SiliconFlow HTTP,无 key)**;**② 向量库走 Milvus Standalone(Docker @19530,非 Milvus Lite——Lite 官方不支持 Windows)**;**③ 代码落 `work/ch03/`(非仓库根 `app/`)**。

## Global Constraints

以下为项目级约束,每个任务都隐含遵守(值从 spec 逐字抄录):

- Python `>=3.12`,包管理 `uv`(`uv run` / `uv add`)。
- 技术选型定死:**FastAPI + SQLAlchemy 2.0 async + MySQL 3307 + Milvus Standalone + LangChain**。**嵌入 = 进程内加载 `FlagEmbedding` 的 BGE-M3**(本地 dense 向量,dim 1024)。**不接 SiliconFlow、无 `EMBED_API_KEY`**;在线/离线都不打嵌入 HTTP。
- **只跑 dense 向量单路**:不做关键词召回 / 混合检索 / 重排(**不引 BM25、不引 rerank**);**不引入 LangGraph / Langfuse**。实现中发现选型矛盾或走不通,**停下问用户,不自行换方案**。
- **`query_faq` 入参出参契约必须一字不改**:入参 `keyword: str`;出参命中 `{"hits": [{"question", "answer"}]}`、未命中 `{"hits": [], "message": ...}`。`faq` 表与 `repository.search_faq` 保留不删(ch02 遗留),但 `query_faq` 不再读它。
- **表结构以 `sql/ch03-ddl.sql` 为权威**,ORM 只映射不反向建表(**不用 `metadata.create_all`**)。
- **不改动 ch01/ch02 行为**:`app/api/chat.py`、`app/api/agent.py`、`app/core/agent.py` 编排、工具注册表逐字节不动;本章只改 `query_faq` 的**内部实现**与新增文件。
- **测试策略**:确定性代码走 TDD(切分 / 去重 / 双写幂等恢复 / 在线检索 / 录入 API / 作业运行器);纯 Prompt(挖知识抽取)与语义召回质量走**标注样例 eval**(`tests/data/{mining,retrieval}_samples.json`)+ 浏览器 demo。**单测里嵌入用 fake(mock `embeddings._model` / `embeddings.embed_texts`),向量库用真实 Milvus Standalone 的独立测试集合**;真实嵌入链路只在冒烟 / eval / 验收里验。
- **页面上的数只从各模块自己的读数函数来**:切块预览与录入都调 `documents.build_chunks`,向量化都调 `dualwrite.vectorize_pending`,材料清单只有 `app/kb/sources.py` 一份——页面显示的和终端跑出来的必须是同一个真相。
- **页面有重跑按钮,但没有 shell**:作业名走白名单,argv 写死在 `app/core/jobs.py`,前端只传作业名;命令一律走 make,配方是 `work/ch03/Makefile` 那一份。
- **依赖缺失是状态而不是错误**:mysql / Milvus / 嵌入任一不可用,只让相关那几个数显示「读不到」,不连坐整页,也不把「读不到」说成「对不上」。
- **HF 镜像**:首次加载 BGE-M3 需从 HuggingFace 下 ~2GB;国内慢/被墙时用环境变量 `HF_ENDPOINT=https://hf-mirror.com`(写进 `.env.example` 注释,不硬编码)。
- **uv 不在 PATH 上**:执行前先 `export PATH="/d/Anaconda3/envs/ace-step/Scripts:$PATH"`(一次性,当前 shell)。
- **`dev-notes/` 被 `.gitignore` 排除**:写它但**不 git add 它**(提交步骤里出现的 `dev-notes/...` 一律略去该路径)。
- 每个任务结束提交一次。

## File Structure

| 文件(均在 `work/ch03/` 下,除标注外) | 职责 |
|---|---|
| `docker-compose.yml` | mysql + etcd + minio + milvus-standalone(全套) |
| `.env.example`(改) | 加嵌入(BGE-M3 本地,无 key)/ Milvus / 检索阈值;HF 镜像注释;test db 名 |
| `.gitignore`(改) | 加 `log/` |
| `pyproject.toml`(改) | 加 `FlagEmbedding`、`pymilvus`、`langchain-text-splitters` |
| `Makefile`(新) | `dev / milvus-up / milvus-down / kb-* / seed-conv / eval-* / test` |
| `app/config.py`(改) | 加 `embed_model / embed_device / milvus_uri / retrieval_top_k / retrieval_min_score`(**无 `embed_api_key`**) |
| `app/core/embeddings.py` | 进程内 BGE-M3:`embed_texts` / `embed_query`(dense) |
| `app/kb/milvus_client.py` | Milvus Standalone 客户端 + 集合 ensure / upsert / search / count / drop |
| `app/db/models.py`(改) | 加 `KnowledgeChunk`、`QaExtractionStaging` ORM(只映射) |
| `app/db/repository.py`(改) | 知识 chunk / 暂存表 CRUD + 盘点方法 |
| `app/kb/chunking.py` | 切分原语:标题切 / 递归切 / 句末重叠 / 表格按行切复表头 |
| `app/kb/documents.py` | 源 `.md` → `Chunk` 记录(字段映射) |
| `app/kb/dualwrite.py` | 写 pending + vectorize(幂等可重跑) |
| `app/kb/dedup.py` | 暂存表文本归一化去重 |
| `app/kb/mining.py` | 对话 → LLM 抽 QA(结构化输出) |
| `app/kb/sources.py` | 建库材料清单(文件 → content_type),CLI 与录入页同读 |
| `app/api/kb.py` | 录入页 API:概览 / 切块预览 / 录入 / 向量化 / 检索自测 / 暂存表 |
| `app/core/jobs.py` | 作业运行器:白名单 make 目标 + 日志落盘 + 长活能停 |
| `app/api/jobs.py` | 作业 API:发起 / 状态与日志尾 / 停止 |
| `app/api/admin.py` | 后台首页聚合:知识库卡片 |
| `app/static/kb.html` | 知识库录入页(录入 / 材料 / 挖知识 / 双写 / 检索自测 / 最近入库) |
| `app/static/admin.html` | 后台管理首页(模块卡片) |
| `app/static/admin.js` | 后台各页共用导航(自带样式注入) |
| `app/main.py`(改) | 加 `/kb`、`/admin` 页面路由与静态目录挂载 + 挂 kb/jobs/admin 路由 |
| `scripts/{smoke_embed,smoke_milvus,build_kb,show_kb,vectorize_kb,mine_knowledge,eval_retrieval,eval_mining}.py` | 冒烟 / CLI / eval |
| `../../sql/ch03-ddl.sql`、`../../sql/ch03-seed.sql` | 仓库根,沿用(权威 DDL / 合成对话种子) |
| `../../data/kb/*.md` | 仓库根,沿用(6 份源文档夹具) |
| `tests/...`、`tests/data/{retrieval,mining}_samples.json` | 见 spec §12 |

## Review Focus

五个 spec 隐含、但没有任何任务测试覆盖、最可能坑到用户的输入/失败模式,各配一条测试落到拥有该代码的任务里(测试代码写在该任务对应 Step 中):

1. **空文档 / 只有标题没正文 → 切出 0 块**:`build_chunks("", ...)` 与「只有标题的行」都不得抛异常,应返回空列表;`/api/kb/ingest` 遇 0 块要 400 且不落任何库。→ **Task 9** 加 `test_empty_and_heading_only_yield_no_chunks`,**Task 17** 加 `test_ingest_empty_document_rejected`。
2. **单块正文超过 Milvus `VARCHAR(max_length)`**:`answer` > 8192 字直接 `upsert` 会被 Milvus 拒(`string length exceeds`),整批向量化中断。`upsert_vectors` 必须把 `question/answer` 裁到 schema 上限再 upsert。→ **Task 3** 加 `test_upsert_truncates_overlong_text`。
3. **Milvus 离线 → 检索/概览降级为「状态」而非崩溃**:Milvus 不可达时 `/api/kb/overview` 要返回 `milvus.online=false`、`consistent=None`(读不到 ≠ 对不上),不抛 500、不连坐材料清单与作业按钮。→ **Task 17** 加 `test_overview_milvus_offline_is_state_not_error`。
4. **重复录入同一份正文 → 幂等**:第二次 `/api/kb/ingest` 同一正文应 `inserted=0`、`skipped=总块数`,库里总数不变(指纹 = 问法 + 正文)。→ **Task 17** 加 `test_ingest_then_reingest_is_idempotent`。
5. **表格块没有数据行(只有表头 + 分隔行)→ 不崩、不丢**:`split_table_rows` 对 0 数据行表格应原样整块返回(不产空块、不抛 `IndexError`),`is_table_block` 仍认它是表格。→ **Task 8** 加 `test_table_with_no_body_rows_stays_whole`。

---

## Task 1: 冒烟风险闸 + 基础设施接线(go/no-go)

**非 TDD**(照 ch02 Task 1,真实服务冒烟由控制器/主会话亲自做)。**最大风险闸**:本地 BGE-M3 能否装上、`FlagEmbedding` 能加载、`len(embedding)==1024`;Milvus Standalone 能否起来并建集合。跑不通则**停下问用户**,只在「本地 BGE-M3」方案内换法(**不擅自换掉选型**)。

**Files:**
- Create: `work/ch03/{app,tests,scripts,pyproject.toml,.env,.env.example,README.md}`(自 `work/ch02/` 拷贝)
- Create: `work/ch03/docker-compose.yml`
- Modify: `work/ch03/.env.example`、`work/ch03/pyproject.toml`、`work/ch03/app/config.py`
- Create: `work/ch03/scripts/smoke_embed.py`、`work/ch03/scripts/smoke_milvus.py`

**Interfaces:**
- Produces: `settings.embed_model / embed_device / milvus_uri / retrieval_top_k / retrieval_min_score`(**无 `embed_api_key`**)。

- [ ] **Step 1: 拷贝 ch02 起底到 work/ch03**

Run(在 `work/ch03/` 为空目录时,先 `cd` 到 `work/`):
```bash
mkdir -p /d/mewhelp/work/ch03 && cd /d/mewhelp/work/ch03
cp -r ../ch02/app ../ch02/tests ../ch02/scripts .
cp ../ch02/pyproject.toml ../ch02/.env ../ch02/.env.example ../ch02/README.md .
rm -rf .venv .pytest_cache
```
Expected: `work/ch03/` 下出现 `app/ tests/ scripts/ pyproject.toml .env .env.example README.md`。
> `.env` 含 ch01/ch02 的聊天 key,复制后本章直接可用(仍被 gitignore,不进 git)。ch02 的 `app/db/models.py`、`repository.py`、`tools/business.py` 等作为起点保留。

- [ ] **Step 2: 加依赖**

Run: `uv add FlagEmbedding pymilvus langchain-text-splitters`
Expected: `pyproject.toml` 出现 `FlagEmbedding`(会带 `torch`)、`pymilvus`、`langchain-text-splitters`,`uv.lock` 更新。
> 若 `FlagEmbedding`/`torch` 装不进(平台/网络问题):**停下问用户**,不自行换嵌入选型。

- [ ] **Step 3: 写 docker-compose.yml(全 4 服务),并先停 ch02 compose**

Create `work/ch03/docker-compose.yml`(mysql 段照 `work/ch02/docker-compose.yml`;milvus/etcd/minio 段照仓库根 `docker-compose.yml`,卷名保持 `mewhelp-*` 以便复用数据):

```yaml
# ch03 · 全套:MySQL(3307)+ Milvus Standalone(19530)
# 注意:mysql 容器名沿用 mewhelp-mysql,与 ch02 compose 同名冲突——起本 compose 前先停 ch02 的。
services:
  mysql:
    image: mysql:8
    container_name: mewhelp-mysql
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: mewhelp
    ports:
      - "3307:3306"   # 宿主机 3307 → 容器 3306;本机 MySQL97 占着 3306
    volumes:
      - mewhelp-mysql-data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-proot"]
      interval: 3s
      timeout: 3s
      retries: 20

  etcd:
    image: quay.io/coreos/etcd:v3.5.16
    container_name: mewhelp-milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - mewhelp-milvus-etcd:/etcd
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    image: minio/minio:RELEASE.2024-05-28T17-19-04Z
    container_name: mewhelp-milvus-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - mewhelp-milvus-minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus-standalone:
    image: milvusdb/milvus:v2.6.22
    container_name: mewhelp-milvus
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - mewhelp-milvus-data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 5
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio

volumes:
  mewhelp-mysql-data:
    name: mewhelp-mysql-data     # 复用 ch02 的卷,数据不丢
  mewhelp-milvus-etcd:
  mewhelp-milvus-minio:
  mewhelp-milvus-data:
```

先停 ch02 compose(同名 `mewhelp-mysql` 冲突),再起本 compose:

Run:
```bash
docker compose -f ../ch02/docker-compose.yml stop 2>/dev/null || true
docker compose up -d mysql
docker compose up -d etcd minio milvus-standalone
echo "等待 Milvus 就绪(healthz)..."; for i in $(seq 1 60); do curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && echo "Milvus OK" && break; sleep 3; done
```
Expected: `Milvus OK`。若起不来/端口被占/内存不足:**停下问用户**。

- [ ] **Step 4: .env.example 加嵌入/Milvus/检索段**

> `log/`(作业日志落盘目录)已被**仓库根 `.gitignore`** 覆盖(第 14 行),`work/ch03/` 下无需另建 `.gitignore`。

Edit `work/ch03/.env.example`,在 MySQL 段上方追加:

```bash
# 嵌入:BGE-M3 在【进程内】本地加载,无需 key;首次加载从 HuggingFace 下 ~2GB
# 国内慢/被墙时打开镜像(取消下面一行注释再启动):
# HF_ENDPOINT=https://hf-mirror.com
EMBED_MODEL=BAAI/bge-m3
EMBED_DEVICE=cpu

# 向量库:Milvus Standalone(Docker, 19530)
MILVUS_URI=http://localhost:19530

# 检索:dense 单路 Top-K 与最低命中分(COSINE 相似度,越小越松)
RETRIEVAL_TOP_K=3
RETRIEVAL_MIN_SCORE=0.4
```

把文件里的 `TEST_DATABASE_URL` 改为本章测试库:

```bash
TEST_DATABASE_URL=mysql+asyncmy://root:root@localhost:3307/mewhelp_ch03_test
```

- [ ] **Step 5: config.py 加配置项**

Edit `work/ch03/app/config.py`,`Settings` 内在 `test_database_url` 下方加字段(**无 `embed_api_key`**):

```python
    embed_model: str = "BAAI/bge-m3"      # 本地模型名;首次加载从 HF 下 ~2GB
    embed_device: str = "cpu"             # cpu / cuda:0;非 cpu 才用 fp16 + devices
    milvus_uri: str = "http://localhost:19530"
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.4
```

> `embed_model` 写的是 HuggingFace 上的真名 `BAAI/bge-m3`,带 `BAAI/` 前缀;写成 `bge-m3` 不会启动报错,而是第一次加载时找不到模型。

- [ ] **Step 6: 写两个冒烟脚本(独立脚本,不依赖 Task 2/3 的封装)**

Create `work/ch03/scripts/smoke_embed.py`:

```python
"""冒烟:进程内加载 BGE-M3,验证能加载、出向量、维度 1024。go/no-go 风险闸。
首次会从 HuggingFace 下 ~2GB;国内慢/被墙先 export HF_ENDPOINT=https://hf-mirror.com
本脚本刻意直接调 FlagEmbedding(不 import app.core.embeddings),作为写封装之前的独立闸门。
运行:PYTHONPATH=. uv run python scripts/smoke_embed.py"""
from FlagEmbedding import BGEM3FlagModel

from app.config import settings


def main() -> None:
    # CPU 冒烟:不传 use_fp16 / devices
    model = BGEM3FlagModel(settings.embed_model)
    out = model.encode(["邮费是多少", "运费怎么算"], batch_size=12, max_length=8192)
    vecs = out["dense_vecs"]
    dims = [len(v) for v in vecs]
    print(f"返回 {len(vecs)} 条向量,维度={dims}")
    assert dims and dims[0] == 1024, f"期望维度 1024,实际 {dims}"
    print("✅ 冒烟通过:进程内 BGE-M3,维度 1024")


if __name__ == "__main__":
    main()
```

Create `work/ch03/scripts/smoke_milvus.py`:

```python
"""冒烟:连 Milvus Standalone,建集合 → upsert → search → count → drop。
独立脚本(不 import app.kb.milvus_client),验证 Standalone 这条链通不通。
需先 make milvus-up(或 docker compose up -d etcd minio milvus-standalone)。
运行:PYTHONPATH=. uv run python scripts/smoke_milvus.py"""
from pymilvus import DataType, MilvusClient

from app.config import settings

COLLECTION = "smoke_knowledge"
DIM = 1024


def main() -> None:
    client = MilvusClient(uri=settings.milvus_uri)
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("question", DataType.VARCHAR, max_length=2048)
    schema.add_field("answer", DataType.VARCHAR, max_length=8192)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)
    client.load_collection(COLLECTION)

    v = [0.0] * DIM
    v[0] = 1.0
    client.upsert(COLLECTION, [{"id": 1, "vector": v, "question": "运费怎么算", "answer": "满99包邮"}])
    client.flush(COLLECTION)
    res = client.search(COLLECTION, data=[v], anns_field="vector", limit=1,
                        output_fields=["question", "answer"],
                        search_params={"metric_type": "COSINE"})
    assert res[0] and res[0][0]["id"] == 1, f"检索异常:{res}"
    cnt = client.query(COLLECTION, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]
    assert cnt == 1, f"计数异常:{cnt}"
    client.drop_collection(COLLECTION)
    print("✅ 冒烟通过:Milvus Standalone 建集合/upsert/search/count 正常")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 跑冒烟**

Run:
```bash
PYTHONPATH=. uv run python scripts/smoke_milvus.py
PYTHONPATH=. uv run python scripts/smoke_embed.py
```
Expected: 两条各打印 `✅ …`。embed 首次会下模型,慢属正常。
**若 embed 失败**(加载不了 / 维度不符):停下,把报错 surface 给用户,只在「本地 BGE-M3」内讨论换法(如 `HF_ENDPOINT` 换镜像、`embed_device` 换 cpu),不自行换选型。
**若 Milvus 失败**(起不来 / 连不上 / healthz 不过):停下问用户,不擅自换向量库。

- [ ] **Step 8: 提交**

```bash
git add docker-compose.yml scripts/ pyproject.toml uv.lock .env.example app/ tests/ README.md
git commit -m "feat(ch03): work/ch03 起底 + Milvus Standalone compose + 进程内 BGE-M3 冒烟(维度 1024 GO)"
```
> 命令都在 `work/ch03/` 下执行,`git add` 的相对路径即以该目录为起点(`git` 按 cwd 解析路径)。

---

## Task 2: 嵌入客户端 `app/core/embeddings.py`(进程内 BGE-M3)

**TDD**(单测 mock `_model`,不加载真实模型)。

**Files:**
- Create: `work/ch03/app/core/embeddings.py`
- Test: `work/ch03/tests/test_embeddings.py`

**Interfaces:**
- Consumes: `settings.embed_model / embed_device`。
- Produces:
  - `_model() -> BGEM3FlagModel`(惰性单例;供测试 monkeypatch)
  - `async embed_texts(texts: list[str]) -> list[list[float]]`(dense, dim 1024)
  - `async embed_query(text: str) -> list[float]`

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_embeddings.py`:

```python
from app.core import embeddings


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size=None, max_length=None):
        self.calls.append((list(texts), batch_size, max_length))
        return {"dense_vecs": [[float(i), 0.0, 1.0] for i, _ in enumerate(texts)]}


async def test_embed_texts_uses_model_and_dense_vecs(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embeddings, "_model", lambda: fake)
    out = await embeddings.embed_texts(["a", "b"])
    assert out == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    assert fake.calls == [(["a", "b"], 12, 8192)]


async def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "_model", lambda: _FakeModel())
    v = await embeddings.embed_query("邮费")
    assert v == [0.0, 0.0, 1.0]
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL(`ModuleNotFoundError: app.core.embeddings`)。
> 注:work/ch03 从 ch02 拷贝,`app/core/` 下**没有** `embeddings.py`;若拷贝带来了旧文件,先删除,本任务重建为进程内版。

- [ ] **Step 3: 实现**

Create `work/ch03/app/core/embeddings.py`:

```python
"""嵌入:进程内加载 BGE-M3(dense, dim 1024),本地推理,不走网络、不需要 key。

模型只在首次调用时加载(惰性单例);CPU 推理是阻塞的,丢到线程跑,别卡住 async 事件循环。
GPU 时用 fp16 + 指定 devices;CPU 时两者都省(FlagEmbedding 会报未知参数或走错路径)。
"""
import asyncio

from FlagEmbedding import BGEM3FlagModel

from app.config import settings

_MODEL = None


def _model() -> BGEM3FlagModel:
    """惰性单例。供测试 monkeypatch(_model 返回带 .encode 的假对象)。"""
    global _MODEL
    if _MODEL is None:
        kwargs: dict = {}
        if settings.embed_device and settings.embed_device != "cpu":
            kwargs = {"use_fp16": True, "devices": [settings.embed_device]}
        _MODEL = BGEM3FlagModel(settings.embed_model, **kwargs)
    return _MODEL


def _encode(texts: list[str]) -> list[list[float]]:
    out = _model().encode(texts, batch_size=12, max_length=8192)
    return [list(map(float, v)) for v in out["dense_vecs"]]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(_encode, texts)


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/core/embeddings.py tests/test_embeddings.py
git commit -m "feat(ch03): 嵌入客户端(进程内 BGE-M3, embed_texts/embed_query, dense)"
```

---

## Task 3: Milvus 客户端 `app/kb/milvus_client.py`(Standalone)

**TDD**,单测用**真实 Milvus Standalone** 的**独立测试集合**(每次随机集合名,跑完 drop);Milvus 不可达则 skip。

**Files:**
- Create: `work/ch03/app/kb/__init__.py`(空)、`work/ch03/app/kb/milvus_client.py`
- Test: `work/ch03/tests/test_milvus_client.py`

**Interfaces:**
- Consumes: `settings.milvus_uri`。
- Produces:
  - `COLLECTION = "knowledge"`、`DIM = 1024`、`MAX_QUESTION = 2048`、`MAX_ANSWER = 8192`
  - `get_client(uri: str | None = None) -> MilvusClient`(默认 uri 走进程内单例)
  - `ensure_collection(client, collection: str = COLLECTION) -> None`
  - `upsert_vectors(client, rows: list[dict], collection: str = COLLECTION) -> None`(row 键 `id:int, vector:list[float], question:str, answer:str`;**超长字段按 schema 上限裁剪**)
  - `search(client, vector: list[float], top_k: int, collection: str = COLLECTION) -> list[dict]`(返回 `{"id","score","question","answer"}`)
  - `count(client, collection: str = COLLECTION) -> int`
  - `flush(client, collection: str = COLLECTION) -> None`
  - `drop(client, collection: str = COLLECTION) -> None`

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_milvus_client.py`:

```python
import uuid

import pytest

from app.kb import milvus_client as mc


@pytest.fixture()
def milvus():
    """真实 Milvus Standalone 上的独立测试集合;连不上就 skip(先 make milvus-up)。"""
    try:
        c = mc.get_client()
        c.list_collections()
    except Exception as e:  # 连不上是「环境没起」,不是被测逻辑错
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


def test_upsert_then_search_returns_fields(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, [
        {"id": 1, "vector": _vec(1.0), "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": _vec(0.0), "question": "发货时效", "answer": "48小时内发货"},
    ], collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.search(c, _vec(0.98), top_k=1, collection=coll)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "运费怎么算"
    assert hits[0]["answer"] == "满99包邮"
    assert isinstance(hits[0]["score"], float)


def test_upsert_is_idempotent_by_pk(milvus):
    c, coll = milvus
    row = {"id": 1, "vector": _vec(1.0), "question": "q", "answer": "a"}
    mc.upsert_vectors(c, [row], collection=coll)
    mc.upsert_vectors(c, [row], collection=coll)   # 同 id 重写
    mc.flush(c, collection=coll)
    assert mc.count(c, collection=coll) == 1


def test_upsert_truncates_overlong_text(milvus):
    """Review Focus 2:超 max_length 的 question/answer 要裁剪后再 upsert,不许整批被 Milvus 拒。"""
    c, coll = milvus
    long_answer = "满99包邮。" * 2000        # 10000 字 > MAX_ANSWER(8192)
    mc.upsert_vectors(c, [
        {"id": 7, "vector": _vec(1.0), "question": "运费怎么算", "answer": long_answer},
    ], collection=coll)
    mc.flush(c, collection=coll)
    assert mc.count(c, collection=coll) == 1
    hits = mc.search(c, _vec(1.0), top_k=1, collection=coll)
    assert len(hits[0]["answer"]) <= mc.MAX_ANSWER
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_milvus_client.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.milvus_client`)。
> 若 Milvus 没起,会 `skipped`——先 `make milvus-up` 再跑,本任务必须有真集合跑过。

- [ ] **Step 3: 实现**

Create `work/ch03/app/kb/__init__.py`(空文件)。

Create `work/ch03/app/kb/milvus_client.py`:

```python
"""Milvus Standalone(HTTP 19530)客户端:pymilvus 的 MilvusClient 高层接口。

主键 id = MySQL knowledge_chunks.id(auto_id=False,由我方提供),使重跑按 id upsert 天然幂等。
COSINE 相似度:Standalone 下 search 回的 distance 即相似度(越大越像),阈值直接比 score。
"""
from pymilvus import DataType, MilvusClient

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024
MAX_QUESTION = 2048
MAX_ANSWER = 8192

_CLIENT: MilvusClient | None = None


def get_client(uri: str | None = None) -> MilvusClient:
    """默认返回进程内单例(连 Standalone);传 uri 返回独立客户端(测试/临时)。"""
    global _CLIENT
    if uri is not None:
        return MilvusClient(uri=uri)
    if _CLIENT is None:
        _CLIENT = MilvusClient(uri=settings.milvus_uri)
    return _CLIENT


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """幂等:已存在则确保加载;不存在则按 schema 建集合 + AUTOINDEX/COSINE 索引 + 加载。"""
    if client.has_collection(collection):
        client.load_collection(collection)
        return
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("question", DataType.VARCHAR, max_length=MAX_QUESTION)
    schema.add_field("answer", DataType.VARCHAR, max_length=MAX_ANSWER)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection, schema=schema, index_params=index_params)
    client.load_collection(collection)


def _fit(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def upsert_vectors(client: MilvusClient, rows: list[dict], collection: str = COLLECTION) -> None:
    """按主键 upsert(同 id 覆盖,天然幂等)。超 max_length 的字段先裁剪,免得整批被 Milvus 拒。"""
    if not rows:
        return
    fitted = [
        {**r, "question": _fit(r["question"], MAX_QUESTION), "answer": _fit(r["answer"], MAX_ANSWER)}
        for r in rows
    ]
    client.upsert(collection, fitted)


def search(client: MilvusClient, vector: list[float], top_k: int,
           collection: str = COLLECTION) -> list[dict]:
    res = client.search(
        collection, data=[vector], anns_field="vector", limit=top_k,
        output_fields=["question", "answer"],
        search_params={"metric_type": "COSINE"},
    )
    return [
        {"id": h["id"], "score": float(h["distance"]),
         "question": h["entity"]["question"], "answer": h["entity"]["answer"]}
        for h in res[0]
    ]


def count(client: MilvusClient, collection: str = COLLECTION) -> int:
    return client.query(collection, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    """刷盘:Standalone 新 upsert 的数据须 flush 后才稳定可检索。"""
    client.flush(collection)


def drop(client: MilvusClient, collection: str = COLLECTION) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_milvus_client.py -v`
Expected: PASS(3 passed)。若 `search`/`count` 解析结构与真实 Standalone 返回不符,按真实返回结构修正(这正是本任务用真集合兜住的点)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/__init__.py app/kb/milvus_client.py tests/test_milvus_client.py
git commit -m "feat(ch03): Milvus Standalone 客户端(集合 ensure/upsert/search,id 主键 COSINE)"
```

---

## Task 4: ORM 模型 + 测试库建 ch03 表

**TDD**(DB 往返)。先扩 conftest 让测试库也建 ch03 两表,再加 ORM。

**Files:**
- Modify: `work/ch03/app/db/models.py`、`work/ch03/tests/conftest.py`
- Test: `work/ch03/tests/test_kb_models.py`

**Interfaces:**
- Produces: `KnowledgeChunk`(字段同 `sql/ch03-ddl.sql`)、`QaExtractionStaging` ORM 类。

- [ ] **Step 1: 扩 conftest 建 ch03 表(两 DDL 文件、ch03 测试库名)**

Edit `work/ch03/tests/conftest.py`,把单一 `_DDL` 改为两文件,`_TABLES` 加 ch03 两表,`_ddl_statements` 遍历两文件,建库/清表用 settings 的测试库名。替换顶部 `_DDL` / `_TABLES` / `_ddl_statements` 定义与 `_test_engine` 内的库名硬编码:

```python
_DDL_FILES = [
    pathlib.Path(__file__).resolve().parents[3] / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parents[3] / "sql" / "ch03-ddl.sql",
]
# 删除顺序:先子表后父表;knowledge_chunks 自引用 FK 靠 FOREIGN_KEY_CHECKS=0 兜
_TABLES = ["messages", "tickets", "conversations", "faq",
           "qa_extraction_staging", "knowledge_chunks"]
_DB_NAME = settings.test_database_url.rsplit("/", 1)[-1]


def _ddl_statements() -> list[str]:
    """把权威 DDL 拆成可逐句执行的语句。

    先剥掉 `--` 行注释再按 `;` 切:DDL 里注释本身含中文分号,直接 `split(";")`
    会切坏语句;且每条 CREATE TABLE 前都有一行注释,不剥注释则过滤会漏掉建表句。
    """
    stmts: list[str] = []
    for ddl in _DDL_FILES:
        lines = [ln for ln in ddl.read_text(encoding="utf-8").splitlines()
                 if not ln.lstrip().startswith("--")]
        stmts += [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    return stmts
```

并在 `_test_engine` 里把 `mewhelp_ch02_test` 换成 `_DB_NAME`:

```python
    async with admin.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {_DB_NAME}"))
        await conn.execute(text(f"CREATE DATABASE {_DB_NAME} CHARACTER SET utf8mb4"))
```

再把 `db_session_factory` 改为**依赖 `db_clean`**(每个用到测试库的用例前后都清表):

```python
@pytest_asyncio.fixture()
async def db_session_factory(_test_engine, db_clean, monkeypatch):
    """把 repository 用到的 async_session 指向测试库;跟着 db_clean 走,每个用例跑在干净的表上。"""
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory
```

> 为什么把 `db_clean` 挂进 `db_session_factory`:本章多处断言**全表计数**(如 `count_chunks_by_status("done") == 2`、`knowledge_stats()["total"]`),测试文件里的用例会共用同一个 `_test_engine`(会话级),不隔离就会互相污染。ch02 的用例各自显式带 `db_clean`;这里一次性挂到工厂上,后续任务里的用例不必逐个再写。

- [ ] **Step 2: 写失败测试**

Create `work/ch03/tests/test_kb_models.py`:

```python
from sqlalchemy import select

from app.db.models import KnowledgeChunk, QaExtractionStaging


async def test_knowledge_chunk_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = KnowledgeChunk(category="售后政策", questions="运费说明", answer="满99包邮")
        s.add(row)
        await s.commit()
        await s.refresh(row)
        assert row.id is not None
        assert row.vectorize_status == "pending"
        assert row.is_key_clause == 0


async def test_staging_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = QaExtractionStaging(batch_no="b1", question="邮费多少", answer="满99包邮")
        s.add(row)
        await s.commit()
        got = (await s.execute(select(QaExtractionStaging))).scalars().all()
        assert len(got) == 1
        assert got[0].status == "extracted"
```

- [ ] **Step 3: 运行看失败**

Run: `uv run pytest tests/test_kb_models.py -v`
Expected: FAIL(`ImportError: cannot import name 'KnowledgeChunk'`)。

- [ ] **Step 4: 实现 ORM**

Edit `work/ch03/app/db/models.py`,末尾追加(顶部 import 需含 `Integer`;`BigInteger, DateTime, Enum, String, Text, func` 已在):

```python
from sqlalchemy import Integer  # 若未导入则加


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[int] = mapped_column(Integer, server_default="0")
    prev_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done"), server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded"), server_default="extracted"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 5: 运行看通过**

Run: `uv run pytest tests/test_kb_models.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 6: 提交**

```bash
git add app/db/models.py tests/conftest.py tests/test_kb_models.py
git commit -m "feat(ch03): KnowledgeChunk/QaExtractionStaging ORM + 测试库建 ch03 表"
```

---

## Task 5: 知识 / 暂存 repository CRUD

**TDD**(DB 往返)。repository 内部用 `db.async_session`(模块属性,受 conftest monkeypatch)。

**Files:**
- Modify: `work/ch03/app/db/repository.py`
- Test: `work/ch03/tests/test_kb_repository.py`

**Interfaces:**
- Produces(全部 async):
  - `insert_knowledge_chunk(category, questions, answer, section_path=None, content_type=None, is_key_clause=0) -> int`
  - `list_pending_chunks() -> list[KnowledgeChunk]`
  - `mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None`
  - `set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None`
  - `count_chunks_by_status(status: str) -> int`
  - `list_all_questions() -> list[str]`
  - `insert_staging(batch_no, source_ref, question, answer) -> int`
  - `list_staging_by_status(status: str) -> list[QaExtractionStaging]`
  - `set_staging_status(ids: list[int], status: str) -> None`

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_kb_repository.py`:

```python
from app.db import repository


async def test_insert_and_list_pending(db_session_factory):
    cid = await repository.insert_knowledge_chunk("cat", "运费说明", "满99包邮", content_type="policy")
    pending = await repository.list_pending_chunks()
    assert [c.id for c in pending] == [cid]
    assert pending[0].questions == "运费说明"


async def test_mark_vectorized_removes_from_pending(db_session_factory):
    cid = await repository.insert_knowledge_chunk("cat", "q", "a")
    await repository.mark_chunk_vectorized(cid, str(cid))
    assert await repository.list_pending_chunks() == []
    assert await repository.count_chunks_by_status("done") == 1


async def test_set_neighbors(db_session_factory):
    a = await repository.insert_knowledge_chunk("c", "qa", "aa")
    b = await repository.insert_knowledge_chunk("c", "qb", "ab")
    await repository.set_chunk_neighbors(b, prev_id=a, next_id=None)
    pending = {c.id: c for c in await repository.list_pending_chunks()}
    assert pending[b].prev_chunk_id == a


async def test_staging_flow(db_session_factory):
    i1 = await repository.insert_staging("b1", "conv:1", "邮费多少", "满99包邮")
    await repository.insert_staging("b1", "conv:2", "怎么退货", "7天无理由")
    extracted = await repository.list_staging_by_status("extracted")
    assert len(extracted) == 2
    await repository.set_staging_status([i1], "discarded")
    assert len(await repository.list_staging_by_status("extracted")) == 1
    assert len(await repository.list_staging_by_status("discarded")) == 1


async def test_list_all_questions(db_session_factory):
    await repository.insert_knowledge_chunk("c", "运费怎么算", "满99包邮")
    assert "运费怎么算" in await repository.list_all_questions()
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_kb_repository.py -v`
Expected: FAIL(`AttributeError: module 'app.db.repository' has no attribute 'insert_knowledge_chunk'`)。

- [ ] **Step 3: 实现**

Edit `work/ch03/app/db/repository.py`:顶部 import 改为 `from sqlalchemy import func, select`,并加 `KnowledgeChunk, QaExtractionStaging` 到 `from app.db.models import ...`;末尾追加:

```python
async def insert_knowledge_chunk(
    category: str, questions: str, answer: str,
    section_path: str | None = None, content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session() as s:
        row = KnowledgeChunk(
            category=category, questions=questions, answer=answer,
            section_path=section_path, content_type=content_type,
            is_key_clause=is_key_clause,
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"
            await s.commit()


async def set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id
            await s.commit()


async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())


async def list_all_questions() -> list[str]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions))
        return list(result.scalars())


async def insert_staging(batch_no: str, source_ref: str | None, question: str, answer: str) -> int:
    async with db.async_session() as s:
        row = QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())


async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(QaExtractionStaging, i)
            if row is not None:
                row.status = status
        await s.commit()
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_kb_repository.py -v`
Expected: PASS(5 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/db/repository.py tests/test_kb_repository.py
git commit -m "feat(ch03): 知识 chunk + 暂存表 repository CRUD"
```

---

## Task 6: 切分原语 — 标题切 + 递归切

**TDD**(纯逻辑)。

**Files:**
- Create: `work/ch03/app/kb/chunking.py`
- Test: `work/ch03/tests/test_chunking_split.py`

**Interfaces:**
- Produces:
  - `HEADERS`、`CJK_SEPARATORS`
  - `split_sections(md: str) -> list[Document]`(每个 `Document.metadata` 含命中的 `h1..h4`)
  - `recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]`

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_chunking_split.py`:

```python
from app.kb import chunking


def test_split_sections_keeps_header_path():
    md = "# 售后手册\n\n## 退货政策\n\n支持7天无理由。\n\n## 运费说明\n\n满99包邮。"
    secs = chunking.split_sections(md)
    hit = [d for d in secs if d.metadata.get("h2") == "运费说明"]
    assert hit, "应切出「运费说明」小节"
    assert hit[0].metadata.get("h1") == "售后手册"
    assert "满99包邮" in hit[0].page_content


def test_recursive_split_breaks_oversized():
    text = "句子内容。" * 60  # 360 字
    parts = chunking.recursive_split(text, chunk_size=50, chunk_overlap=0)
    assert len(parts) > 1
    assert max(len(p) for p in parts) <= 50
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_chunking_split.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.chunking`)。

- [ ] **Step 3: 实现**

Create `work/ch03/app/kb/chunking.py`:

```python
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
# 中文无词边界,分隔符优先段落/换行,再句末标点,最后逐字
CJK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", " ", ""]


def split_sections(md: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    return splitter.split_text(md)


def recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=CJK_SEPARATORS, is_separator_regex=False, length_function=len,
    )
    return splitter.split_text(text)
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_chunking_split.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/chunking.py tests/test_chunking_split.py
git commit -m "feat(ch03): 切分原语 标题切 split_sections + 递归切 recursive_split"
```

---

## Task 7: 切分原语 — 句末重叠裁剪

**TDD**(纯逻辑)。重叠取上一块结尾的**完整句子**(不超过 overlap 字数),句中不截断。

**Files:**
- Modify: `work/ch03/app/kb/chunking.py`
- Test: `work/ch03/tests/test_chunking_overlap.py`

**Interfaces:**
- Produces: `apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]`(首块原样;后续块前置上一块的尾部完整句)。

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_chunking_overlap.py`:

```python
from app.kb import chunking


def test_overlap_is_whole_trailing_sentence():
    chunks = ["前面很多内容。中间一句话。最后收尾句。", "下一块正文。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=6)
    assert out[0] == chunks[0]
    # 末6字符恰为「最后收尾句。」,是完整句 → 整句作为重叠前缀
    assert out[1] == "最后收尾句。下一块正文。"


def test_overlap_never_starts_mid_sentence():
    chunks = ["这是一个非常非常非常长的句子没有中间标点结尾才有。", "新块。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=5)
    # overlap=5 放不下整句,但不能留半句 → 宁可整句(优先不留半截话)
    assert out[1].startswith("这是") or out[1] == "新块。"
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_chunking_overlap.py -v`
Expected: FAIL(`AttributeError: ... 'apply_sentence_overlap'`)。

- [ ] **Step 3: 实现**

Edit `work/ch03/app/kb/chunking.py`,加 `import re` 与:

```python
import re

_SENT_RE = re.compile(r"[^。！？!?…\n]*[。！？!?…\n]|[^。！？!?…\n]+$")


def _split_sentences(text: str) -> list[str]:
    return [m for m in _SENT_RE.findall(text) if m]


def _trailing_sentences(text: str, max_chars: int) -> str:
    """取 text 结尾若干**完整句**作为重叠,总长尽量不超过 max_chars;
    单句超长则整句保留(优先「不留半截话」)。"""
    out: list[str] = []
    total = 0
    for s in reversed(_split_sentences(text)):
        if out and total + len(s) > max_chars:
            break
        out.insert(0, s)
        total += len(s)
    return "".join(out)


def apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]:
    if not chunks:
        return []
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        ov = _trailing_sentences(chunks[i - 1], overlap)
        out.append(ov + chunks[i] if ov else chunks[i])
    return out
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_chunking_overlap.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/chunking.py tests/test_chunking_overlap.py
git commit -m "feat(ch03): 句末重叠裁剪 apply_sentence_overlap(整句重叠,不留半截话)"
```

---

## Task 8: 切分原语 — 大表格按行切 + 复制表头

**TDD**(纯逻辑)。

**Files:**
- Modify: `work/ch03/app/kb/chunking.py`
- Test: `work/ch03/tests/test_chunking_table.py`

**Interfaces:**
- Produces:
  - `is_table_block(text: str) -> bool`
  - `split_table_rows(table_md: str, max_rows: int) -> list[str]`(超 max_rows 数据行则按组切,每组重贴表头行 + 分隔行)

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_chunking_table.py`:

```python
from app.kb import chunking


TABLE = "\n".join([
    "| 商品 | 价格 |",
    "| --- | --- |",
    "| A | 1 |",
    "| B | 2 |",
    "| C | 3 |",
])


def test_is_table_block():
    assert chunking.is_table_block(TABLE)
    assert not chunking.is_table_block("普通段落文字。")


def test_split_table_replicates_header():
    out = chunking.split_table_rows(TABLE, max_rows=2)
    assert len(out) == 2
    assert out[0] == "| 商品 | 价格 |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
    assert out[1] == "| 商品 | 价格 |\n| --- | --- |\n| C | 3 |"
    for block in out:
        assert block.startswith("| 商品 | 价格 |\n| --- | --- |")  # 每块都带表头


def test_split_table_small_stays_whole():
    assert chunking.split_table_rows(TABLE, max_rows=10) == [TABLE]


def test_table_with_no_body_rows_stays_whole():
    """Review Focus 5:只有表头 + 分隔行、没有数据行的表格要原样整块返回,不崩不丢。"""
    head_only = "| 商品 | 价格 |\n| --- | --- |"
    assert chunking.is_table_block(head_only)
    assert chunking.split_table_rows(head_only, max_rows=2) == [head_only]
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_chunking_table.py -v`
Expected: FAIL(`AttributeError: ... 'is_table_block'`)。

- [ ] **Step 3: 实现**

Edit `work/ch03/app/kb/chunking.py`,追加:

```python
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def is_table_block(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return (
        len(lines) >= 2
        and lines[0].lstrip().startswith("|")
        and bool(_TABLE_SEP_RE.match(lines[1])) and "-" in lines[1]
    )


def split_table_rows(table_md: str, max_rows: int) -> list[str]:
    lines = [ln for ln in table_md.strip().splitlines() if ln.strip()]
    header, sep, rows = lines[0], lines[1], lines[2:]
    if len(rows) <= max_rows:
        return [table_md.strip()]   # 0 数据行也走这里:整块原样返回
    out: list[str] = []
    for i in range(0, len(rows), max_rows):
        out.append("\n".join([header, sep, *rows[i:i + max_rows]]))
    return out
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_chunking_table.py -v`
Expected: PASS(4 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/chunking.py tests/test_chunking_table.py
git commit -m "feat(ch03): 大表格按行切 split_table_rows(每块复制表头;0 数据行整块返回)"
```

---

## Task 9: 源文档 → Chunk 记录 `app/kb/documents.py`

**TDD**(纯逻辑)。组合 Task 6/7/8,做字段映射。

**Files:**
- Create: `work/ch03/app/kb/documents.py`
- Test: `work/ch03/tests/test_documents.py`

**Interfaces:**
- Consumes: `chunking.split_sections / recursive_split / apply_sentence_overlap / is_table_block / split_table_rows`。
- Produces:
  - `@dataclass Chunk(category, questions, answer, section_path, content_type, is_key_clause=0)`
  - `build_chunks(md: str, content_type: str, chunk_size=400, overlap=60, table_max_rows=10) -> list[Chunk]`

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_documents.py`:

```python
from app.kb.documents import Chunk, build_chunks


def test_policy_maps_heading_and_parent_path():
    md = "# 售后政策\n\n## 运费说明\n\n单笔订单满99元包邮,未满收取10元运费。偏远地区另计。"
    chunks = build_chunks(md, content_type="policy")
    c = [c for c in chunks if c.questions == "运费说明"][0]
    assert c.category == "售后政策"
    assert c.section_path == "售后政策 / 运费说明"
    assert "满99" in c.answer
    assert c.content_type == "policy"
    assert c.is_key_clause == 1  # 含「运费」关键条款


def test_table_section_splits_by_rows_with_header():
    rows = "\n".join(f"| 商品{i} | {i} |" for i in range(1, 15))
    md = f"# 商品价格表\n\n## 价目\n\n| 商品 | 价格 |\n| --- | --- |\n{rows}"
    chunks = build_chunks(md, content_type="manual", table_max_rows=5)
    price = [c for c in chunks if c.questions == "价目"]
    assert len(price) >= 3  # 14 行按 5 切成 >=3 块
    for c in price:
        assert c.answer.startswith("| 商品 | 价格 |")  # 每块复制表头


def test_returns_chunk_dataclass():
    chunks = build_chunks("# A\n\n## B\n\n正文。", content_type="faq")
    assert isinstance(chunks[0], Chunk)


def test_empty_and_heading_only_yield_no_chunks():
    """Review Focus 1:空文档 / 只有标题没有正文,都不得抛异常,返回空列表。"""
    assert build_chunks("", content_type="faq") == []
    assert build_chunks("# 只有标题\n\n## 也是标题\n", content_type="policy") == []
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_documents.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.documents`)。

- [ ] **Step 3: 实现**

Create `work/ch03/app/kb/documents.py`:

```python
from dataclasses import dataclass

from app.kb import chunking

_KEY_TERMS = ("退款", "退货", "时效", "运费", "邮费", "费用", "保修", "赔偿", "期限", "包邮")


@dataclass
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: int = 0


def _is_key(title: str, body: str) -> int:
    head = title + body[:40]
    return int(any(t in head for t in _KEY_TERMS))


def build_chunks(
    md: str, content_type: str,
    chunk_size: int = 400, overlap: int = 60, table_max_rows: int = 10,
) -> list[Chunk]:
    out: list[Chunk] = []
    for sec in chunking.split_sections(md):
        path = [sec.metadata[k] for k in ("h1", "h2", "h3", "h4") if sec.metadata.get(k)]
        section_path = " / ".join(path)
        title = path[-1] if path else content_type
        # category:政策/手册用上级标题路径;顶层无上级则用 content_type
        category = " / ".join(path[:-1]) if len(path) > 1 else (path[0] if path else content_type)
        body = sec.page_content.strip()
        if not body:
            continue                       # 只有标题没正文 → 跳过(空文档即得空列表)
        if chunking.is_table_block(body):
            pieces = chunking.split_table_rows(body, table_max_rows)
        else:
            base = chunking.recursive_split(body, chunk_size, chunk_overlap=0)
            pieces = chunking.apply_sentence_overlap(base, overlap)
        for piece in pieces:
            out.append(Chunk(
                category=category, questions=title, answer=piece,
                section_path=section_path, content_type=content_type,
                is_key_clause=_is_key(title, piece),
            ))
    return out
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_documents.py -v`
Expected: PASS(4 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/documents.py tests/test_documents.py
git commit -m "feat(ch03): 源文档→Chunk 记录 build_chunks(标题/表格分派 + 字段映射)"
```

---

## Task 10: 双写 `app/kb/dualwrite.py` + vectorize CLI(幂等恢复 = 验收2)

**TDD**(真实 test MySQL + 真实 Milvus Standalone 测试集合;嵌入 mock)。**验收2 的核心任务**。

**Files:**
- Create: `work/ch03/app/kb/dualwrite.py`、`work/ch03/scripts/vectorize_kb.py`
- Test: `work/ch03/tests/test_dualwrite.py`

**Interfaces:**
- Consumes: `repository.*`(Task 5)、`milvus_client.*`(Task 3)、`embeddings.embed_texts`(Task 2)、`documents.Chunk`(Task 9)。
- Produces:
  - `async write_pending(chunks: list[Chunk]) -> list[int]`(顺序插入 + 同文档内连 prev/next)
  - `async vectorize_pending(client=None, batch_size=64, collection=COLLECTION) -> int`(取 pending → embed → Milvus upsert → flush → 标 done;幂等可重跑)

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_dualwrite.py`:

```python
import uuid

import pytest

from app.db import repository
from app.kb import dualwrite
from app.kb import milvus_client as mc
from app.kb.documents import Chunk


def _chunk(q, a):
    return Chunk(category="c", questions=q, answer=a, section_path="c / " + q, content_type="policy")


@pytest.fixture()
def milvus():
    try:
        c = mc.get_client()
        c.list_collections()
    except Exception as e:
        pytest.skip(f"Milvus 不可用,先 make milvus-up:{e}")
    coll = f"knowledge_test_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(c, collection=coll)
    yield c, coll
    mc.drop(c, collection=coll)


async def test_write_pending_links_neighbors(db_session_factory):
    ids = await dualwrite.write_pending([_chunk("q1", "a1"), _chunk("q2", "a2"), _chunk("q3", "a3")])
    assert len(ids) == 3
    by_id = {c.id: c for c in await repository.list_pending_chunks()}
    assert by_id[ids[1]].prev_chunk_id == ids[0]
    assert by_id[ids[1]].next_chunk_id == ids[2]
    assert by_id[ids[0]].prev_chunk_id is None


async def test_vectorize_resumes_after_crash(db_session_factory, milvus, monkeypatch):
    c, coll = milvus
    await dualwrite.write_pending([_chunk(f"q{i}", f"a{i}") for i in range(4)])

    # 第一趟:embed 在第 2 批抛错(batch_size=2 → 前 2 条 done,后 2 条仍 pending)
    calls = {"n": 0}

    async def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("嵌入中断")
        return [[float(i), 0.0, 1.0] + [0.0] * (mc.DIM - 3) for i, _ in enumerate(texts)]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", flaky_embed)
    with pytest.raises(RuntimeError):
        await dualwrite.vectorize_pending(c, batch_size=2, collection=coll)
    assert await repository.count_chunks_by_status("done") == 2
    assert await repository.count_chunks_by_status("pending") == 2

    # 第二趟:恢复正常 → 只捡剩下 2 条 pending,补齐;Milvus 无重(按 id upsert)
    async def ok_embed(texts):
        return [[9.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3) for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok_embed)
    done_now = await dualwrite.vectorize_pending(c, batch_size=2, collection=coll)
    assert done_now == 2
    assert await repository.count_chunks_by_status("pending") == 0
    assert await repository.count_chunks_by_status("done") == 4
    assert mc.count(c, collection=coll) == 4  # 无重无漏
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_dualwrite.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.dualwrite`)。

- [ ] **Step 3: 实现**

Create `work/ch03/app/kb/dualwrite.py`:

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
    """幂等可重跑:取 pending → 拼 category+questions+answer 向量化 → Milvus upsert(PK=id)
    → flush → 回填 vector_id、status=done。崩在任意批,重跑只捡剩余 pending(Milvus 按 id upsert 无重)。"""
    pending = await repository.list_pending_chunks()
    done = 0
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        rows = [
            {"id": r.id, "vector": v, "question": r.questions, "answer": r.answer}
            for r, v in zip(batch, vectors)
        ]
        c = client or milvus_client.get_client()
        milvus_client.ensure_collection(c, collection=collection)
        milvus_client.upsert_vectors(c, rows, collection=collection)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    if done:
        c = client or milvus_client.get_client()
        milvus_client.flush(c, collection=collection)
    return done
```

Create `work/ch03/scripts/vectorize_kb.py`:

```python
"""CLI:把 knowledge_chunks 里 pending 的块向量化写入 Milvus(幂等可重跑)。
运行:PYTHONPATH=. uv run python scripts/vectorize_kb.py"""
import asyncio

from app.kb import dualwrite, milvus_client


async def main() -> None:
    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    n = await dualwrite.vectorize_pending(client)
    print(f"✅ 本次向量化 {n} 块;Milvus 现有 {milvus_client.count(client)} 条")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_dualwrite.py -v`
Expected: PASS(2 passed)。恢复测试证明验收 2:中断后重跑,漏向量化的块被捡起补齐、无重无漏。

- [ ] **Step 5: 提交**

```bash
git add app/kb/dualwrite.py scripts/vectorize_kb.py tests/test_dualwrite.py
git commit -m "feat(ch03): 双写 write_pending + vectorize_pending(幂等恢复)+ vectorize CLI"
```

---

## Task 11: 去重 `app/kb/dedup.py`

**TDD**(纯逻辑)。文本归一化:去空白/标点,保留中文。

**Files:**
- Create: `work/ch03/app/kb/dedup.py`
- Test: `work/ch03/tests/test_dedup.py`

**Interfaces:**
- Produces:
  - `normalize_question(q: str) -> str`
  - `dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]`(返回 `(kept, discarded)`;`items` 是有 `.question` 属性的对象)

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_dedup.py`:

```python
from dataclasses import dataclass

from app.kb import dedup


@dataclass
class Item:
    question: str


def test_normalize_strips_punct_keeps_chinese():
    assert dedup.normalize_question(" 邮费,是多少? ") == "邮费是多少"


def test_dedupe_within_batch_and_against_existing():
    items = [Item("邮费是多少"), Item("邮费是多少?"), Item("怎么退货"), Item("运费怎么算")]
    kept, discarded = dedup.dedupe(items, existing_questions=["运费怎么算"])
    kept_q = [i.question for i in kept]
    assert "邮费是多少" in kept_q          # 首次出现保留
    assert "怎么退货" in kept_q
    assert "运费怎么算" not in kept_q       # 与库内已有重复 → 丢弃
    assert len(kept) == 2 and len(discarded) == 2
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.dedup`)。

- [ ] **Step 3: 实现**

Create `work/ch03/app/kb/dedup.py`:

```python
import re

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)  # \W 不含中文(中文属 \w),故只去空白/标点


def normalize_question(q: str) -> str:
    return _STRIP_RE.sub("", q.strip().lower())


def dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]:
    seen = {normalize_question(q) for q in existing_questions}
    kept, discarded = [], []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            discarded.append(item)
        else:
            seen.add(key)
            kept.append(item)
    return kept, discarded
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/dedup.py tests/test_dedup.py
git commit -m "feat(ch03): 暂存表文本归一化去重 dedupe"
```

---

## Task 12: 对话挖知识 `app/kb/mining.py` + CLI(Prompt → eval)

**纯 Prompt 任务:不写单测,用标注样例 eval 验证**(spec §12)。抽取用 `PROMPT | model.with_structured_output(...)`。

**Files:**
- Create: `work/ch03/app/kb/mining.py`、`work/ch03/scripts/mine_knowledge.py`
- Modify: `work/ch03/app/core/prompts.py`(加 `MINING_SYSTEM` / `MINING_PROMPT`)、`work/ch03/app/db/repository.py`(加 `list_conversations_with_messages`)

**Interfaces:**
- Consumes: `get_chat_model`、`repository`(staging/knowledge)、`dedup.dedupe`、`dualwrite.write_pending`、`documents.Chunk`。
- Produces:
  - Pydantic `QaPair(question, answer)` / `QaExtraction(questions, answers)`
  - `async extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]`
  - `async mine(batch_size=20, model=None) -> dict`(读对话→分批抽取写 staging→整体去重→kept 写 knowledge pending;返回统计)

- [ ] **Step 1: 加抽取 prompt**

Edit `work/ch03/app/core/prompts.py`,末尾追加:

```python
MINING_SYSTEM = """你是客服知识库构建助手。下面是若干条历史客服对话(用户问 + 客服答)。
请从中抽取「可复用的问答对」,用于沉淀到 FAQ 知识库。要求:
- 只抽有普适价值的问答(政策、流程、时效、费用等),忽略闲聊、纯个案(如某具体订单号的状态)。
- question 用简洁的通用问法(去掉具体订单号/人名),answer 忠于客服原答、不编造承诺。
- 一条对话可能不含任何可复用问答,此时不要硬抽。
- 退款/售后时效统一表述为「以平台售后规则为准」。"""

MINING_PROMPT = ChatPromptTemplate.from_messages(
    [("system", MINING_SYSTEM), ("human", "历史对话:\n{conversations}")]
)
```

> 若 `app/core/prompts.py` 未导入 `ChatPromptTemplate`,补 `from langchain_core.prompts import ChatPromptTemplate`。

- [ ] **Step 2: 实现 mining**

Create `work/ch03/app/kb/mining.py`:

```python
import logging
from datetime import datetime

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db import repository
from app.kb import dedup, dualwrite
from app.kb.documents import Chunk

logger = logging.getLogger(__name__)


class QaPair(BaseModel):
    question: str
    answer: str


class QaExtraction(BaseModel):
    """结构化输出 schema:用两个并列标量数组一一对应。
    嵌套对象数组(list[对象])在部分 OpenAI 兼容上游会 502,标量 list[str] 稳,故并列数组规避。"""

    questions: list[str] = Field(default_factory=list,
                                 description="问法列表,与 answers 一一对应;无可复用问答则为空")
    answers: list[str] = Field(default_factory=list,
                               description="答案列表,与 questions 一一对应")


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    model = model or get_chat_model()
    chain = MINING_PROMPT | model.with_structured_output(QaExtraction)
    result: QaExtraction = await chain.ainvoke({"conversations": "\n---\n".join(conversation_texts)})
    qs, ans = result.questions, result.answers
    n = min(len(qs), len(ans))
    if len(qs) != len(ans):
        logger.warning("并列数组长度不一致 questions=%d answers=%d,按较短对齐", len(qs), len(ans))
    return [QaPair(question=qs[i], answer=ans[i]) for i in range(n)]


async def _load_conversation_texts() -> list[tuple[str, str]]:
    """返回 [(source_ref, 对话文本)];对话文本由该会话的 user/assistant 消息拼成。"""
    convs = await repository.list_conversations_with_messages()
    out: list[tuple[str, str]] = []
    for conv_id, msgs in convs:
        lines = [f"{m.role}: {m.content}" for m in msgs if m.content]
        if lines:
            out.append((f"conv:{conv_id}", "\n".join(lines)))
    return out


async def mine(batch_size: int = 20, model=None) -> dict:
    sources = await _load_conversation_texts()
    batch_no = datetime.now().strftime("mine-%Y%m%d%H%M%S")
    # 1) 分批抽取 → 写 staging(extracted)
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        for p in pairs:
            await repository.insert_staging(batch_no, batch[0][0], p.question, p.answer)
    # 2) 整体去重(本轮 extracted 内 + 对已有 knowledge)
    staged = [s for s in await repository.list_staging_by_status("extracted") if s.batch_no == batch_no]
    existing = await repository.list_all_questions()
    kept, discarded = dedup.dedupe(staged, existing)
    await repository.set_staging_status([s.id for s in kept], "kept")
    await repository.set_staging_status([s.id for s in discarded], "discarded")
    # 3) kept 写 knowledge_chunks(pending, content_type=mined, questions=真实问法)
    chunks = [Chunk(category="历史对话", questions=s.question, answer=s.answer,
                    section_path="mined", content_type="mined") for s in kept]
    if chunks:
        await dualwrite.write_pending(chunks)
    return {"sources": len(sources), "extracted": len(staged),
            "kept": len(kept), "discarded": len(discarded)}
```

- [ ] **Step 3: 加 repository 依赖方法**

Edit `work/ch03/app/db/repository.py`,末尾追加(mining 需要按会话取消息):

```python
async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    async with db.async_session() as s:
        conv_ids = list((await s.execute(select(Conversation.id).order_by(Conversation.id))).scalars())
        out: list[tuple[int, list[Message]]] = []
        for cid in conv_ids:
            msgs = list((await s.execute(
                select(Message).where(Message.conversation_id == cid).order_by(Message.id)
            )).scalars())
            out.append((cid, msgs))
        return out
```

- [ ] **Step 4: 写 CLI**

Create `work/ch03/scripts/mine_knowledge.py`:

```python
"""CLI(可重跑 job):从历史对话挖 QA → 暂存表 → 去重 → 写 knowledge_chunks(pending)。
之后跑 scripts/vectorize_kb.py 向量化。运行:PYTHONPATH=. uv run python scripts/mine_knowledge.py"""
import asyncio

from app.kb import mining


async def main() -> None:
    stats = await mining.mine()
    print(f"✅ 挖知识:{stats}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 冒烟(结构验证,不算 eval)**

先确保有历史对话(Task 16 会 seed;若还没有,本步可跑通 0 条)。
Run: `PYTHONPATH=. uv run python scripts/mine_knowledge.py`
Expected: 打印统计 dict,不报错。真正的抽取质量在 Task 16 的 `eval_mining.py` 里用标注样例验。

- [ ] **Step 6: 提交**

```bash
git add app/kb/mining.py scripts/mine_knowledge.py app/core/prompts.py app/db/repository.py
git commit -m "feat(ch03): 对话挖知识 mining(结构化抽取→staging→去重→pending)+ CLI"
```

---

## Task 13: 在线检索 `app/core/retrieval.py`

**TDD**(mock embed + 真实 Milvus Standalone 测试集合)。

**Files:**
- Create: `work/ch03/app/core/retrieval.py`
- Test: `work/ch03/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `embeddings.embed_query`、`milvus_client.*`、`settings.retrieval_top_k / retrieval_min_score`。
- Produces: `async search_knowledge(query: str, top_k: int | None = None, min_score: float | None = None, client=None, collection: str = COLLECTION) -> list[dict]`(返回 `{"id","score","question","answer"}`,按 min_score 过滤)。

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_retrieval.py`:

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
        {"id": 1, "vector": v_hit, "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": v_other, "question": "发货时效", "answer": "48小时"},
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

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL(`ModuleNotFoundError: app.core.retrieval`)。

- [ ] **Step 3: 实现**

Create `work/ch03/app/core/retrieval.py`:

```python
from app.config import settings
from app.core import embeddings
from app.kb import milvus_client


async def search_knowledge(
    query: str, top_k: int | None = None, min_score: float | None = None,
    client=None, collection: str = milvus_client.COLLECTION,
) -> list[dict]:
    """dense 单路:embed(query) → Milvus Top-K → 按 min_score 过滤。
    Standalone COSINE 的 score 即相似度(越大越像),故直接 score >= min_score。"""
    top_k = top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if min_score is None else min_score
    vector = await embeddings.embed_query(query)
    c = client or milvus_client.get_client()
    milvus_client.ensure_collection(c, collection=collection)
    hits = milvus_client.search(c, vector, top_k, collection=collection)
    return [h for h in hits if h["score"] >= min_score]
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/core/retrieval.py tests/test_retrieval.py
git commit -m "feat(ch03): 在线检索 search_knowledge(embed→Milvus Top-K→阈值过滤)"
```

---

## Task 14: `query_faq` 改写(契约不变)

**TDD**(mock retrieval)。只换 `query_faq` 内部,签名/出参/docstring 不动;registry 不动。

**Files:**
- Modify: `work/ch03/app/tools/business.py`
- Test: `work/ch03/tests/test_tool_faq_vector.py`
- Delete: `work/ch03/tests/test_tool_faq.py`(ch02 断言 LIKE 查表,新实现已取代)

**Interfaces:**
- Consumes: `retrieval.search_knowledge`。
- Produces: `query_faq(keyword)` 出参不变:命中 `{"hits": [{"question","answer"}]}`,未命中 `{"hits": [], "message": ...}`。

- [ ] **Step 1: 写失败测试**

Create `work/ch03/tests/test_tool_faq_vector.py`:

```python
from app.tools.business import query_faq


async def test_query_faq_maps_hits_to_contract(monkeypatch):
    async def fake_search(keyword, **kw):
        return [{"id": 1, "score": 0.8, "question": "运费怎么算", "answer": "满99包邮"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "邮费是多少"})
    assert out == {"hits": [{"question": "运费怎么算", "answer": "满99包邮"}]}


async def test_query_faq_empty_returns_message(monkeypatch):
    async def fake_search(keyword, **kw):
        return []
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "无关问题"})
    assert out["hits"] == []
    assert "未找到" in out["message"]
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_tool_faq_vector.py -v`
Expected: FAIL(`AttributeError: module 'app.tools.business' has no attribute 'retrieval'`)。

- [ ] **Step 3: 实现**

Edit `work/ch03/app/tools/business.py`:顶部在 `from app.db import repository` 旁加 `from app.core import retrieval`;改写 `query_faq` 函数体(**docstring 与 `FaqInput` / 签名保持不变**),把调 `repository.search_faq` 换成 `retrieval.search_knowledge`:

```python
@tool(args_schema=FaqInput)
async def query_faq(keyword: str) -> dict:
    """根据关键词查询常见问题解答(FAQ)。用于用户咨询政策、规则、操作流程等通用问题时。"""
    hits = await retrieval.search_knowledge(keyword)
    if not hits:
        return {"hits": [], "message": f"未找到与「{keyword}」相关的常见问题"}
    return {"hits": [{"question": h["question"], "answer": h["answer"]} for h in hits]}
```

(`repository` 在 `business.py` 仍被 `create_ticket` 用,保留 import。删掉 `work/ch03/tests/test_tool_faq.py`——它断言的是 ch02 的 LIKE 查表行为,已被本任务语义取代。)

- [ ] **Step 4: 运行看通过 + 全量回归**

Run: `uv run pytest tests/test_tool_faq_vector.py -v && uv run pytest -q`
Expected: 新测 PASS;全量绿(ch01/ch02 不回归)。

- [ ] **Step 5: 提交**

```bash
rm -f tests/test_tool_faq.py
git add app/tools/business.py tests/test_tool_faq_vector.py
git add -u tests/test_tool_faq.py
git commit -m "feat(ch03): query_faq 内部改向量检索(入参出参契约不变)"
```

---

## Task 15: 源文档夹具 + sources.py + build_kb/show_kb CLI + Makefile

**集成/夹具**(主会话)。源文档已在仓库根,校验后收进单一材料清单,写 build/preview CLI,加 Makefile 目标。

**Files:**
- Verify: `../../data/kb/*.md`(仓库已有 6 份)、`../../sql/ch03-ddl.sql`、`../../sql/ch03-seed.sql`
- Create: `work/ch03/app/kb/sources.py`、`work/ch03/scripts/build_kb.py`、`work/ch03/scripts/show_kb.py`、`work/ch03/Makefile`
- Modify: `work/ch03/app/db/repository.py`(加 `count_chunks_by_content_types`)

**Interfaces:**
- Consumes: `documents.build_chunks`、`dualwrite.write_pending`。
- Produces: `app.kb.sources.{KB_DIR, SOURCE_TYPES, CONTENT_TYPES, CONTENT_TYPE_DESC}`。

- [ ] **Step 1: 校验源文档在仓库根(6 份)**

Run:
```bash
ls -1 ../../data/kb/
```
Expected: `after-sales-manual.md billing-shipping.md member-benefits.md product-faq.md product-specs.md returns-policy.md`。
> 这 6 份已是本章夹具(**含运费/邮费块**——验收1 的召回目标;大表格;长段落;政策类无天然问题章节;规格类含具体型号)。无需新建。

- [ ] **Step 2: 写 `app/kb/sources.py`(材料清单一处定义)**

Create `work/ch03/app/kb/sources.py`:

```python
"""建库材料清单:哪些文件进知识库、各自算什么内容类型。

一处定义、三处共用——离线建库 CLI(scripts/build_kb.py)、切块预览(scripts/show_kb.py)、
录入页(/kb)。三边各写一份就会出现「预览看到三份、实际入库四份」这种对不上的账。
"""
import pathlib

KB_DIR = pathlib.Path(__file__).resolve().parents[4] / "data" / "kb"   # work/ch03/app/kb → 仓库根

# 文件 → content_type
SOURCE_TYPES: dict[str, str] = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
    "product-specs.md": "spec",      # 带具体型号的商品规格
    "member-benefits.md": "policy",  # 会员等级与积分,规则型内容归 policy
    "billing-shipping.md": "policy", # 开票与配送,同上
}

# 录入页允许选的内容类型:政策/手册这类没有天然问题,questions 落章节标题
CONTENT_TYPES: tuple[str, ...] = ("faq", "policy", "manual", "spec")
CONTENT_TYPE_DESC: dict[str, str] = {
    "faq": "商品 FAQ:questions 填真实问法",
    "policy": "政策条款:questions 填章节标题、category 填上级路径",
    "manual": "售后手册:同政策,按标题层级切",
    "spec": "商品规格:含具体型号",
}
```

- [ ] **Step 3: 写 build_kb CLI(幂等守卫)+ show_kb 预览 CLI**

Create `work/ch03/scripts/build_kb.py`:

```python
"""CLI:把 data/kb/*.md 切块写入 knowledge_chunks(pending)。之后跑 vectorize_kb.py。
运行:PYTHONPATH=. uv run python scripts/build_kb.py(或在录入页 /kb 上按「离线建库」)"""
import asyncio

from app.db import repository
from app.kb import documents, dualwrite
from app.kb.sources import KB_DIR, SOURCE_TYPES as DOCS


async def main() -> None:
    # 幂等守卫:文档块非幂等插入,重跑会重复。已存在则跳过,重建走 make kb-reset。
    existing = await repository.count_chunks_by_content_types(set(DOCS.values()))
    if existing:
        print(f"⚠️ 已存在 {existing} 条文档块,跳过以防重复插入。重建请先: make kb-reset")
        return
    total = 0
    for fname, ctype in DOCS.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        ids = await dualwrite.write_pending(chunks)
        total += len(ids)
        print(f"  {fname}: {len(ids)} 块")
    print(f"✅ 建库(pending):共 {total} 块。下一步:scripts/vectorize_kb.py")


if __name__ == "__main__":
    asyncio.run(main())
```

Create `work/ch03/scripts/show_kb.py`:

```python
"""验收展示(dry-run):列出建库材料 + 预览结构化切块结果,不写库、不碰 Milvus、不调上游。
运行:PYTHONPATH=. uv run python scripts/show_kb.py"""
import pathlib

from app.kb import chunking, documents
from app.kb.sources import KB_DIR, SOURCE_TYPES as DOCS

ROOT = pathlib.Path(__file__).resolve().parents[3]
SEED_SQL = ROOT / "sql" / "ch03-seed.sql"


def _preview(text: str, n: int = 46) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[:n] + "…"


def main() -> None:
    print("=== 离线建库材料清单 ===")
    for fname, ctype in DOCS.items():
        raw = (KB_DIR / fname).read_text(encoding="utf-8")
        print(f"  data/kb/{fname:<24} [{ctype:<6}] {len(raw):>4} 字符 / {raw.count(chr(10)) + 1} 行")
    print(f"历史会话种子:{SEED_SQL.relative_to(ROOT)}(喂 make kb-mine)\n")
    print("=== 切块预览(dry-run,不写库)===")
    tally: dict[str, int] = {}
    key_total = 0
    for fname, ctype in DOCS.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        tally[ctype] = tally.get(ctype, 0) + len(chunks)
        by_path: dict[str, list[documents.Chunk]] = {}
        for c in chunks:
            by_path.setdefault(c.section_path, []).append(c)
        print(f"\n▼ {fname} [{ctype}]  →  {len(chunks)} 块")
        for path, group in by_path.items():
            multi = f"  (该节切成 {len(group)} 块)" if len(group) > 1 else ""
            print(f"  ┌ 节:{path}{multi}")
            for j, c in enumerate(group, 1):
                key_total += c.is_key_clause
                flag = " ★关键条款" if c.is_key_clause else ""
                kind = "表格块" if chunking.is_table_block(c.answer) else "文本块"
                seq = f"{j}/{len(group)}" if len(group) > 1 else "-"
                print(f"  │ [{kind} {seq}]{flag}  Q={c.questions}  category={c.category}")
                print(f"  │   A({len(c.answer)}字): {_preview(c.answer)}")
    print("\n=== 汇总 ===")
    print(f"总块数 {sum(tally.values())};类型分布 " + " ".join(f"{k}={v}" for k, v in tally.items()))
    print(f"关键条款(is_key_clause=1)共 {key_total} 块")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 加 repository 依赖方法 + 写 Makefile**

Edit `work/ch03/app/db/repository.py`,末尾追加:

```python
async def count_chunks_by_content_types(content_types) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.content_type.in_(list(content_types)))
        )
        return int(result.scalar_one())
```

Create `work/ch03/Makefile`:

```makefile
# Python 强制 UTF-8(中文 Windows 默认 GBK,读带中文的 markdown/jsonl 会 UnicodeDecodeError)
export PYTHONUTF8 = 1

.PHONY: dev milvus-up milvus-down kb-preview kb-build kb-vectorize kb-mine seed-conv kb-reset eval-retrieval eval-mining test

dev:
	uv run uvicorn app.main:app --reload --port 8000

# ch03: Milvus Standalone 起停(etcd + minio + milvus)
milvus-up:
	docker compose up -d etcd minio milvus-standalone
	@echo "等待 Milvus 就绪(healthz)..."; \
	for i in $$(seq 1 60); do \
	  curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && echo "Milvus OK" && exit 0; \
	  sleep 3; done; echo "Milvus 未就绪" && exit 1

milvus-down:
	docker compose stop etcd minio milvus-standalone

kb-preview:
	PYTHONPATH=. uv run python scripts/show_kb.py

kb-build:
	PYTHONPATH=. uv run python scripts/build_kb.py

kb-vectorize:
	PYTHONPATH=. uv run python scripts/vectorize_kb.py

kb-mine:
	PYTHONPATH=. uv run python scripts/mine_knowledge.py

seed-conv:
	docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < ../../sql/ch03-seed.sql

# 干净重建知识库:清 MySQL 两表 + drop Milvus collection
kb-reset:
	docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SET FOREIGN_KEY_CHECKS=0; DELETE FROM knowledge_chunks; DELETE FROM qa_extraction_staging; SET FOREIGN_KEY_CHECKS=1;"
	PYTHONPATH=. uv run python -c "from app.kb import milvus_client as m; m.drop(m.get_client(), 'knowledge')"
	@echo "KB 已重置。重跑: make kb-build && make kb-vectorize"

eval-retrieval:
	PYTHONPATH=. uv run python scripts/eval_retrieval.py

eval-mining:
	PYTHONPATH=. uv run python scripts/eval_mining.py

test:
	uv run pytest -v
```

- [ ] **Step 5: 验证 CLI 可跑(dry-run)**

Run: `make kb-preview`
Expected: 打印材料清单与各文件切块结果(含表格按行拆迹象),不写库。

- [ ] **Step 6: 提交**

```bash
git add app/kb/sources.py scripts/build_kb.py scripts/show_kb.py app/db/repository.py Makefile
git commit -m "feat(ch03): 材料清单 sources.py + build_kb/show_kb CLI + Makefile(milvus-up/kb-*)"
```

---

## Task 16: 标注样例 + eval 脚本 + 合成对话 seed

**eval**(主会话)。给验收1 建检索 eval;给挖知识建抽取 eval;seed 合成历史对话喂 mining。

**Files:**
- Create: `work/ch03/tests/data/retrieval_samples.json`、`work/ch03/tests/data/mining_samples.json`、`work/ch03/scripts/eval_retrieval.py`、`work/ch03/scripts/eval_mining.py`
- Verify: `../../sql/ch03-seed.sql`(仓库根已有)、`work/ch03/Makefile` 的 `seed-conv` / `eval-retrieval` / `eval-mining` 三个目标(Task 15 已写好)

- [ ] **Step 1: 检索标注样例(换说法 → 期望命中)**

Create `work/ch03/tests/data/retrieval_samples.json`:

```json
[
  {"query": "邮费是多少", "expect_answer_contains": "包邮"},
  {"query": "快递费怎么收", "expect_answer_contains": "运费"},
  {"query": "多久能发货", "expect_answer_contains": "48"},
  {"query": "东西不想要了能退吗", "expect_answer_contains": "7 天"},
  {"query": "怎么开发票", "expect_answer_contains": "开票"}
]
```

- [ ] **Step 2: 检索 eval 脚本(直接调 search_knowledge,纯测向量召回)**

Create `work/ch03/scripts/eval_retrieval.py`:

```python
"""验收1 eval:对换说法的问题跑向量检索,核对是否召回期望内容。
需已建库并向量化(make kb-build && make kb-vectorize)+ 真实进程内嵌入。
运行:PYTHONPATH=. uv run python scripts/eval_retrieval.py"""
import asyncio
import json
import pathlib
import sys

from app.core import retrieval

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests/data/retrieval_samples.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        hits = await retrieval.search_knowledge(s["query"])
        top = hits[0] if hits else None
        ok = bool(top) and s["expect_answer_contains"] in top["answer"]
        failures += not ok
        print(f"{'✅' if ok else '❌'} {s['query']!r} -> "
              f"{(top['question'] + ' | ' + top['answer'][:30]) if top else '(空)'}")
    print(f"\n召回正确 {len(samples) - failures}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 3: 挖知识标注样例 + eval**

Create `work/ch03/tests/data/mining_samples.json`:

```json
[
  {
    "conversation": "user: 你们运费怎么算啊\nassistant: 单笔订单满99元包邮,未满收取10元运费,偏远地区另计。",
    "expect_question_contains": "运费"
  },
  {
    "conversation": "user: 我的订单MH123到哪了\nassistant: 您的包裹在广州分拨中心,预计明天送达。",
    "expect_empty": true
  }
]
```

Create `work/ch03/scripts/eval_mining.py`:

```python
"""挖知识 eval:对样例对话跑抽取,核对该抽的抽到、不该抽的(纯个案)不硬抽。
需聊天上游可用。运行:PYTHONPATH=. uv run python scripts/eval_mining.py"""
import asyncio
import json
import pathlib
import sys

from app.kb import mining

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests/data/mining_samples.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        pairs = await mining.extract_qa([s["conversation"]])
        if s.get("expect_empty"):
            ok = len(pairs) == 0
        else:
            ok = any(s["expect_question_contains"] in p.question for p in pairs)
        failures += not ok
        print(f"{'✅' if ok else '❌'} 抽到 {len(pairs)} 对:{[p.question for p in pairs]}")
    print(f"\n抽取符合预期 {len(samples) - failures}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 4: 校验收合成对话 seed 与 Makefile 目标**

Run:
```bash
head -5 ../../sql/ch03-seed.sql
grep -E '^(seed-conv|eval-retrieval|eval-mining):' Makefile
```
Expected: 前者有 `SET NAMES utf8mb4;` 与幂等 `DELETE FROM ... WHERE user_id LIKE 'seed-%'`;后者三行目标都在(Task 15 已写好),`seed-conv` 指向 `../../sql/ch03-seed.sql`。

- [ ] **Step 5: 跑 seed + eval 试跑(可选,需服务/上游)**

Run(需 mysql 容器在跑):
```bash
make seed-conv
```
Expected: 打印 `after_conv=2`、`after_msg=4`。

- [ ] **Step 6: 提交**

```bash
git add tests/data/retrieval_samples.json tests/data/mining_samples.json scripts/eval_retrieval.py scripts/eval_mining.py
git commit -m "feat(ch03): 检索/挖知识标注样例 + eval 脚本(seed 复用仓库根 ch03-seed.sql)"
```

---

## Task 17: 录入 API + 作业运行器(TDD)

**TDD**。页面要的读数与写入都在这一层:库存盘点、切块预览(dry-run)、录入落库、补向量、检索自测;外加一个只能跑白名单 make 目标的作业运行器。

**Files:**
- Modify: `work/ch03/app/db/repository.py`(加 `knowledge_stats` / `list_recent_chunks` / `list_chunk_pairs` / `staging_stats`)
- Create: `work/ch03/app/core/jobs.py`、`work/ch03/app/api/jobs.py`、`work/ch03/app/api/kb.py`、`work/ch03/app/api/admin.py`
- Modify: `work/ch03/app/main.py`(挂 kb/jobs/admin 三个 router)
- Test: `work/ch03/tests/test_kb_api.py`、`work/ch03/tests/test_jobs_api.py`、`work/ch03/tests/test_admin_api.py`
- Modify: `work/ch03/tests/conftest.py`(加 `client` fixture)

**Interfaces:**
- Produces:
  - `GET /api/kb/overview`、`POST /api/kb/preview`、`POST /api/kb/ingest`、`POST /api/kb/vectorize`、`POST /api/kb/search`、`GET /api/kb/staging`
  - `GET /api/jobs`、`POST /api/jobs/{name}`、`GET /api/jobs/{name}`、`POST /api/jobs/{name}/stop`
  - `GET /api/admin/overview`
  - `jobs.JOBS`(ch03 白名单)、`jobs.{start,stop,status,status_all,tail}`

- [ ] **Step 1: conftest 加 client fixture + 写录入 API 四条口径测试(先红)**

Edit `work/ch03/tests/conftest.py`,末尾追加:

```python
@pytest_asyncio.fixture()
async def client():
    from httpx import ASGITransport, AsyncClient
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
```

Create `work/ch03/tests/test_kb_api.py`:

```python
from app.db import repository


async def test_preview_is_dry_run(client, db_session_factory):
    """预览只切不写:按完预览再看库存,一块都不该多。"""
    r = await client.post("/api/kb/preview",
                          json={"text": "# 售后\n\n## 运费说明\n\n满99包邮,未满10元运费。", "content_type": "policy"})
    assert r.status_code == 200
    assert r.json()["total"] >= 1
    assert await repository.count_chunks_by_status("pending") == 0


async def test_ingest_then_reingest_is_idempotent(client, db_session_factory):
    """Review Focus 4:同一份正文重录:第二次 inserted=0、skipped=全部,库里总数不变。"""
    body = {"text": "# 售后\n\n## 运费说明\n\n满99包邮,未满10元运费。", "content_type": "policy",
            "vectorize": False}
    r1 = await client.post("/api/kb/ingest", json=body)
    assert r1.status_code == 200 and r1.json()["inserted"] >= 1
    total1 = (await repository.knowledge_stats())["total"]
    r2 = await client.post("/api/kb/ingest", json=body)
    assert r2.json()["inserted"] == 0 and r2.json()["skipped"] == r1.json()["chunks"]
    assert (await repository.knowledge_stats())["total"] == total1


async def test_ingest_keeps_multiple_pieces_of_one_section(client, db_session_factory):
    """大表格按行拆出的多块共用节标题——只按问法查重会误杀,这里锁住不许。"""
    rows = "\n".join(f"| 商品{i} | {i} |" for i in range(1, 40))
    text = f"# 价目表\n\n## 价目\n\n| 商品 | 价格 |\n| --- | --- |\n{rows}"
    r = await client.post("/api/kb/ingest", json={"text": text, "content_type": "manual", "vectorize": False})
    assert r.status_code == 200
    assert r.json()["inserted"] >= 3           # 同一节多块都进了,没被误杀


async def test_ingest_empty_document_rejected(client, db_session_factory):
    """Review Focus 1:空正文 / 只有标题 → 400,且不落任何库。"""
    r = await client.post("/api/kb/ingest", json={"text": "# 只有标题\n", "content_type": "policy"})
    assert r.status_code == 400
    assert (await repository.knowledge_stats())["total"] == 0


async def test_preview_rejects_bad_input(client, db_session_factory):
    """空正文 / 非法 content_type / 不在材料清单里的文件名 一律 400。"""
    assert (await client.post("/api/kb/preview", json={"text": "   ", "content_type": "faq"})).status_code == 400
    assert (await client.post("/api/kb/preview",
                              json={"text": "正文。", "content_type": "nope"})).status_code == 400
    assert (await client.post("/api/kb/preview",
                              json={"file": "../secret.md"})).status_code == 400


async def test_overview_milvus_offline_is_state_not_error(client, db_session_factory, monkeypatch):
    """Review Focus 3:Milvus 离线 → overview 返回 online=false、consistent=None,不 500。"""
    def boom():
        raise RuntimeError("connection refused")
    monkeypatch.setattr("app.kb.milvus_client.get_client", boom)
    r = await client.get("/api/kb/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["milvus"]["online"] is False
    assert body["consistent"] is None
```

- [ ] **Step 2: 写作业运行器安全边界测试(先红)**

Create `work/ch03/tests/test_jobs_api.py`:

```python
import pathlib

from app.core import jobs

MAKEFILE = pathlib.Path(__file__).resolve().parents[1] / "Makefile"   # tests/ → work/ch03/


async def test_unknown_job_is_rejected(client):
    """作业名不在白名单 → 404;命令片段没有任何地方能传进来。"""
    assert (await client.post("/api/jobs/rm-rf")).status_code == 404
    assert (await client.get("/api/jobs/rm-rf")).status_code == 404


async def test_running_job_refuses_reentry(client, monkeypatch):
    """同名作业还在跑 → 409,别让两份进程抢同一批产物。"""
    async def running(name):
        raise RuntimeError("正在运行中")
    monkeypatch.setattr(jobs, "start", running)
    assert (await client.post("/api/jobs/kb-build")).status_code == 409


def test_every_job_target_exists_in_makefile():
    """注册表里每个作业都得是 Makefile 里真有的目标,免得按钮指向不存在的配方。"""
    text = MAKEFILE.read_text(encoding="utf-8")
    for spec in jobs.JOBS.values():
        assert spec.argv[:2] == ("make", spec.argv[1])
        assert f"\n{spec.argv[1]}:" in text, f"{spec.argv[1]} 不在 Makefile"
```

- [ ] **Step 3: 写后台聚合 API 测试(先红)**

Create `work/ch03/tests/test_admin_api.py`:

```python
async def test_admin_overview_has_kb_card(client, db_session_factory, monkeypatch):
    from app.api import kb
    async def fake_milvus():
        return {"online": True, "count": 0, "collection": "knowledge"}
    monkeypatch.setattr(kb, "milvus_state", fake_milvus)
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    card = next(c for c in r.json()["modules"] if c["key"] == "kb")
    assert card["title"] == "知识库"
    assert card["page"] == "/kb"
```

- [ ] **Step 4: 运行看失败**

Run: `uv run pytest tests/test_kb_api.py tests/test_jobs_api.py tests/test_admin_api.py -v`
Expected: 全红(模块还不存在 / 路由未挂载)。

- [ ] **Step 5: 实现 repository 盘点方法**

Edit `work/ch03/app/db/repository.py`,末尾追加:

```python
async def knowledge_stats() -> dict:
    """知识库盘点:总块数 + 双写状态分布 + 内容类型分布 + 关键条款数。"""
    async with db.async_session() as s:
        total = (await s.execute(select(func.count()).select_from(KnowledgeChunk))).scalar_one()
        by_status = {k: int(v) for k, v in (await s.execute(
            select(KnowledgeChunk.vectorize_status, func.count())
            .group_by(KnowledgeChunk.vectorize_status))).all()}
        by_type = {(k or "未标注"): int(v) for k, v in (await s.execute(
            select(KnowledgeChunk.content_type, func.count())
            .group_by(KnowledgeChunk.content_type))).all()}
        key_clause = (await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.is_key_clause == 1))).scalar_one()
    return {"total": int(total), "pending": by_status.get("pending", 0),
            "done": by_status.get("done", 0), "by_content_type": by_type,
            "key_clause": int(key_clause)}


async def list_recent_chunks(limit: int = 20) -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit))
        return list(result.scalars())


async def list_chunk_pairs() -> list[tuple[str, str]]:
    """全表的 (questions, answer) 对,给录入查重算指纹(只按问法会误杀同节多块)。"""
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions, KnowledgeChunk.answer))
        return [(q, a) for q, a in result.all()]


async def staging_stats() -> dict:
    """抽 QA 暂存表盘点:三个状态各几条 + 批次数 + 最近批次号。"""
    async with db.async_session() as s:
        counts = {k: int(v) for k, v in (await s.execute(
            select(QaExtractionStaging.status, func.count())
            .group_by(QaExtractionStaging.status))).all()}
        batches = (await s.execute(
            select(func.count(func.distinct(QaExtractionStaging.batch_no))))).scalar_one()
    return {"counts": {k: counts.get(k, 0) for k in ("extracted", "kept", "discarded")},
            "total": sum(counts.values()), "batches": int(batches)}
```

- [ ] **Step 6: 实现作业运行器 `app/core/jobs.py`**

Create `work/ch03/app/core/jobs.py`:

```python
"""后台作业运行器:把 make 目标搬到浏览器上按,日志轮询回显,长活能停。

安全边界:只能跑 JOBS 注册表里的目标,argv 全部写死在本模块。前端只传一个 job 名,
传不进任何命令片段、参数或路径——页面有「重跑」按钮,但没有 shell。
"""
import asyncio
import contextlib
import datetime as dt
import logging
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]      # work/ch03
LOG_DIR = REPO_ROOT / "log" / "acceptance"
TAIL_LINES = 400


@dataclass(frozen=True)
class JobSpec:
    name: str
    title: str
    argv: tuple[str, ...]
    needs: str
    heavy: bool = False


JOBS: dict[str, JobSpec] = {
    j.name: j for j in (
        JobSpec("kb-preview", "材料清单与切块预览", ("make", "kb-preview"), "本地跑,不写库"),
        JobSpec("kb-build", "离线建库(文档切块 → pending)", ("make", "kb-build"), "需 mysql"),
        JobSpec("kb-mine", "对话挖知识(抽 QA → 去重 → pending)", ("make", "kb-mine"),
                "需 mysql + 聊天上游", heavy=True),
        JobSpec("kb-vectorize", "向量化(嵌入 → Milvus → 回标 done)", ("make", "kb-vectorize"),
                "需 mysql + Milvus"),
        JobSpec("seed-conv", "灌历史会话种子", ("make", "seed-conv"), "需 docker mysql 容器"),
        JobSpec("kb-reset", "清库重建(清两表 + drop 集合)", ("make", "kb-reset"),
                "需 mysql + Milvus;会清空知识库", heavy=True),
    )
}


@dataclass
class JobRun:
    status: str = "idle"          # idle | running | ok | failed | stopped
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    watcher: asyncio.Task | None = field(default=None, repr=False)


_runs: dict[str, JobRun] = {}


def _run(name: str) -> JobRun:
    return _runs.setdefault(name, JobRun())


def log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


async def start(name: str) -> JobRun:
    """发起一次运行。同名作业未结束时拒绝重入(RuntimeError)。"""
    spec = JOBS[name]
    run = _run(name)
    if run.status == "running":
        raise RuntimeError(f"{spec.title}正在运行中(pid {run.pid}),等它跑完再发起")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = log_path(name)
    header = (f"$ {' '.join(spec.argv)}\n"
              f"# {dt.datetime.now().isoformat(timespec='seconds')} 由后台页发起\n\n")
    path.write_text(header, encoding="utf-8")          # 每次覆盖:窗口里只看本次
    fh = path.open("a", encoding="utf-8", buffering=1)
    try:
        proc = await asyncio.create_subprocess_exec(
            *spec.argv, cwd=REPO_ROOT, stdout=fh, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,     # 独立会话:kill 时连 make → uv → python 一起收
        )
    except Exception:
        fh.close()
        raise

    run.status, run.pid = "running", proc.pid
    run.started_at = dt.datetime.now().isoformat(timespec="seconds")
    run.finished_at = run.returncode = None
    run.proc = proc
    run.watcher = asyncio.create_task(_watch(name, run, proc, fh))
    return run


async def _watch(name: str, run: JobRun, proc: asyncio.subprocess.Process, fh) -> None:
    try:
        rc = await proc.wait()
    finally:
        with contextlib.suppress(Exception):
            fh.close()
    run.returncode = rc
    run.finished_at = dt.datetime.now().isoformat(timespec="seconds")
    if run.status != "stopped":        # 人为 kill 的退出码也非 0,别报成「失败」冤枉它
        run.status = "ok" if rc == 0 else "failed"
    run.proc = None


async def stop(name: str) -> None:
    run = _run(name)
    proc = run.proc
    if run.status != "running" or proc is None:
        raise RuntimeError("该作业当前没有在运行")
    run.status = "stopped"
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def tail(name: str, lines: int = TAIL_LINES) -> str:
    path = log_path(name)
    if not path.exists():
        return ""
    rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(rows[-lines:])


def status(name: str, with_log: bool = False) -> dict:
    spec, run = JOBS[name], _run(name)
    path = log_path(name)
    out = {
        "name": name, "title": spec.title, "cmd": " ".join(spec.argv),
        "needs": spec.needs, "heavy": spec.heavy,
        "status": run.status, "pid": run.pid,
        "started_at": run.started_at, "finished_at": run.finished_at,
        "returncode": run.returncode,
        "log_mtime": (dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                      if path.exists() else None),
    }
    if with_log:
        out["log"] = tail(name)
    return out


def status_all() -> list[dict]:
    return [status(n) for n in JOBS]
```

- [ ] **Step 7: 实现作业 API `app/api/jobs.py`**

Create `work/ch03/app/api/jobs.py`:

```python
"""作业 API:后台页「重跑」按钮的唯一入口。发起 / 查状态与日志尾 / 停止。
白名单在 app/core/jobs.py,argv 写死在那边;名字不在注册表里就 404。"""
from fastapi import APIRouter, HTTPException

from app.core import jobs

router = APIRouter(prefix="/api/jobs")


def _known(name: str) -> None:
    if name not in jobs.JOBS:
        raise HTTPException(status_code=404, detail=f"未注册的作业:{name}")


@router.get("")
async def jobs_list() -> dict:
    return {"jobs": jobs.status_all()}


@router.post("/{name}")
async def job_start(name: str) -> dict:
    _known(name)
    try:
        await jobs.start(name)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"命令起不来:{e}") from e
    return jobs.status(name, with_log=True)


@router.get("/{name}")
async def job_status(name: str) -> dict:
    _known(name)
    return jobs.status(name, with_log=True)


@router.post("/{name}/stop")
async def job_stop(name: str) -> dict:
    _known(name)
    try:
        await jobs.stop(name)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return jobs.status(name, with_log=True)
```

- [ ] **Step 8: 实现录入 API `app/api/kb.py`**

Create `work/ch03/app/api/kb.py`:

```python
"""知识库录入 API:材料清单、切块预览、录入入库、向量化、检索自测。

手工录入(POST /ingest):正文在浏览器里贴,预览什么样就入库什么样——切块用
app.kb.documents.build_chunks,与离线 CLI 同一个函数。查重按「问法 + 正文」的指纹,
同一份正文重复录入天然幂等(指纹全撞、全跳过)。"""
import datetime as dt
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core import jobs, retrieval
from app.db import repository
from app.kb import chunking, dedup, documents, dualwrite, milvus_client
from app.kb.sources import CONTENT_TYPE_DESC, CONTENT_TYPES, KB_DIR, SOURCE_TYPES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb")

MAX_TEXT_CHARS = 40000
KB_JOBS = ("kb-preview", "kb-build", "kb-mine", "kb-vectorize", "seed-conv", "kb-reset")


def _fingerprint(questions: str, answer: str) -> str:
    return dedup.normalize_question(questions) + "|" + dedup.normalize_question(answer)


def _chunk_view(seq: int, c: documents.Chunk, dup: bool | None) -> dict:
    return {"seq": seq, "section_path": c.section_path or "(无标题层级)", "category": c.category,
            "questions": c.questions, "answer": c.answer, "chars": len(c.answer),
            "is_key_clause": bool(c.is_key_clause), "is_table": chunking.is_table_block(c.answer),
            "duplicate": dup}


def _features(chunks: list[documents.Chunk]) -> dict:
    groups: dict[str, list[documents.Chunk]] = {}
    for c in chunks:
        groups.setdefault(c.section_path, []).append(c)
    table_split = any(len(g) > 1 and chunking.is_table_block(g[0].answer) for g in groups.values())
    overlap = any(len(g) > 1 and not chunking.is_table_block(g[0].answer) for g in groups.values())
    return {"sections": len(groups), "table_split": table_split, "overlap": overlap,
            "multi_piece_sections": [p for p, g in groups.items() if len(g) > 1]}


async def _existing_fingerprints() -> set[str] | None:
    """库里已有的指纹集合;库连不上回 None——预览照样能看,只是查重那列显示未知。"""
    try:
        return {_fingerprint(q, a) for q, a in await repository.list_chunk_pairs()}
    except Exception:
        return None


async def milvus_state() -> dict:
    """Milvus 探活 + 条数。连不上不是错误,是一种状态:页面显示离线,双写一致那一栏不下结论。"""
    try:
        c = milvus_client.get_client()
        milvus_client.ensure_collection(c)
        return {"online": True, "count": milvus_client.count(c), "collection": milvus_client.COLLECTION}
    except Exception as e:
        return {"online": False, "count": None, "detail": f"{type(e).__name__}: {e}"}


def _sources() -> list[dict]:
    out = []
    for fname, ctype in SOURCE_TYPES.items():
        path = KB_DIR / fname
        item = {"file": fname, "content_type": ctype, "present": path.exists(),
                "path": f"data/kb/{fname}"}
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            chunks = documents.build_chunks(raw, content_type=ctype)
            st = path.stat()
            item.update({"chars": len(raw), "lines": raw.count("\n") + 1,
                         "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                         "chunks": len(chunks), "key_clause": sum(c.is_key_clause for c in chunks),
                         "features": _features(chunks)})
        out.append(item)
    return out


@router.get("/overview")
async def overview() -> dict:
    """录入页总览:库存 + 双写一致 + 暂存表 + 材料清单 + 作业状态。"""
    milvus = await milvus_state()
    try:
        stats = await repository.knowledge_stats()
        recent = [{"id": r.id, "questions": r.questions, "answer": r.answer[:120],
                   "category": r.category, "section_path": r.section_path,
                   "content_type": r.content_type, "is_key_clause": bool(r.is_key_clause),
                   "status": r.vectorize_status,
                   "created_at": r.created_at.isoformat(timespec="seconds") if r.created_at else None}
                  for r in await repository.list_recent_chunks(12)]
        staging = await repository.staging_stats()
        db_err = None
    except Exception as e:                     # mysql 没起时不连坐整页
        stats = {"total": None, "pending": None, "done": None, "by_content_type": {}, "key_clause": None}
        recent, staging, db_err = [], None, f"{type(e).__name__}: {e}"

    # 双写一致 = 没有块卡在 pending,且 MySQL 已向量化数与 Milvus 条数对得上。
    # 任一边读不到就是 None(不知道),不当成不一致——「读不到」和「对不上」是两回事。
    consistent = None
    if db_err is None and milvus["online"]:
        consistent = stats["pending"] == 0 and stats["done"] == milvus["count"]

    return {"chunks": stats, "recent": recent, "staging": staging, "db_error": db_err,
            "milvus": milvus, "consistent": consistent, "sources": _sources(),
            "content_types": [{"key": k, "desc": CONTENT_TYPE_DESC[k]} for k in CONTENT_TYPES],
            "jobs": [jobs.status(n) for n in KB_JOBS]}


class PreviewIn(BaseModel):
    text: str | None = None
    file: str | None = None                    # 只能是材料清单里的文件名,不接受路径
    content_type: str = "faq"


def _resolve_preview(body: PreviewIn) -> tuple[str, str, str]:
    """回 (正文, content_type, 来源标注)。file 走白名单精确匹配,传不进任何路径。"""
    if body.file:
        if body.file not in SOURCE_TYPES:
            raise HTTPException(status_code=400, detail=f"不在建库材料清单里:{body.file}")
        path = KB_DIR / body.file
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"材料文件不存在:data/kb/{body.file}")
        return path.read_text(encoding="utf-8"), SOURCE_TYPES[body.file], f"data/kb/{body.file}"
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="正文是空的,贴一段 Markdown 再预览")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"正文 {len(text)} 字,超过单次上限 {MAX_TEXT_CHARS} 字,拆开录")
    if body.content_type not in CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"content_type 只能是 {'/'.join(CONTENT_TYPES)}")
    return text, body.content_type, "手工录入"


@router.post("/preview")
async def preview(body: PreviewIn) -> dict:
    """切块预览(dry-run):不写库、不碰 Milvus、不调上游。"""
    text, ctype, source = _resolve_preview(body)
    chunks = documents.build_chunks(text, content_type=ctype)
    seen = await _existing_fingerprints()
    views, dups, batch = [], 0, set()
    for i, c in enumerate(chunks, 1):
        fp = _fingerprint(c.questions, c.answer)
        dup = None if seen is None else (fp in seen or fp in batch)
        dups += bool(dup)
        batch.add(fp)
        views.append(_chunk_view(i, c, dup))
    return {"source": source, "content_type": ctype, "chars": len(text), "total": len(chunks),
            "duplicates": dups, "key_clause": sum(c.is_key_clause for c in chunks),
            "features": _features(chunks), "chunks": views, "dedup_known": seen is not None}


class IngestIn(BaseModel):
    text: str
    content_type: str = "faq"
    vectorize: bool = True


@router.post("/ingest")
async def ingest(body: IngestIn) -> dict:
    """录入:切块 → 按指纹查重 → 写 MySQL(pending)→ 可选立刻向量化。
    先写 MySQL 再进 Milvus 的顺序不能反:MySQL 是原文权威源,挂了重跑捡 pending 补齐。"""
    text, ctype, _ = _resolve_preview(PreviewIn(text=body.text, content_type=body.content_type))
    chunks = documents.build_chunks(text, content_type=ctype)
    if not chunks:
        raise HTTPException(status_code=400, detail="这段正文切不出块,检查是不是只有标题没有正文")
    seen = await _existing_fingerprints()
    if seen is None:
        raise HTTPException(status_code=503, detail="连不上 MySQL,录入这条路走不通(查重与落库都要它)")

    kept, skipped = [], []
    for c in chunks:
        fp = _fingerprint(c.questions, c.answer)
        if fp in seen:
            skipped.append({"questions": c.questions, "answer": c.answer[:80]})
            continue
        seen.add(fp)
        kept.append(c)

    ids = await dualwrite.write_pending(kept) if kept else []
    vectorized = None
    if body.vectorize and ids:
        try:
            vectorized = await dualwrite.vectorize_pending()
        except Exception as e:      # 向量化失败不回滚:块已在 MySQL 记 pending,补跑一次就齐
            raise HTTPException(
                status_code=502,
                detail=(f"已入库 {len(ids)} 块(pending),向量化失败:{type(e).__name__}: {e}。"
                        "修好嵌入/Milvus 后按「向量化待补块」补齐,不必重录")) from e
    return {"content_type": ctype, "chunks": len(chunks), "inserted": len(ids), "skipped": len(skipped),
            "skipped_samples": skipped[:5], "ids": ids, "vectorized": vectorized,
            "milvus": await milvus_state(), "chunk_stats": await repository.knowledge_stats()}


@router.post("/vectorize")
async def vectorize() -> dict:
    """把库里所有 pending 块补齐向量(幂等可重跑)。与 make kb-vectorize 同一个函数。"""
    try:
        n = await dualwrite.vectorize_pending()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"向量化失败:{type(e).__name__}: {e}") from e
    return {"vectorized": n, "chunk_stats": await repository.knowledge_stats(),
            "milvus": await milvus_state()}


class SearchIn(BaseModel):
    q: str
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/search")
async def search(body: SearchIn) -> dict:
    """检索自测(dense 单路):换个说法问一句,看召回的是不是该召回的那块。"""
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="问一句话再检索")
    try:
        hits = await retrieval.search_knowledge(q, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"检索失败({type(e).__name__}: {e});Milvus 要在线") from e
    return {"q": q, "top_k": body.top_k,
            "hits": [{"id": h.get("id"), "question": h.get("question"), "answer": h.get("answer"),
                      "score": h.get("score")} for h in hits]}


@router.get("/staging")
async def staging(limit: int = 30) -> dict:
    """抽 QA 暂存表:分批抽出 → 整体去重 → kept 待人审。"""
    stats = await repository.staging_stats()
    rows = {}
    for st in ("extracted", "kept", "discarded"):
        rows[st] = [{"id": r.id, "batch_no": r.batch_no, "source_ref": r.source_ref,
                     "question": r.question, "answer": r.answer[:160]}
                    for r in (await repository.list_staging_by_status(st))[:limit]]
    return {"stats": stats, "rows": rows}
```

- [ ] **Step 9: 实现后台聚合 `app/api/admin.py`**

Create `work/ch03/app/api/admin.py`:

```python
"""后台首页聚合:知识库一张卡。当 Milvus/MySQL 没起时,只让它自己这张卡显示读数失败,不连坐整页。"""
from fastapi import APIRouter

from app.api import kb
from app.db import repository

router = APIRouter(prefix="/api/admin")


async def _kb_card() -> dict:
    card = {"key": "kb", "title": "知识库", "page": "/kb",
            "lede": "文档与对话挖出来的知识 → 切块 → 双写 MySQL 与 Milvus",
            "status": "error", "headline": "读数失败", "metrics": [], "note": None}
    try:
        stats = await repository.knowledge_stats()
    except Exception as e:
        card["note"] = f"{type(e).__name__}: {e}"
        return card
    milvus = await kb.milvus_state()
    card["metrics"] = [
        {"label": "知识块", "value": stats["total"]},
        {"label": "待向量化", "value": stats["pending"]},
        {"label": "Milvus", "value": milvus["count"] if milvus["online"] else "离线"},
        {"label": "关键条款", "value": stats["key_clause"]},
    ]
    if not stats["total"]:
        card["status"], card["headline"] = "missing", "库是空的,先录入或跑一次离线建库"
    elif not milvus["online"]:
        card["status"], card["headline"] = "attention", f"MySQL 有 {stats['total']} 块,Milvus 离线"
    elif stats["pending"] or stats["done"] != milvus["count"]:
        card["status"] = "attention"
        card["headline"] = (f"{stats['pending']} 块待向量化,"
                            f"MySQL 已向量化 {stats['done']} 对 Milvus {milvus['count']}")
    else:
        card["status"], card["headline"] = "ok", f"{stats['total']} 块双写一致,可被语义检索"
    return card


@router.get("/overview")
async def overview() -> dict:
    return {"modules": [await _kb_card()]}
```

- [ ] **Step 10: 挂载三个 router 到 main.py**

Edit `work/ch03/app/main.py`(加 import 与 `include_router`):

```python
from app.api.admin import router as admin_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
...
app.include_router(kb_router)
app.include_router(admin_router)
app.include_router(jobs_router)
```

- [ ] **Step 11: 运行看通过**

Run: `uv run pytest tests/test_kb_api.py tests/test_jobs_api.py tests/test_admin_api.py -v`
Expected: 全绿。若 `test_overview_milvus_offline_is_state_not_error` 里 `milvus_state` 的 `ensure_collection` 未被 mock 到而仍尝试连真 Milvus——fixture 已把 `get_client` 打桩为抛错,确保 `milvus_state` 走 except 分支。

- [ ] **Step 12: 提交**

```bash
git add app/core/jobs.py app/api/jobs.py app/api/kb.py app/api/admin.py app/main.py \
        app/db/repository.py tests/conftest.py tests/test_kb_api.py tests/test_jobs_api.py tests/test_admin_api.py
git commit -m "feat(ch03): 知识库录入 API + 白名单作业运行器 + 后台聚合(预览 dry-run、录入按指纹幂等)"
```

---

## Task 18: 知识库录入页 + 后台管理外壳

**非 TDD(Vibe Coding)**:页面按 spec §10 的六块布局,数全来自 Task 17 的接口;验收在浏览器里点,不写前端单测。静态页只加一条「能打开且引了共用外壳」的可达性断言。
> work/ch03 无 ch02 的 `acceptance.*` 资产:本任务的三份静态文件**自包含**(样式内联 + `admin.js` 自带注入),不依赖其它章的资源。

**Files:**
- Create: `work/ch03/app/static/kb.html`、`work/ch03/app/static/admin.html`、`work/ch03/app/static/admin.js`
- Modify: `work/ch03/app/static/index.html`(聊天页顶栏加「后台管理」入口)
- Modify: `work/ch03/app/main.py`(`/kb`、`/admin` 页面路由 + `/static` 挂载)
- Test: `work/ch03/tests/test_static.py`

- [ ] **Step 1: 录入页六块(`kb.html`)**

① 手工录入(内容类型 + 正文 + 预览 / 入库 + 「入库后顺手向量化」)、② 建库材料(清单 + 就地切块 + 逐份「看切块」+ 作业按钮)、③ 对话挖知识(暂存三态 + 作业按钮 + 逐条展开)、④ 向量化与双写(pending/done/Milvus 三数 + 补向量按钮 + 作业按钮)、⑤ 检索自测(问句 + 取 Top-K + 预设问句)、⑥ 最近入库。
预览里逐块标出:节路径、问法、正文、字数、是否关键条款、是否表格块、是否与库里重复;顶部标出三个切块特性(标题切 / 表格按行拆 / 句末重叠)有没有触发。
数据全部取自 Task 17:`GET /api/kb/overview`(库存闸条 / 材料 / 暂存 / 作业状态 / 最近入库)、`POST /api/kb/preview`、`POST /api/kb/ingest`、`POST /api/kb/vectorize`、`POST /api/kb/search`、`GET /api/kb/staging`;作业按钮打 `POST /api/jobs/{name}` 并轮询 `GET /api/jobs/{name}`。库存闸条五个数(知识块 / 待向量化 / Milvus / 关键条款 / 双写一致)直接读 `overview.chunks`、`overview.milvus`、`overview.consistent`。

- [ ] **Step 2: 后台外壳(`admin.js` + `admin.html`)**

`admin.js` 导出 `mountAdminNav(active)`:自带样式注入(挂在样式内联的旧页上也不打架)、包在 IIFE 里(不与宿主页的 `$` / `el` 撞名)、两层导航(模块行 + 当前模块的子页行)。`/admin` 聚合首页每模块一张卡:状态药丸(正常 / 要干活 / 没数据 / 读数失败)+ 一句结论 + 几个关键数 + 进入按钮。**各模块页面保持原路径**,首页只收入口。本任务首页只放「知识库」一张卡(读 `GET /api/admin/overview`)。

- [ ] **Step 3: 作业日志不许在跑完那一刻消失**

作业进终态会回调页面重新取数,取数把按钮与日志窗口整个重建。按作业名缓存日志尾,重建时贴回去——不然最该读的结论正好被刷掉。

- [ ] **Step 4: main.py 加页面路由与静态挂载 + 聊天页加入口**

Edit `work/ch03/app/main.py`(在 Task 17 已挂三个 router 的基础上,加静态挂载与两个页面路由):

```python
from fastapi.staticfiles import StaticFiles   # 顶部 import 区补

# 后台各页共用外壳(样式 + 取数/重跑脚本 + 导航),抽成文件放静态目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/kb", include_in_schema=False)
async def kb_page() -> FileResponse:
    """ch03 知识库录入页:贴文档就能切块入库、向量化、当场检索自测。"""
    return FileResponse(STATIC_DIR / "kb.html")


@app.get("/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    """后台管理首页:知识库一张卡,一处进出;各模块页面保持各自原路径。"""
    return FileResponse(STATIC_DIR / "admin.html")
```

> `STATIC_DIR` 在 ch02 的 `main.py` 里已定义,不要重复定义。
> `kb.html` 与 `admin.html` 的 `<head>` 里都要引 `<script src="/static/admin.js"></script>`(Step 5 的可达性断言就在查这一条)。

Edit `work/ch03/app/static/index.html` 顶栏,`<header>` 里「+ 新对话」旁加一个入口:

```html
<a href="/admin" style="color:#fff;text-decoration:none;margin-left:8px;">后台管理</a>
```

- [ ] **Step 5: 写可达性测试**

Create `work/ch03/tests/test_static.py`:

```python
async def test_admin_pages_reachable_and_share_nav(client):
    for path in ("/kb", "/admin"):
        r = await client.get(path)
        assert r.status_code == 200
        assert "/static/admin.js" in r.text
```

- [ ] **Step 6: 运行测试 + 浏览器实测**

Run: `uv run pytest tests/test_static.py -q`
Expected: PASS。
浏览器:`make dev` 后开 `http://localhost:8000/kb`,走一遍①贴正文→预览→入库、⑤检索自测;开 `/admin` 看知识库卡片出得来。

- [ ] **Step 7: 提交**

```bash
git add app/static/kb.html app/static/admin.html app/static/admin.js app/static/index.html app/main.py tests/test_static.py
git commit -m "feat(ch03): 知识库录入页 + 后台管理聚合首页与共用导航"
```

---

## Task 19: 真实建库全链路 + 三项验收(全在浏览器里)

**集成验收**(主会话,真实服务)。起 `make dev` 之后一律在浏览器里走:建库、补向量、检索自测、语义召回,截图留档 `dev-notes/generated-images/`(被 gitignore,不提交)。

**Files:** 无新增(必要时 `work/ch03/README.md` 补 ch03 启动/验收段)。

- [ ] **Step 1: 起服务 + 在页面上建库**

先确认 Milvus 与 MySQL 起好(`make milvus-up`;mysql 由 `docker compose up -d mysql` 起,注意先停 ch02 compose 避免同名冲突)。
`make dev`(应用;MySQL 复用 mewhelp-mysql),开 `/kb`:
1. ② 建库材料按「材料清单与切块预览」→ 日志窗口里看到各文件块数;页面表格里的块数与日志里的总块数对得上。
2. 按「离线建库」→ chunks 落 pending;顶部闸条 `待向量化` 有数。
3. ④ 区按「向量化待补块」→ pending 归零、`done` 与 `Milvus 条数` 相等、`双写` 显示「一致」。

Expected: 三步都在页面上完成,不回终端;库存闸条五个数自洽。

- [ ] **Step 2: 验收1(语义召回)—— 页面自测 + 聊天页**

1. `/kb` ⑤ 检索自测,点预设「邮费是多少」:Top-1 落在运费/包邮那块,分数与其余候选拉开;截图。
2. 聊天页 `/` 输入「邮费是多少」:出现「🔧 调用了 query_faq」徽章 + 答出满 99 包邮 / 运费规则;截图。
3. `make eval-retrieval` 作为回归线(≥4/5 通过)。

Expected: 换说法能召回,页面与聊天页两处一致。

- [ ] **Step 3: 验收2(中断重跑)—— 页面上演一遍**

1. ④ 区按「清库重建」(heavy,二次确认)→ 两表清空、集合 drop。
2. 按「离线建库」→ 只有 pending,`双写` 显示「对不上」。
3. 按「向量化待补块」中途停掉(或临时把 Milvus 停掉让它失败)→ 闸条显示 done 与 pending 并存。
4. 修好后再按一次 → 只捡剩余 pending 补齐,全部 done,`Milvus 条数 == 知识块数`,无重无漏。

Expected: 幂等恢复的实证不用回终端看 SQL——闸条那三个数就是账。
(该性质已被 Task 10 的 `test_vectorize_resumes_after_crash` 单测证明;此处做一次真实链路演示。)

- [ ] **Step 4: 验收3(手工录入)—— 贴一段进去**

1. ① 区「填一段示例」→ 内容类型选 `policy` → 「切块预览」:两节两块,关键条款标出来,表格块标出来,查重显示「无重复,可入库」。
2. 「录入入库」(勾了顺手向量化)→ toast 报入库块数与向量化块数;闸条总数增加、pending 仍为 0。
3. 再按一次「录入入库」→ 预览区每块标「重复」,入库 0 跳过全部——录入天然幂等。
4. ⑤ 检索自测再问一次「邮费是多少」:刚录进去的那块出现在 Top-K 里(带它的 id)。

Expected: 四步都在浏览器里,截图留档。

- [ ] **Step 5: 挖知识全链路(需聊天上游)**

`/kb` ③ 区:「灌历史会话种子」→「对话挖知识」(heavy,二次确认)→ 暂存三态出现落差(抽出 / 保留 / 丢弃),逐条能对照;再按 ④ 区补向量。
`make eval-mining` 作为抽取质量回归线。

Expected: 去重那一刀在页面上看得见;新挖的块向量化后可被检索到。

- [ ] **Step 6: 后台首页与全量回归**

`/admin` 上知识库那张卡出得来:关掉 Milvus 只让它说「Milvus 离线」,不连坐整页。

```bash
uv run pytest -q
git add work/ch03/README.md
git commit -m "test(ch03): 建库全链在浏览器里跑通,三项验收(语义召回 / 中断重跑 / 手工录入幂等)通过"
```

---

## Self-Review(计划对照 spec)

**1. Spec coverage:**
- 功能1 文档处理 → Task 6/7/8/9(标题切 / 递归切 / 句末重叠 / 表格行切复表头 + 字段映射)✓
- 功能2 对话挖知识 job → Task 12(mining + CLI,staging→去重→pending)+ Task 16(seed + eval)✓
- 功能3 落库结构(三格拼向量 + 四类元数据)→ Task 4/5(ORM+CRUD 全字段)、Task 9(字段映射)、Task 10(向量化文本 = category+questions+answer)✓
- 功能4 双写幂等 → Task 10(write_pending + vectorize_pending 按 id upsert,恢复单测)✓
- 功能5 在线检索替换 → Task 13(search_knowledge)+ Task 14(query_faq 契约不变)✓
- 功能6 录入页 → Task 17(录入 API + 作业运行器,TDD)+ Task 18(六块页面 + 后台外壳)✓
- 基础设施(**进程内 BGE-M3 / Milvus Standalone / env / deps**)→ Task 1 ✓
- 验收1 → Task 16 eval + Task 19 Step 2;验收2 → Task 10 单测 + Task 19 Step 3;验收3(建库全链在浏览器里)→ Task 19 Step 1/4/5 ✓
- spec §10 不变量各有归属:切块一份实现(Task 15 sources.py + Task 17 复用 build_chunks)、双写顺序不许反与失败不回滚(Task 17 Step 8)、指纹查重不误杀同节多块(Task 17 Step 1 用例)、读不到 ≠ 对不上(Task 17 Step 1 用例)、没有 shell(Task 17 Step 2 用例)✓
- 三处定稿差异全部落实:① 进程内 BGE-M3(Task 1/2)、② Milvus Standalone Docker(Task 1/3/10/13)、③ 代码落 `work/ch03/`(全任务路径)✓

**2. Placeholder scan:** 无 TBD/TODO/"similar to Task N";每个代码步给出完整测试 + 实现;夹具给出具体 markdown/JSON/SQL。`retrieval_min_score=0.4` 为初值,Task 19 按 eval 结果微调(已注明)。

**3. Type consistency:** `Chunk`(documents.py)贯穿 documents/dualwrite/mining;`search_knowledge` / `milvus_client.search` 返回 `{id,score,question,answer}` 在 retrieval/query_faq/eval 一致;repository 方法名(`insert_knowledge_chunk`/`list_pending_chunks`/`mark_chunk_vectorized`/`set_chunk_neighbors`/`count_chunks_by_status`/`list_all_questions`/`insert_staging`/`list_staging_by_status`/`set_staging_status`/`list_conversations_with_messages`/`count_chunks_by_content_types`/`knowledge_stats`/`list_recent_chunks`/`list_chunk_pairs`/`staging_stats`)跨任务一致;`jobs.{JOBS,start,stop,status,status_all,tail}` 在 jobs.py/api/jobs.py/测试一致。

**4. Review Focus wiring:** 五条各落到拥有该代码的任务——① 空文档 → Task 9 `test_empty_and_heading_only_yield_no_chunks` + Task 17 `test_ingest_empty_document_rejected`;② 超长块 → Task 3 `test_upsert_truncates_overlong_text`;③ Milvus 离线 → Task 17 `test_overview_milvus_offline_is_state_not_error`;④ 重复录入幂等 → Task 17 `test_ingest_then_reingest_is_idempotent`;⑤ 空表格 → Task 8 `test_table_with_no_body_rows_stays_whole`。✓

**已知需实现时留意:** Milvus Standalone `search`/`count` 返回结构以 Task 3 真集合跑通为准修正;`upsert` 后若紧接 `search` 读不到,靠 `flush`(Task 3/10 已内置);`test_tool_faq.py`(ch02 查表断言)在 Task 14 删除,不保留查表断言;`upsert` 的 `search` 用 `anns_field="vector"`,若某版本 Standalone 单向量字段可省,以实测为准。
