# MewHelp Ch03 设计:知识库向量语义检索(RAG 基础)

> 重走版,2026-10-15。仓库已有 ch03 成品,本文件是**重新走一遍 ch03 时新写的 spec**,
> 不是改写成品的补丁。成品代码不动、留作参考答案;本章代码落在独立目录 `work/ch03/`(自 ch02 拷贝起步)。
> 范本:`docs/superpowers/specs/2026-07-14-ch03-rag-knowledge-base-design.md`。

## 1. 背景与目标

ch02 的 `query_faq` 用**关键词查表**(`SELECT * FROM faq WHERE question LIKE '%kw%'`),对换说法的问题会**漏召回**——典型反例「邮费是多少」:答案在库里「运费怎么算」那条,但字面 `LIKE` 对不上,答不出来。

**本章目标**:给客服系统建知识库,把 `query_faq` 的**内部实现**从关键词查表升级成**向量语义检索**;工具的**入参出参契约保持不变**(签名 `keyword: str`,出参 `{"hits": [{"question", "answer"}]}`)。因为 `query_faq` 已长在前端在用的 `/api/chat` 工具链上,升级后**无需改前端**,验收在浏览器直接跑。

**一句话架构**:MySQL 当原文权威源(+ 双写状态),向量落 **Milvus Standalone**,嵌入用 **进程内加载的 BGE-M3**,在线检索 **dense 单路 Top-K**。

## 2. 范围

**做:**
1. 离线建库·文档处理:Markdown 按标题层级结构感知切分;超长递归切;块间重叠裁到最近句号;大表格按行切且每块复制表头。
2. 离线建库·对话挖知识:一条命令可启动、可重跑的可重跑批处理 job,分批喂 LLM 抽问答对,先进暂存表、整体去重再入库。
3. 落库结构:每条知识 `category + questions + answer` 三格拼文本做向量化;四类元数据(章节路径 / 内容类型 / 是否关键条款 / 前后块指针)只存不进向量。
4. 双写落库:MySQL `knowledge_chunks` 权威源 + Milvus `knowledge` 集合;先写 MySQL 记 pending,再写 Milvus 回填 `vector_id`、转 done;按主键幂等、挂了能重跑。
5. 在线检索:问题向量化 → Milvus Top-K → 替换 `query_faq` 的关键词查表实现。
6. 知识库录入页 `/kb`:建库这条链搬上浏览器——材料清单与切块预览、手工贴一段文档就能入库、向量化补 pending、检索自测;离线那几个 make 目标也在页面上按。详见 §10。

**不做(用户明确):**
- 关键词召回、混合检索、重排——**只跑 dense 向量单路**。
- 录入页不做登录鉴权、操作审计、多人协同(本地单人开发用的后台)。
- 录入页不做知识的改与删:本章只解决「灌进来」;改错了走清库重建。
- 不引入 **LangGraph / Langfuse**:挖知识 job 用普通 async + LangChain 文本切分器足够。
- 真实 cron / APScheduler 调度=部署关注点,不在本章;挖知识做成命令可启动、可重跑的 job。

## 3. 架构总览与数据流

两条离线链路把知识灌进库,一条在线链路服务检索:

```
离线·文档   data/kb/*.md ──chunking──→ chunk 记录 ──┐
离线·对话   conversations/messages ──LLM 抽取──→ qa_extraction_staging ──去重(kept)──┤
                                                                                     │
                                                     ↓ 都写入 MySQL knowledge_chunks(vectorize_status=pending)
                                              vectorize job(幂等可重跑):
                                     取 pending → BGE-M3 embed(dim 1024)→ Milvus upsert(PK=id, 带 question/answer)
                                              → 回填 vector_id、status=done

在线   query_faq(keyword) → embed(keyword) → Milvus search Top-K(output_fields=question,answer)
                          → {"hits": [{"question","answer"}]}(低于阈值算未命中,保留空 hits 分支)
```

## 4. 基础设施(本章定稿,与教程范本不同的三处)

| 项 | 内容 | 与教程范本的差异 |
|---|---|---|
| **嵌入** | **进程内加载 BGE-M3**(`FlagEmbedding` 的 `BGEM3FlagModel`),应用内本地推理产 dense 向量(dim 1024)。**不走网络、不需要 key**。 | 范本用 SiliconFlow HTTP(`EMBED_API_KEY`);我们改成本地进程内。 |
| **向量库** | **Milvus Standalone**(Docker:etcd + minio + milvus,端口 **19530**),`MilvusClient(uri="http://localhost:19530")`。 | 范本用 Milvus **Lite**(嵌入式,零容器);我们改用 Standalone(**因 Lite 官方不支持 Windows**;仍是 Milvus,不换选型)。 |
| **存储** | MySQL(Docker,库 `mewhelp`,**端口 3307**,沿用 ch02)。 | — |
| **依赖** | `FlagEmbedding`(含 `torch`)、`pymilvus`、`langchain-text-splitters`(LangChain 已在)。 | — |
| **配置项** | `app/config.py` 加 `embed_model="BAAI/bge-m3"`、`embed_device="cpu"`、`milvus_uri="http://localhost:19530"`、`retrieval_top_k`、`retrieval_min_score`。**无 `embed_api_key`**。 | — |
| **HF 镜像** | 首次加载 BGE-M3 需从 HuggingFace 下 ~2GB;国内慢/被墙时用环境变量 `HF_ENDPOINT=https://hf-mirror.com`(写进 `.env.example` 注释,不硬编码)。 | — |

> **docker-compose**:`work/ch03/docker-compose.yml` 定义全套(mysql + etcd + minio + milvus-standalone)。**启动前先停 ch02 的 compose**(两者都定义容器名 `mewhelp-mysql`,同名冲突);mysql 卷 `mewhelp-mysql-data` 复用,数据不丢。
>
> 具体 `FlagEmbedding` / `pymilvus` / `MarkdownHeaderTextSplitter` 的确切 API,在 **writing-plans 阶段先用 Context7 核对最新官方文档**再落计划(沿用 ch02「写计划前 Context7 核对」的做法),本 spec 只定设计意图。

## 5. 数据模型

### 5.1 MySQL(权威源,DDL 见 `sql/ch03-ddl.sql`)

- **`knowledge_chunks`**:知识 chunk 原文权威源。`id`(与 Milvus 主键对齐)、`category / questions / answer`(拼成向量化文本)、`section_path / content_type / is_key_clause / prev_chunk_id / next_chunk_id`(元数据,只存不进向量)、`vector_id`(Milvus 主键回填)、`vectorize_status ∈ {pending, done}`(双写幂等靠它)。
- **`qa_extraction_staging`**:对话挖知识离线中转。`batch_no`(分批,防串味 + 按批追溯)、`source_ref`(来源溯源)、`question / answer`(LLM 抽出)、`status ∈ {extracted, kept, discarded}`(已抽待去重 / 去重保留 / 去重丢弃)。保留(kept)项入 `knowledge_chunks`,建库完成可清空。

### 5.2 Milvus 集合 `knowledge`(在线检索服务读模型)

- `id`:INT64 **主键 = MySQL 的 `knowledge_chunks.id`**(不用 auto_id),使重跑按 id `upsert` 天然幂等。
- `vector`:FLOAT_VECTOR,**dim 1024**(BGE-M3 dense;冒烟时读 `len(embedding)` 实测确认)。
- `question` / `answer`:VARCHAR,**检索时直接从 Milvus 取回**(在线不回 MySQL,一次取回更简单)。
- metric **COSINE**,索引 AUTOINDEX。

> **权威源 vs 读模型**:MySQL `knowledge_chunks` 持有全字段(含元数据)且是重建/重向量化的来源;Milvus 是派生的服务读模型,只放服务在线检索够用的 `{id, vector, question, answer}`。两者按 `id` 对齐。

## 6. 离线建库 · 文档处理

`app/kb/chunking.py`(纯确定性逻辑 → **保留 TDD**)+ `app/kb/documents.py`(源文档 → chunk 记录):

- **结构感知切分**:按 Markdown 标题层级切(LangChain `MarkdownHeaderTextSplitter`),保留标题路径。
- **超长递归切**:段内超长用 `RecursiveCharacterTextSplitter` 递归细切。
- **重叠裁到句号**:块间加重叠,重叠边界裁到最近句末标点(`。!?…` 及换行),不留半截话。
- **大表格按行切 + 复制表头**:识别 Markdown 表格,超长时按行分组,每组重贴表头行 + 分隔行。
- **字段映射**:
  - 政策 / 手册(无天然问题):`questions` = 本节标题,`category` = 上级标题路径,`content_type ∈ {policy, manual}`。
  - 商品 FAQ / 挖出的问答对:`questions` = 真实问法,`content_type ∈ {faq, mined}`。
  - `is_key_clause`:源文档用约定标注(如标题打标记),缺省 0。
  - `prev/next_chunk_id`:同一文档按阅读序,入库拿到 id 后二次 pass 连指针。

## 7. 离线建库 · 对话挖知识

`app/kb/mining.py` + `scripts/mine_knowledge.py`(**命令可启动、可重跑的 job**):

1. 读 `conversations / messages` 历史对话,**分批**(`batch_no`,几十个会话一批)。
2. 每批喂 LLM(经聊天上游)抽问答对,**结构化输出** → 写 `qa_extraction_staging`(status=extracted)。← **纯 Prompt,用标注样例 eval 代替 TDD**(work-req 1)。
3. **整体去重**(`app/kb/dedup.py`):按**问法归一化文本**(小写 / 去标点 / trim)在 staging 内 + 对已有 `knowledge_chunks` 去重;保留标 kept、重复标 discarded。← 用户定:dense 只留给检索,去重走文本,更简且确定。
4. kept 项入 `knowledge_chunks`(status=pending,`content_type=mined`,`questions`=真实问法)。

## 8. 双写落库

`app/kb/dualwrite.py`(可单测含崩溃恢复 → **保留 TDD**),拆两 phase:

- **写 MySQL(pending)**:chunk / 去重后的挖知识 → 插 `knowledge_chunks`,`vectorize_status=pending`。
- **vectorize job**(`scripts/vectorize_kb.py`,**幂等可重跑**):
  `SELECT id, category, questions, answer FROM knowledge_chunks WHERE vectorize_status='pending'`
  → 批量 embed(`category+questions+answer` 拼文本)
  → Milvus `upsert([{id, vector, question, answer}])`
  → `UPDATE knowledge_chunks SET vector_id=id, vectorize_status='done' WHERE id=?`。
- **幂等 / 恢复**:崩在任意批 → 重跑只捡 `status='pending'` → 按 id `upsert`(重写已 done 的同 id 无副作用)→ 直到全 done。满足验收 2。

## 9. 在线检索(`query_faq` 契约保持)

- `app/core/embeddings.py`:`embed_texts(texts) -> list[list[float]]`,**进程内 BGE-M3** 产 dense 向量。
- `app/core/retrieval.py`:`search_knowledge(query, top_k) -> list[hit]`:embed(query) → Milvus `search(output_fields=["question","answer"])` → 带 score;**score 低于 `retrieval_min_score` 算未命中**(保留契约的空 hits 分支)。
- `app/tools/business.py` 的 `query_faq`:**签名 `keyword: str`、出参 `{"hits": [{"question","answer"}]}` / 未命中 `{"hits": [], "message": ...}` 一字不改**,内部由 `repository.search_faq` 改调 `retrieval.search_knowledge(keyword, top_k)`。`faq` 表与 `repository.search_faq` 保留不删(ch02 遗留),但 `query_faq` 不再读它。

## 10. 知识库录入页(浏览器里建库)

建库这条链本身是离线的,但**不该只能在终端敲**:知识长什么样、切成几块、有没有块卡在 pending、换个说法能不能召回,都是要反复看的东西。落成一页后台:`/kb`。

### 10.1 页面上的六块

| 区块 | 看什么 / 按什么 |
|---|---|
| 库存闸条 | 知识块数、待向量化、Milvus 条数、关键条款数、双写是否一致 |
| ① 手工录入 | 贴一段 Markdown → 切块预览(dry-run)→ 录入入库(可勾选顺手向量化) |
| ② 建库材料 | `data/kb/*.md` 清单 + 就地切块结果(各切多少块、关键条款几块、触发了哪些切块特性),逐份可送预览区细看;作业按钮 kb-preview / kb-build |
| ③ 对话挖知识 | 暂存表三态计数(抽出 / 保留 / 丢弃)+ 批次数,可展开逐条看;作业按钮 seed-conv / kb-mine |
| ④ 向量化与双写 | pending / done / Milvus 三个数摆一起 + 「向量化待补块」;作业按钮 kb-vectorize / kb-reset |
| ⑤ 检索自测 | 输一句话、取 Top-K,当场看召回哪几块、各自多少分 |
| ⑥ 最近入库 | 按 id 倒序看最近入库的块与它们的状态 |

### 10.2 API 与不变量

- API:`app/api/kb.py`,前缀 `/api/kb`。`overview` 一次给齐库存 / 双写 / 暂存 / 材料清单 / 作业状态;`preview` 是 dry-run;`ingest` 落库;`vectorize` 补 pending;`search` 检索自测;`staging` 看暂存表逐条。
- **切块只有一份实现**:页面预览与录入都调 `app/kb/documents.build_chunks`,离线 CLI 调的也是它;建库材料清单(哪些文件、各算什么 `content_type`)由 `app/kb/sources.py` 一处定义,CLI 与页面同读。
- **双写顺序不许反**:`ingest` 先写 MySQL 记 pending,再走 `dualwrite.vectorize_pending`。向量化失败回 502 并明说「已入库 N 块(pending),补跑一次即可」——**不回滚**。
- **查重按「问法 + 正文」的指纹**:只按问法会误杀同一节切出的多块。同一份正文重复录入则指纹全撞、全跳过,录入天然幂等。
- **依赖缺失是一种状态,不是错误**:Milvus 离线时「双写是否一致」显示「读不到」而不是「对不上」;mysql 没起也只让库存那几个数变空,材料清单与作业按钮照样能用。

### 10.3 就地重跑与安全边界

- 作业运行器 `app/core/jobs.py` + `app/api/jobs.py`(前缀 `/api/jobs`):只能跑注册表里的 make 目标,argv 写死在模块里,前端只传一个作业名——页面有重跑按钮,但**没有 shell**。
- 命令一律走 make:日志落 `log/acceptance/<job>.log`,页面轮询回显尾部。
- `kb-mine`(要调 LLM)与 `kb-reset`(清空知识库)标 heavy,页面二次确认才发起;同名作业没跑完拒绝重入。
- **录入这类写操作走 API 而不是作业**:它没有对应的 make 目标——正文是浏览器里现给的。

### 10.4 后台管理外壳

`/kb` 不会是后台里唯一一块,所以外壳一次搭好:一份共用导航 `app/static/admin.js` + 一个聚合首页 `/admin`,每块一张卡给几个关键数与一句结论。某块的依赖没起只让它自己那张卡显示读数失败,不连坐整页。

## 11. 夹具 / 种子数据(已在仓库)

- **源文档 `data/kb/*.md`**(仓库已有,6 份):含①运费 / 邮费说明(验收 1 的召回目标)②大表格 ③长段落 ④政策类无天然问题的章节。ch02 那 6 条 faq 内容并入这些文档。
- **合成历史对话**:`sql/ch03-seed.sql`(seed-u1 / seed-u2 等)喂挖知识 job。

## 12. 测试与验证策略(work-req 1)

| 部分 | 性质 | 验证方式 |
|---|---|---|
| chunking(标题切 / 递归切 / 句末重叠 / 表格行切复制表头) | 确定性代码 | **TDD** 单测 |
| 去重(文本归一化) | 确定性代码 | **TDD** 单测 |
| 双写幂等 / 崩溃恢复 | 确定性代码 | **TDD** 单测(注入中途失败 → 断言剩 pending → 重跑 → 全 done) |
| 在线检索回灌 / 阈值未命中 | 确定性代码 | **TDD** 单测(fake embed + 本地 Milvus) |
| 挖知识抽取 Prompt | 纯 Prompt | **标注样例 eval**(`tests/data/mining_samples.json`) |
| 语义召回质量 | 端到端 | **eval 集**(`tests/data/retrieval_samples.json`,含「邮费是多少」→ 运费块)+ 浏览器 demo |
| 录入 API / 作业运行器安全边界 / 后台首页 | 确定性代码 | **TDD** 单测 |

**嵌入 / Milvus 走真实服务的部分**:单测用 fake embed + 真实 Milvus Standalone(或 fake);真实嵌入链路在冒烟任务与 eval / demo 里验。

## 13. 验收标准映射

1. **「邮费是多少」换说法召回运费说明并答对** → §9 在线检索 + §11 源文档含运费块 + `retrieval_samples.json` eval + 浏览器实证(聊天页 `/` 问一句、录入页 `/kb` ⑤ 检索自测看 Top-K)。
2. **故意中断建库任务再重跑,漏向量化的块被捡起补齐** → §8 vectorize job 幂等恢复 + TDD 崩溃恢复单测 + 页面实证(`/kb` ④ pending 有数 → 按「向量化待补块」→ pending 归零)。
3. **建库全链在浏览器里走得通** → §10:`kb-preview / kb-build / kb-mine / kb-vectorize` 在页面上按得动。

## 14. 决策记录

| 决策 | 选择 | 理由/备选 |
|---|---|---|
| 代码落点 | `work/ch03/`(自 ch02 拷贝) | 用户确认;成品不动 |
| **嵌入** | **进程内 FlagEmbedding 加载 BGE-M3** | 用户选「本地/自建跑」(非 SiliconFlow);省 key、无网络依赖;代价=venv 装 torch + 下 ~2GB 模型 |
| **向量库** | **Milvus Standalone(Docker @19530)** | Milvus Lite 官方**不支持 Windows**;Standalone 仍是 Milvus、Windows 最稳(教程 ch04 也是这个形态) |
| MySQL | 沿用 ch02 容器(3307),ch03 compose 一并管 | 同名容器冲突,ch03 启动前停 ch02 compose;卷复用 |
| 范围 | **含 `/kb` 录入页** + 后台外壳 + 作业运行器 | 用户确认一并做 |
| 在线取数 | Milvus 直接存 `question/answer`、在线从 Milvus 取回 | 一次取回更简单;MySQL 仍是权威源 |
| 去重 | 文本归一化(不引 embedding 相似度) | dense 本章只服务检索 |
| id | = Milvus 主键,重跑按 id upsert 幂等 | 支撑验收 2 |
| 不做 | 关键词/混合/重排、LangGraph、Langfuse、鉴权、知识改删 | 用户明确 |
| 查重指纹 | 取「问法 + 正文」而非只取问法 | 同一节切出的多块共用节标题,只按问法会误杀 |
| 材料清单 | 收进 `app/kb/sources.py` 一处 | CLI 与页面同读,免得「预览三份、入库四份」 |

## 15. 风险与 fail-fast 闸

- **最大风险**:本地 BGE-M3 能否装上、跑通、dim 是否 1024。**第一个 fail-fast 冒烟任务**(照 ch02 Task 1):装 `FlagEmbedding` → 加载 BGE-M3 → embed 一句 → 断言 `len(embedding)==1024`。跑不通则停下问用户,只在「本地 BGE-M3」方案内换法(**不擅自换掉选型**)。附属风险:HF 下载被墙 → `HF_ENDPOINT=https://hf-mirror.com`。
- **Milvus Standalone 资源**:etcd+minio+milvus 三容器,内存占用不小;起不来/端口占用时停下问用户。
- **前置依赖**:首次跑冒烟 / eval / 建库前,`FlagEmbedding` 与 BGE-M3 模型须就绪。

## 16. 文件布局(新增为主,不碰 ch01/ch02 受保护文件)

```
work/ch03/
  app/core/embeddings.py        # 进程内 BGE-M3:embed_texts()
  app/core/retrieval.py         # search_knowledge:embed → Milvus search → hits
  app/kb/{__init__,chunking,documents,mining,dedup,dualwrite,milvus_client,sources}.py
  app/api/{kb,jobs,admin}.py    # 录入页 / 作业 / 后台首页 API
  app/core/jobs.py              # 作业运行器(白名单 make 目标 + 日志落盘)
  app/static/{kb.html,admin.html,admin.js}
  app/db/{models,repository}.py # +KnowledgeChunk +QaExtractionStaging ORM / +知识·暂存 CRUD·盘点
  app/main.py                   # +/kb、/admin 路由 + 静态挂载
  scripts/{build_kb,mine_knowledge,vectorize_kb,eval_retrieval,eval_mining}.py
  docker-compose.yml            # mysql + etcd + minio + milvus-standalone
  tests/...                     # 见 §12
  tests/data/{retrieval_samples,mining_samples}.json
sql/ch03-ddl.sql / ch03-seed.sql   # 仓库根,沿用
data/kb/*.md                       # 仓库根,沿用(夹具)
```

## 17. 落点与隔离

- 成品 `app/ tests/ scripts/` 一律不动,当参考答案。
- 本章代码自包含在 `work/ch03/`,**自 `work/ch02/` 拷贝起步**(app/tests/scripts/pyproject.toml/.env 复制过来再扩展)。
- 过程留痕:`dev-notes/ch03.md`(`.gitignore` 已排除,不进 git)。
