# Ch04 设计:混合检索 + 重排 + 评估体系

状态:定稿(2026-10-19 brainstorm 通过)。
前置:ch03 已交付 dense 单路语义检索(`query_faq` 内部走 Milvus Standalone dense Top-K,契约 `keyword: str` → `{hits:[{question,answer}]}`)。
分支:`ch04-hybrid-retrieval-rerank-eval`(自 `ch03-rag-knowledge-base` 切出)· 代码落点:`work/ch04/`(自 ch03 拷贝)。

## §0 目标与不做

**做**:混合检索(Milvus 原生 BM25 + dense,`hybrid_search` + RRF)、重排(bge-reranker-v2-m3,本地进程内)、元数据过滤、Query 理解(改写 + 检索侧同义词扩展)、生成质量控制(可点引用 / 两道拒答闸 / 自评落池 / 负面知识)、评估体系(100 题五桶、四策略对比、分桶报告、编造个案台账)、聊天页引用角标与 👍/👎 反馈(Vibe Coding)。

**不做**(本章边界):指代消解、多轮改写;LangGraph / Langfuse。后台 RAG 评估页原不在范围,Task 20/21 按用户要求追加实现(只读报告 + 编造个案台账 + 重跑作业)。

**固定选型(定死,不自换)**:Milvus 2.5+ 原生 BM25 全文检索 + `hybrid_search` RRF;重排模型 `bge-reranker-v2-m3`。嵌入沿用 ch03 的 `BAAI/bge-m3`。

## §1 与教程范本的差异(环境事实)

| # | 教程 | 本章 | 原因 |
|---|---|---|---|
| 1 | 嵌入走 SiliconFlow HTTP | **本地进程内 BGEM3FlagModel** | ch03 已定,不依赖外部 key |
| 2 | 重排走上游 `/rerank` | **本地进程内 FlagReranker** | 上游 DeepSeek 无 `/rerank`(实测 404,`/embeddings` 也 404,`/models` 只有 2 个聊天模型);用户拍板改本地 |
| 3 | Milvus Lite → Standalone | **已是 Standalone** | ch03 已按 Standalone 部署 |
| 4 | 两闸都写低置信度池 | **只有自评闸(`self_check`)写池**;机械闸只拒答 | 用户明确「本章只靠 useful 自评判入池」 |
| 5 | 有 `/rag-eval` 后台页 | **Task 20/21 追加实现**(用户要求;只读报告 + 编造个案台账 + 重跑作业) | 原需求未包含,终审后追加 |
| 6 | 评估集 4 桶 80 题 | **100 题**:A/B/C/E 各 20(可作答)+ D 库外 20 | 用户拍板,保证拒答率有独立样本 |
| 7 | 裁判 glm-5.2 | 裁判用**现有 DeepSeek 上游** | 本项目唯一可用模型 |

## §2 架构总览

### 2.1 Milvus 集合 schema(重建 `knowledge`)

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64 主键(auto_id=False) | = MySQL `knowledge_chunks.id`,upsert 幂等 |
| `dense` | FLOAT_VECTOR(dim=1024, COSINE) | BGE-M3 稠密向量 |
| `text` | VARCHAR(enable_analyzer, `analyzer_params={"type":"chinese"}`) | BM25 输入,内容 = `category\nquestions\nanswer`(与 dense 同源) |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 自动生成 |
| `question` / `answer` | VARCHAR | 检索一次取回 |
| `section_path` / `content_type` / `category` | VARCHAR | 引用溯源 + 元数据过滤 |

- BM25 Function:`Function(name="bm25_fn", input_field_names=["text"], output_field_names="sparse", function_type=FunctionType.BM25)`,`create_collection` 前 `schema.add_function`。
- 索引:`dense` → AUTOINDEX/COSINE;`sparse` → SPARSE_INVERTED_INDEX/BM25(k1=1.2,b=0.75)。
- pymilvus 实测版本 **3.0.2**;文档以 Context7 查到的当前 API 为准,冒烟任务先验证。

### 2.2 在线主链路

保持 ch01–03 的单一聊天入口与 tool-calling 编排;事务型工具原样保留。`query_faq` 内部升级为完整 RAG 管线:

```
/api/chat → agent(bind_tools) → 命中 query_faq
  query_faq 内部:
    ① Query 理解   口语→标准问法改写 + 同义词扩展(仅检索侧)
    ② 元数据过滤   category 等字段先过滤(Milvus filter)
    ③ 混合召回     dense Top-50 + BM25 Top-50 → hybrid_search(RRFRanker(k=60))
    ④ 重排         bge-reranker-v2-m3 精排 Top-10
    ⑤ 机械闸       最高 rerank 分 < rerank_min_score(初值 0.3,评估后微调)或 无命中 → 证据不足(拒答,不落池)
    ⑥ 自评闸       结构化自评 useful? false → 证据不足 + 落池(self_check)
    ⑦ 首尾组装     按 rerank 分降序,第 1 条放首、第 2 条放尾,其余按序居中(防 lost-in-the-middle)
    ⑧ 返回         编号证据 + citations 元数据(暂存 ToolRun)
  → agent 收敛:证据足 → 带 [n] 引用作答;证据不足 → 显式拒答
  → SSE 推 citations 事件(编号 → section_path / 原文)
```

## §3 检索管线组件(确定性代码,TDD)

| 组件 | 职责 | 测试要点 |
|---|---|---|
| `milvus_client`(扩展) | 新 schema + BM25 Function;`hybrid_search`;`dense_search` / `bm25_search` 单路(评估用);filter 参数 | 真实临时集合:两路各召回、RRF 有序、filter 生效、BM25 命中型号词 |
| `rerank`(新) | 本地 `FlagReranker`,`compute_score(pairs, normalize=True)`,`asyncio.to_thread` 包裹,截断 Top-10 | mock 模型:重排正确、Top-10 截断、空输入安全 |
| `retrieval.search_knowledge`(重写) | 编排 ①②③④⑤;`strategy` 切换四策略;首尾组装 | mock 子步:管线顺序、首尾排列、strategy 分支 |
| `query_understanding`(新,Prompt) | 口语→标准问法 + 同义词(检索侧) | 纯 Prompt → 标注样例验证 |
| `citations`(新) | 编号→chunk 映射、ToolRun 暂存、SSE 帧 | TDD:编号连续、映射完整、无证据时不产帧 |

**`strategy` 入参**:`vector` / `bm25` / `hybrid` / `hybrid_rerank`。在线默认 `hybrid_rerank`;评估复用同一函数,保证「评估的就是线上的」。

## §4 生成质量控制

- **引用**:生成 System Prompt 要求句/段末标 `[n]`;`query_faq` 返回编号证据并在 `ToolRun.citations` 暂存 `{n, chunk_id, section_path, question, answer}`;`/api/chat` 在工具帧之后、正文 delta 之前推 `{"event":"citations","items":[...]}`。
- **自评闸**:重排后、组装前,一次结构化调用,输出**扁平字段** `useful: bool` + `reason: str`(不用嵌套对象;沿用 ch03 的 `method="function_calling"` + `extra_body={"thinking":{"type":"disabled"}}` 方案);`useful=false` → 不组装证据、返回拒答信号、**确定性写** `low_confidence_questions(source='self_check')`。
- **机械闸**:最高 rerank 分 < `rerank_min_score` 或无命中 → 证据不足信号;拒答但**不落池**。
- **负面知识**:RAG 生成 System Prompt 明列禁止承诺项(到账/到货/赔偿时效等),统一「以平台售后规则为准」。
- **拒答是确定性的,不赌模型听话**:`query_faq` 命中任一闸时返回 `evidence_insufficient` 信号并写入 `ToolRun`;**`stream_agent_turn` / `run_agent_turn` 检测到该信号时跳过收敛调用,直接输出固定拒答话术**(流式也走同一分支)。这样验收 4 不依赖模型是否遵守指令,可用单测钉死。

## §5 评估体系

- **评估集** `tests/data/eval_ch04.jsonl`,100 题:
  - `A_policy` 20:政策/流程通用类
  - `B_model` 20:型号/具体名词类(含易混型号族,验收 2)
  - `C_colloquial` 20:口语模糊改写类
  - `E_multi` 20:多跳(一问需 2~3 个不同小节)
  - `D_offtopic` 20:库外问题(验收 4)
  - 每题字段:`id` / `bucket` / `query` / `expect_section`(或 `expect_sections_all` 用于 E)/ `expect_points`(A/B/C 用)/ `difficulty`
- **指标**:
  - 检索:A/B/C/E 桶 Recall@5、MRR(按 `expect_section` 命中 `section_path`;E 桶按组给部分分;D 不计)
  - 证据覆盖:四策略 Top-K 证据机械匹配 `expect_points` 的比例(零 LLM)
  - 生成:四策略答案覆盖度、A/B/C 桶 Faithfulness、D 桶拒答率
- **四策略** = `vector` / `bm25` / `hybrid` / `hybrid_rerank`,复用 `search_knowledge(strategy=...)`。
- **裁判**:DeepSeek 上游,`temperature=0`;Faithfulness 提示词含豁免条款与边界样例。
- **健壮性**:生成/裁判调用带超时 + 失败跳过 + 低并发;检索段与证据覆盖度不依赖 LLM,始终产出。
- **`--limit`**:按桶抽样快速跑(默认全量)。
- **产物**:`data/ch04/reports/rag_eval.txt`(人读日志)+ `rag_eval.json`(机读)。
- **编造个案**:Faithfulness 判出的个案写入 `faith_cases`(一题一行,复发只更新并累加),证据全集快照存 `citations`。

## §6 数据模型

`sql/ch04-ddl.sql` 为权威(用户提供,原样落地):

- `low_confidence_questions`:`id / conversation_id(FK) / raw_question / source / reason / created_at`。本章只写 `self_check`;`retrieval_low_conf` 与 `user_feedback` 两个枚举值为后续章节预留。
- `faith_cases`:`id / eval_id(唯一) / bucket / query / strategy / answer / reason / citations(JSON) / judge_model / status / seen_count / first_seen_at / last_seen_at / resolution / resolved_at`。
- `knowledge_chunks` 复用 ch03 表结构,建库时把标量字段同步进 Milvus。

## §7 前端(Vibe Coding,不套 brainstorm/TDD/review)

- 回答里的 `[n]` 渲染成可点角标;点开浮层/侧栏显示 `section_path` + 来源 chunk 原文。数据来自 SSE `citations` 事件。
- 每段助手回答左下角 👍/👎:点击点亮所选、另一个淡出、显示「已反馈」并一次性锁定;本条信号本章**只做前端采集**,不落库(为用户反馈入口预留)。
- 效果由用户描述、迭代式直接改。

## §8 测试策略、冒烟与验收

### 8.1 TDD(确定性代码)
新 schema/混合检索/filter/RRF、rerank 客户端、`search_knowledge` 编排与首尾组装、citation 解析与暂存、`low_confidence_questions` 与 `faith_cases` 写入。

### 8.2 标注样例验证(替代 TDD)
Query 改写、结构化自评、Faithfulness 裁判、评估集本身。

### 8.3 Fail-fast 冒烟(排最前,red line)
1. Milvus BM25 + `analyzer_params={"type":"chinese"}` + `hybrid_search(RRFRanker)` 跑通,中文分词命中型号词;
2. 本地 `bge-reranker-v2-m3` 下载 + `compute_score` 跑通,并实测 CPU 上 50 对耗时;
3. 任一不通 → 停下问用户,只在固定选型内换法。

### 8.4 验收映射

| 验收 | 覆盖 |
|---|---|
| 1 四策略对比报告能跑出数字 | §5 `make eval-rag` + `data/ch04/reports/` |
| 2 型号具体问题 BM25 路能命中 | §2.2③ + B 桶 + 浏览器实测 |
| 3 引用编号定位原文、聊天页可点 | §4 + §7 + 浏览器实测 |
| 4 库外问题明确拒答且进低置信池 | §4 两闸 + D 桶 + 浏览器实测落库 |

## §9 风险闸

- **本地重排**:模型 ~2.3GB,HF 直连不稳(镜像站曾 403);CPU 上 50 对约 2~5 秒 → 评估总时长上升,用 `--limit` 控制迭代。
- **集合重建**:drop 后到重新向量化完成前,聊天检索不可用;建库脚本沿用 ch03 的幂等三段式。
- **pymilvus 3.0.2**:与教程写作时的 2.5.x 有 API 差异,一律以 Context7 当前文档 + 冒烟为准。
- **裁判自评偏袒**:同一模型生成与裁判,属已知取舍;`temperature=0` + 明确判据缓解。

## §10 文件布局(预计)

- 改:`app/kb/milvus_client.py`(新 schema / hybrid / 单路 / filter)、`app/kb/dualwrite.py`(写入新标量字段)、`app/core/retrieval.py`(重写 + strategy)、`app/tools/*`(query_faq 返回编号证据 + citations)、`app/core/agent.py` + `app/api/chat.py`(citations 帧、拒答)、`app/core/prompts.py`、`app/config.py`、`app/static/index.html`、`Makefile`。
- 新:`app/core/rerank.py`、`app/core/query_understanding.py`、`app/db/repository.py`(两表写入)、`scripts/build_eval_set.py`、`scripts/eval_ch04.py`、`scripts/smoke_ch04.py`、`tests/data/eval_ch04.jsonl`、`tests/test_*.py`。
