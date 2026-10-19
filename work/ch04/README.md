# MewHelp — Ch04 混合检索 + 重排 + 评估

在 ch03 的知识库上把 RAG 检索做深一层:Milvus 原生 BM25 + dense 混合召回(RRF k=60)、
本地 `bge-reranker-v2-m3` 精排、Query 理解(口语归一 + 同义词扩展)、元数据过滤、
机械闸 + LLM 自评闸两道证据闸、可点引用,以及 100 题五桶 × 四策略的
检索/生成评估。评估页把报告变成可读的浏览器页面,`low_confidence_questions` /
`faith_cases` 两张新表分别承接低置信问题与编造个案。

## 启动

    cd work/ch04
    cp .env.example .env                    # 填 CHAT_MODEL / CHAT_API_KEY(真实 key)
    # 若 mewhelp-mysql 已在运行(ch02/ch03 栈),跳过下面这一行,不要在 ch04 重复 up
    docker compose up -d mysql              # 仅容器未起时;宿主 3307 → 容器 3306
    make milvus-up                          # etcd + minio + milvus-standalone(v2.6.22)
    make dev                                # Windows 上必须这么起,原因见下

页面:

- `http://localhost:8000/` 聊天页(引用角标、👍/👎 反馈)
- `/kb` 知识库录入页(预览 / 建库 / 向量化 / 检索自测 / 挖知识暂存)
- `/rag-eval` RAG 评估页(报告只读展示 + 编造个案台账 + 重跑/停止)
- `/admin` 后台首页(知识库与评估入口)

模型缓存:`HF_HOME=E:\model`、`HF_HUB_OFFLINE=1`。Makefile 已 export;直接 `uv run`
跑脚本时请自行 export:

    HF_HOME=E:/model HF_HUB_OFFLINE=1 uv run pytest -q

> **Windows 起服务别走 `uv run uvicorn …`(uv + CPython 3.14 venv 转发器,本机实测)**:
> 经 `.venv\Scripts\python.exe`(45KB 的 venv 转发器)起的进程树是 Low integrity,再往下起
> `make`(Git sh)会 `NtCreateDirectoryObject(…) 0xC0000022` 直接死——`/kb`、`/rag-eval`
> 页上的作业按钮(走 `/api/jobs` → `make`)会整片失败。`--reload` 同样带不得:uvicorn 在
> Windows 上配到 SelectorEventLoop,它不支持 asyncio 子进程,作业按钮一样全灭。
> `make dev` 已绕开转发器、不带 `--reload`、直接用 uv 基础解释器起;等价手起命令:

    BASE=$(grep -m1 '^home = ' .venv/pyvenv.cfg | cut -d' ' -f3)
    PYTHONPATH="$PWD/.venv/Lib/site-packages" HF_HUB_OFFLINE=1 "$BASE/python.exe" -m uvicorn app.main:app --port 8000

> `make milvus-up` 的等待循环在某些 Git sh 下会报 `unexpected EOF`、退出码 2
> (容器其实已经起来了),用 `curl -sf http://localhost:9091/healthz` 确认就绪即可。

### 停 Milvus 的作用域

    make milvus-down   # docker compose stop etcd minio milvus-standalone

`docker compose` 按 **compose project**(本目录名 `ch04`)作用域操作,不会按容器名
「顺手全停」。本机当前这套 Milvus 容器实际是 **ch03 的 compose 项目**起的
(容器 label `com.docker.compose.project=ch03`),因此:

- `work/ch04` 下的 `make milvus-down` 只作用于 ch04 项目自己的容器:**不会停**
  ch03 项目起的 `mewhelp-milvus` 三件套,**不会停**共享的 `mewhelp-mysql`,
  也不会误停其他 compose project 的容器。
- 要停当前这套(ch03 项目)Milvus,请在 `work/ch03` 下 `make milvus-down`,或直接
  `docker stop mewhelp-milvus mewhelp-milvus-etcd mewhelp-milvus-minio`。
- `milvus-up` 的「已起」判定只看容器是否 running:若容器在跑就直接复用(Milvus
  数据卷随启动它的项目走);若存在遗留的**已停止**同名固定容器,`docker compose up`
  会因容器名冲突失败——先 `docker rm` 掉那个停着的容器再起。

数据库容器要继续用就保持运行;`mewhelp-mysql-data` 是显式命名的共享卷。

## 检索管线

    query_faq:
      ① Query 理解   口语→标准问法 + 同义词扩展(只作用于 BM25 检索侧)
      ② 元数据过滤   category(Milvus filter)
      ③ 混合召回     dense Top-50 + BM25 Top-50 → hybrid_search(RRFRanker(k=60))
      ④ 重排         bge-reranker-v2-m3 精排 Top-10(本地 CPU,asyncio.to_thread)
      ⑤ 机械闸       无命中或最高 rerank 分 < rerank_min_score → 拒答(不落池)
      ⑥ 自评闸       useful=false → 拒答 + 落池(source=self_check)
      ⑦ 首尾组装     第 1 条放首、第 2 条放尾、其余居中
      ⑧ 编号证据 + citations(ToolRun 暂存 → SSE citations 事件)

命中任一闸时 `query_faq` 返回 `signal=evidence_insufficient`;agent 检测到后
**跳过收敛模型调用**,直接输出固定拒答话术。两道闸的落池口径不同:

- **机械闸**(`source=retrieval_low_conf`):只拒答,**不写** `low_confidence_questions`。
  例:问「你们支持比特币付款吗」→ `sufficient=False`、`source=retrieval_low_conf`。
- **自评闸**(`source=self_check`):拒答并写
  `low_confidence_questions(source='self_check')`。例:问
  「这边的plus会员有什么权益吗」→ `sufficient=False`、`source=self_check`,
  池里出现该原话。

自评上游异常/超时按 fail-open 放行,不会把评审模型故障误判成「证据不足」。

## 引用

回答里的 `[n]` 就是喂给模型的编号证据序号。聊天页点击角标会浮出该条证据的
`section_path` + 原文;流式接口通过 SSE `citations` 事件下发
`{n, chunk_id, section_path, question, answer}`。证据按「首、尾、中间」重排,
缓解 lost-in-the-middle,但编号始终从 1 连续。

## 反馈

每条回答左下角有 👍/👎,点一次即锁定并显示「已反馈」。这是**一次性前端反馈**:
本章只采集交互、不落库(留给后续数据飞轮章接 `low_confidence_questions` 的
`user_feedback` 入口)。

## 建库与 schema 迁移

    make kb-preview        # 材料清单 + 切块预览(不写库;本仓文档材料 52 块)
    make kb-build          # data/kb/*.md → MySQL knowledge_chunks(pending)
    make kb-mine           # 历史会话挖 QA → 暂存 → 去重 → pending(需聊天上游)
    make kb-vectorize      # pending → BGE-M3 + BM25 text → Milvus upsert(id 幂等)
    make kb-revectorize    # ch04 schema 迁移:全部 chunk 置回 pending + drop 旧集合
    make kb-reset          # 清空 MySQL 两表 + drop 集合(慎用)
    make seed-conv         # 灌历史会话种子

当前 Milvus 知识库 82 块(文档切块 + 挖掘 QA);`kb-preview` 的 52 块只统计
`data/kb` 的文档材料。离线环境首次加载 BGE-M3 走本地 `HF_HOME`。

## 评估

    make eval-check                         # 评估集结构 + ground truth 对真实 KB 切块
    make eval-rag ARGS="--limit 5"          # 每桶 5 题快速跑
    make eval-rag                           # 100 题全量(四策略 + 生成段)
    make eval-rag ARGS="--skip-generation"  # 只跑确定性检索段/证据覆盖度
    make judge-check                        # 6 条标注样例重放裁判
    make eval-rag-prompt                    # RAG 生成 Prompt 的 6 条标注样例(需上游)
    make verify-ch04                        # 验收四条程序化检查(不跑全量评估)

产物:`data/ch04/reports/rag_eval.txt`(人读)+ `rag_eval.json`(机读)。

指标:

- 检索:A/B/C/E 桶 Recall@5、MRR(按 `expect_section` 命中 `section_path`;E 按组部分分)
- 证据覆盖度:四策略 Top-10 机械匹配 `expect_points`(零 LLM)
- 生成:四策略答案覆盖度、A/B/C Faithfulness(hybrid_rerank)、D 桶拒答率
- 个案:Faithfulness 判出的编造写入 `faith_cases`,一题一行,复发只累加 `seen_count`
- `generation.completed` 只认 **A/B/C 生成任务有成功**;D 桶拒答成功不算生成段完成

### 评估页

`/rag-eval` **只读产物、不重算**:原样透传 `retrieval` / `evidence_coverage` /
`generation`,只把四个答案桶的 MRR 做算术平均得到 `retrieval_overall`,并注明口径。
页面展示检索质量、证据覆盖度、生成段、D 桶拒答、错误和编造个案台账;个案可改
「未解决 / 已解决 / 无需解决」并填写处置说明。重跑/停止走 `/api/jobs/eval-rag`
(白名单作业,argv 写死),长任务在后台跑、日志轮询回显。

## 两张新表

ch04 在 MySQL 新增两张表(DDL 见 `sql/ch04-ddl.sql`):

- `low_confidence_questions`:低置信问题池。`source` 为
  `retrieval_low_conf` / `self_check` / `user_feedback`;本章只有自评闸
  (`self_check`)会写入,机械闸不写,前端 👍/👎 也暂不落库。
- `faith_cases`:编造个案台账。Faithfulness 裁判判出的每条编造一题一行
  (`eval_id` 唯一),记录查询、答案、理由、引用快照、裁判模型与处置状态;
  再次被判出只更新快照并 `seen_count+1`,已处置的题会退回「未解决」标记复发。

知识库本身仍沿用 ch03 的 `knowledge_chunks`;`kb-revectorize` 负责 ch04 schema 迁移。

## 验收四条

1. **四策略报告能跑出数字**:`make eval-rag` →
   `data/ch04/reports/rag_eval.{txt,json}`;`make verify-ch04` 的 1a/1b 通过
   (4 策略 × 4 可作答桶都有 Recall@5 / MRR / n)。
2. **型号问题 BM25 能命中**:`make verify-ch04` 的 2 检查
   「MH-W40 滤芯多久换一次」的 BM25 Top-1 含 MH-W40;B 桶检索指标可佐证。
3. **引用编号可定位原文**:聊天页回答带 `[n]`,点击显示 `section_path` + 原文;
   `make verify-ch04` 的 3 检查 citations 从 1 连续、每条都有 `chunk_id` 和
   `section_path`。截图存本地 `dev-notes/`(该目录已被 gitignore,不入库),
   路径如 `dev-notes/ch04-acceptance3.png`。
4. **库外问题拒答且进低置信池**:
   - 「你们支持比特币付款吗」→ 机械闸拒答,`source=retrieval_low_conf`,**不落池**;
   - 「这边的plus会员有什么权益吗」→ 自评闸拒答,`source=self_check`,落池
     (脚本查 `low_confidence_questions` 确认;若池里还没有,会用 agent 同一条
     落池 helper 写入后再查)。
   若某次该问被机械闸先拦住,`make verify-ch04` 会自动改问「猫砂盆的滤芯多久换
   一次」验证自评落池;若该问也过不了分数闸(当前 KB 实测 top≈0.054),脚本会
   明确报「未触发 self_check」,此时换一个能过分数闸但库内无答案的问法再验。
   `make verify-ch04` 的 4 覆盖上述两种口径。

   > **落池快照口径**:`low_confidence_questions` 里可能已有历史 `self_check`
   > 行(例如用户在聊天页问过同一句)。`verify-ch04` 用**调用前后快照**判定:
   > 比特币问只要求本次直接调用 `query_faq` **不新增**行(不要求全表为 0);
   > plus 问要求出现新行,或已有同题 `self_check` 行且行数不减少。聊天路径下
   > 模型的关键词/改写可能让同一句走 `self_check` 而非机械闸,这是允许的;
   > 脚本断言的是它**直接调用 `query_faq` 的本次结果**。

## 接口

    # 非流式程序化出口(返回 conversation_id / answer / tool_calls / tool_results)
    curl http://localhost:8000/api/agent -H 'Content-Type: application/json' \
      -d '{"user_id":"u1","message":"邮费是多少"}'

    # 评估报告(只读产物)与编造个案台账
    curl http://localhost:8000/api/rag-eval/overview
    curl http://localhost:8000/api/rag-eval/faith-cases

    # 后台作业(白名单在 app/core/jobs.py,argv 写死;…/{name}/stop 停)
    curl -X POST http://localhost:8000/api/jobs/eval-rag

流式聊天(SSE:工具轨迹帧 + 逐 token 答 + citations 事件):`POST /api/chat`。

## 测试

    uv run pytest -q          # 单测;Milvus 没起时真集合那几条自动 skip
    make verify-ch04          # 验收四条程序化检查(需 Milvus + MySQL + 上游)
    make eval-retrieval       # ch03 检索回归线(需建库)
