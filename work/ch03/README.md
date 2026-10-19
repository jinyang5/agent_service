# MewHelp — Ch03 RAG 知识库

在 ch02 的客服工具链上接本地知识库:进程内 BGE-M3 嵌入 + Milvus Standalone + MySQL 双写。
文档按材料清单切块入库,历史对话挖 QA 去重入库,`query_faq` 由查表换成语义检索;
`/kb` 录入页把整条链路搬到浏览器上,`/admin` 是后台聚合首页(知识库一张卡)。

## 启动

    cd work/ch03
    cp .env.example .env                    # 填 CHAT_MODEL / CHAT_API_KEY(真实 key)
    make milvus-up                          # etcd + minio + milvus-standalone(先停 ch02 compose 避免同名冲突)
    docker compose up -d mysql              # MySQL 容器内 3306 → 宿主 3307(复用 mewhelp-mysql)
    make dev                                # 起服务(Windows 上必须这么起,原因见下框)

页面:http://localhost:8000 聊天页;`/kb` 知识库录入页(预览 / 建库 / 向量化 / 检索自测 / 挖知识暂存);`/admin` 后台首页。

> **Windows 起服务别走 `uv run uvicorn …`(uv + CPython 3.14 venv 转发器,本机实测)**:经
> `.venv\Scripts\python.exe`(45KB 的 venv 转发器)起的进程树是 Low integrity,再往下起 `make`(Git sh)会
> `NtCreateDirectoryObject(…) 0xC0000022` 直接死——`/kb` 页上的作业按钮(走 `/api/jobs` → `make`)会整片失败。
> `--reload` 同样带不得:uvicorn 在 Windows 上配到 SelectorEventLoop,它不支持 asyncio 子进程,作业按钮一样全灭。
> `make dev` 已绕开转发器、不带 `--reload`、直接用 uv 基础解释器起,作业链路即通;等价手起命令:

    BASE=$(grep -m1 '^home = ' .venv/pyvenv.cfg | cut -d' ' -f3)
    PYTHONPATH="$PWD/.venv/Lib/site-packages" HF_HUB_OFFLINE=1 "$BASE/python.exe" -m uvicorn app.main:app --port 8000

> `make milvus-up` 的等待循环在某些 Git sh 下会报 `unexpected EOF`、退出码 2(容器其实已经起来了),
> 用 `curl -sf http://localhost:9091/healthz` 确认就绪即可。

## 建库(CLI;与 /kb 页按钮同一套函数)

    make kb-preview     # 材料清单 + 切块预览(dry-run,不写库)— 本仓材料总块数 52
    make kb-build       # data/kb/*.md 切块 → MySQL knowledge_chunks(pending)
    make kb-mine        # 历史会话挖 QA → 暂存表 → 去重 → pending(需聊天上游)
    make kb-vectorize   # pending → BGE-M3 → Milvus upsert(id 幂等)→ 回标 done;中断重跑只补剩余
    make kb-reset       # 清 knowledge_chunks/qa_extraction_staging + drop Milvus 集合
    make seed-conv      # 灌历史会话种子(../../sql/ch03-seed.sql → mewhelp-mysql)

离线环境:首次加载 BGE-M3 走本地 cache(HF_HOME),起服务 / 跑脚本时设 `HF_HUB_OFFLINE=1`。

## 验收(三项)

1. **语义召回**:`/kb` ⑤ 检索自测「邮费是多少」→ Top-1 落在运费/包邮那块;聊天页问同一句 →
   「🔧 调用了 query_faq」徽章 + 答出满 99 包邮 / 运费规则。回归线:`make eval-retrieval` 逐样例核对 Top-1,
   任一样例未中即退出码 1——没有 ≥4/5 的宽松线。本机实测 3/5:两项未中(「东西不想要了能退吗」
   「怎么开发票」)的期望内容都落在 rank-2、与 Top-1 分差只有 ~0.01–0.02,语义上是一线之隔;
   按名次择优(rerank)是 ch04 的活,这里如实记录现状。
2. **中断重跑**:`make kb-vectorize`(或 `/kb` ④ 区按钮)中途停 → 闸条 done 与 pending 并存;
   再按一次只捡剩余 pending 补齐,`Milvus 条数 == done`,无重无漏(闸条那三个数就是账)。
3. **录入幂等**:`/kb` ① 区贴一段 → 切块预览(切块 / 表格 / 关键条款 / 查重列)→「录入入库」;
   同一段再录一次 inserted=0 全跳过;⑤ 区检索能召回刚录进去的块(带 id)。

## 接口

    # 非流式程序化出口(返回 conversation_id / answer / tool_calls / tool_results):
    curl http://localhost:8000/api/agent -H 'Content-Type: application/json' \
      -d '{"user_id":"u1","message":"邮费是多少"}'

    # 知识库:总览闸条 / 切块预览(dry-run)/ 录入 / 补向量 / 检索自测 / 暂存表
    curl http://localhost:8000/api/kb/overview
    curl -X POST http://localhost:8000/api/kb/search -H 'Content-Type: application/json' \
      -d '{"q":"邮费是多少","top_k":5}'

    # 后台作业(白名单在 app/core/jobs.py,argv 写死;…/{name}/stop 停):
    curl -X POST http://localhost:8000/api/jobs/kb-vectorize

流式聊天(SSE:工具轨迹帧 + 逐 token 答):`POST /api/chat`(ch02 起就有)。

## 测试

    uv run pytest -q          # 单测;Milvus 没起时真集合那几条自动 skip
    make eval-retrieval       # 检索回归线(需建库 + 向量化完成)
    make eval-mining          # 挖知识抽取回归线(需聊天上游)
