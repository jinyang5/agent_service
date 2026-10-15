# MewHelp Ch01 设计:纯对话客服(SSE 流式 + Prompt 模板 + 结构化提取)

> 重走版,2026-10-12。仓库已有 ch01–ch10 成品,本文件是**重新走一遍 ch01 时新写的 spec**,
> 不是改写成品的补丁。成品代码不动、留作参考答案;本章代码落在独立目录 `work/ch01/`。
> 范本:`docs/superpowers/specs/2026-07-13-ch01-pure-chat-design.md`。

## 目标

电商智能客服系统第一章:跑通纯对话。不做工具调用、不做 Agent 循环。

验收标准(三条 curl):
1. 调对话接口能看到流式回复(逐 token)。
2. 连续问两轮,第二轮能接住第一轮的上下文。
3. 发一段售后描述,能拿到结构化 JSON(订单号 / 诉求类型 / 期望方案)。

## 技术栈(定死,不可自行更换)

- Python 3.12 + FastAPI + LangChain(`langchain-openai`)
- 模型直连上游,应用侧统一说 OpenAI 协议;地址、模型名、密钥全在 `.env`(本章只有 `CHAT_*` 一组)
- 依赖管理:uv
- 验收上游:**DeepSeek 官方**(`https://api.deepseek.com/v1`),key 由用户提供,只放 `.env`、不进 git;实现时用 Context7 核对官方最新模型名

## 落点与隔离

- 成品 `app/ tests/ scripts/` 一律不动,当参考答案。
- 本章代码自包含在 `work/ch01/`,有自己的 `pyproject.toml` / `.env.example` / `app/` / `tests/`,不污染根目录依赖。
- 过程留痕:`dev-notes/ch01.md`(仓库根;`.gitignore` 已排除 `dev-notes/`,不进 git)。

## 架构:单进程

单进程 FastAPI(`:8000`)。`ChatOpenAI` 直连 `settings.chat_base_url`,换上游只改 `.env`,不动代码。
会话历史存服务端内存 dict,按 `session_id` 索引;每次调用前按 token 预算裁剪。

## 目录结构

```
work/ch01/
  app/
    main.py            # FastAPI 入口,挂两个路由
    config.py          # pydantic-settings 读 .env
    core/llm.py        # ChatOpenAI 工厂
    core/prompts.py    # PromptTemplate 集中管理
    core/memory.py     # 内存会话存储 + token 预算裁剪
    schemas/chat.py    # ChatRequest
    schemas/extract.py # ExtractRequest + 结构化提取模型
    api/chat.py        # POST /api/chat(SSE)
    api/extract.py     # POST /api/extract
  tests/
  pyproject.toml
  .env.example
  README.md
```

## 接口契约

### POST /api/chat(SSE 流式对话)

请求:

```json
{"session_id": "s1", "message": "你们家猫粮发什么快递?"}
```

响应 `text/event-stream`,逐 token 推送:

```
data: {"delta": "我"}
data: {"delta": "们"}
...
data: [DONE]
```

- `session_id` 客户端指定,同一会话多轮复用;服务端内存 dict 按它记账,重启丢历史(接受)。
- 每次调用前按 token 预算裁剪(默认 2000,可配)。
- 上游 LLM 出错:SSE 推 `event: error` + 错误信息后关流。

### POST /api/extract(结构化提取,非流式)

请求:

```json
{"text": "我上周买的猫爬架,订单号 MH20260701123,到货就散架了,我要退款"}
```

响应(`with_structured_output` 绑定 pydantic 模型):

```json
{"order_id": "MH20260701123", "request_type": "退款", "expected_solution": "到货损坏要求退款"}
```

- 订单号缺失返回 `null`,禁止编造。
- 上游失败返回 502 JSON。

## Prompt 管理

`core/prompts.py` 集中管理两个模板:

1. **客服对话 system prompt**:写清角色设定(电商客服)与行为约束——不臆造订单/物流/库存/价格信息、超范围话题引导回购物相关、语气规范、情绪安抚。
2. **提取任务 system prompt**:面向售后描述的字段提取指令(三个字段各自怎么取、订单号缺失怎么处理)。

## 历史裁剪与 token 预算

- 用 `langchain_core` 的 `trim_messages` + `count_tokens_approximately`(近似计数,不引 tiktoken,跨上游通用;实现时用 Context7 核对接口)。
- 策略:保 system prompt + 从最近往前保留完整轮次(`start_on="human"`,不留半轮)。
- 预算 `TOKEN_BUDGET` 默认 2000,可经 `.env` 配置。

## 测试与验证

按「可单测代码走 TDD、纯 Prompt 产出走标注样例」拆两类:

**TDD(pytest,`FakeListChatModel`,不打真实模型):**
- `core/memory.py`:会话存储、token 预算裁剪
- SSE 组帧
- schema 校验(结构化提取模型枚举、可空字段)

**标注样例验证(真实模型,验收时跑):**
- 5 条标注好的售后描述跑 `/api/extract` 核对字段,覆盖 edge case:订单号缺失、诉求模糊。
- 脚本连发两轮对话,验证第二轮接住第一轮上下文。

## 本章不做

- 工具调用、Agent 循环
- 会话持久化(重启丢历史,接受)
- 鉴权、限流、多副本
- 对话中自动触发提取(提取是独立接口)

## 关键决策记录

| 决策 | 选择 | 备选与理由 |
|---|---|---|
| 代码落点 | `work/ch01/` 独立目录,成品不动 | 用户确认「另开目录」;清空重建/就地覆盖均有破坏性 |
| 验收上游 | DeepSeek 官方 | 用户确认 |
| 会话管理 | 服务端内存 + session_id | 客户端全量 history 对 curl 验收不友好;与 ch02 落库演进方向一致 |
| 提取形态 | 独立 `/api/extract` | 融合进对话需意图判断,超出本章「纯对话」范围 |
| token 计数 | 近似计数(不引 tiktoken) | 跨上游通用、零额外依赖 |

## 开放项(验收时与用户敲定,现按默认值实现)

1. `request_type` 枚举档位:现用 `退款 / 换货 / 维修 / 投诉 / 其他`(5 档)。
2. `expected_solution` 形态:现用自由文本 `str`。
3. 会话字段命名:现用 `session_id`(比最初讨论的 `user_id + conversation_id` 精简,ch01 无多用户持久化,单 key 够用)。
