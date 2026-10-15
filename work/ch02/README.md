# MewHelp — Ch02 Function Calling 工具链

FastAPI + LangChain,模型直连 DeepSeek 官方(OpenAI 兼容协议)。
本章在客服系统上接 Function Calling 工具链:模型按需调用工具(订单/商品/物流/FAQ/工单),单轮收敛,结果回灌再作答。

## 启动

    cd work/ch02
    cp .env.example .env            # 填 CHAT_MODEL / CHAT_API_KEY(真实 key)
    docker compose up -d            # 起 MySQL(容器内 3306 → 宿主 3307)
    uv run python scripts/seed.py   # 灌 faq 种子(幂等)
    uv run uvicorn app.main:app --port 8000

浏览器开 http://localhost:8000 即为聊天页(带工具轨迹徽章 + markdown 渲染)。

## 验收(服务运行中)

    # 1. 标注样例评估:核对模型是否选中预期工具(真实上游,含验收 3 漏召回观测)
    uv run python scripts/eval_agent.py

    # 2. 程序化 demo:三条验收标准一眼看轨迹(走 /api/agent 非流式)
    #    Windows 无 exec 位,用 bash 调用(勿用 ./scripts/demo_agent.sh)
    bash scripts/demo_agent.sh

## 接口

    # 流式主入口(SSE:工具轨迹帧 + 逐 token 答):
    curl -N http://localhost:8000/api/chat -H 'Content-Type: application/json' \
      -d '{"user_id":"u1","message":"订单 1001 的物流到哪了"}'

    # 非流式程序化出口(返回 conversation_id / answer / tool_calls / tool_results):
    curl http://localhost:8000/api/agent -H 'Content-Type: application/json' \
      -d '{"user_id":"u1","message":"退货政策是什么"}'

## 测试

    uv run pytest -v                       # 单测(不打真实模型)
    uv run python scripts/eval_extract.py  # ch01 遗留:结构化提取标注样例评估(需服务运行中)
