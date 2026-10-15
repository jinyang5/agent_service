# MewHelp — Ch01 纯对话

FastAPI + LangChain,模型直连 DeepSeek 官方(OpenAI 兼容协议)。

## 启动

    cd work/ch01
    cp .env.example .env   # 填 CHAT_MODEL / CHAT_API_KEY(真实 key)
    uv run uvicorn app.main:app --port 8000

## 验收(服务运行中)

    # 1. 流式对话(逐 token)
    curl -N http://localhost:8000/api/chat -H 'Content-Type: application/json' \
      -d '{"session_id": "s1", "message": "你们卖猫粮吗?"}'

    # 2. 两轮上下文
    uv run python scripts/demo_chat.py

    # 3. 结构化提取
    curl http://localhost:8000/api/extract -H 'Content-Type: application/json' \
      -d '{"text": "订单 MH20260701123 的猫爬架散架了,我要退款"}'

## 测试

    uv run pytest -v                    # 单测(不打真实模型)
    uv run python scripts/eval_extract.py  # 标注样例评估(需服务运行中)
