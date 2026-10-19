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
