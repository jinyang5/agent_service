from langchain_openai import ChatOpenAI

from app.config import settings


def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    """直连聊天上游,说 OpenAI 协议。地址、模型名、密钥全从 .env 来,换上游不动这里。"""
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=0.3,
    )
