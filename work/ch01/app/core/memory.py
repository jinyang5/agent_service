from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


class SessionStore:
    """内存会话存储,session_id -> 消息列表。本章不做持久化。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    if not messages:
        return []
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )
    # 兜底:单条消息就超预算时,trim_messages(strategy="last") 返回空列表,
    # 连当前轮一起丢 —— 那样模型只收到 system prompt、等于没接到问题。
    # 至少保最新一条(当前轮)。
    if not trimmed:
        trimmed = [messages[-1]]
    return trimmed
