import json

from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.api import chat as chat_api
from app.main import app


class _FakeChat(BaseChatModel):
    """按固定 chunk 序列吐字的假模型;error 非空时 astream 直接抛。不依赖库内置 fake 的粒度。"""

    chunks: tuple[str, ...] = ()
    error: Exception | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("".join(self.chunks)))])

    async def _astream(self, messages, stop=None, **kwargs):
        if self.error:
            raise self.error
        for c in self.chunks:
            yield ChatGenerationChunk(message=AIMessageChunk(content=c))


def make_client(model: BaseChatModel) -> TestClient:
    app.dependency_overrides[chat_api.get_model] = lambda: model
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()
    chat_api.store._sessions.clear()


def _collect_sse(resp) -> tuple[list[str], str]:
    deltas: list[str] = []
    last = ""
    for line in resp.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            last = payload
        else:
            deltas.append(json.loads(payload)["delta"])
    return deltas, last


def test_chat_streams_deltas_and_done():
    client = make_client(_FakeChat(chunks=("你", "好", "呀")))
    with client.stream("POST", "/api/chat", json={"session_id": "s1", "message": "在吗"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        deltas, last = _collect_sse(resp)
    assert deltas == ["你", "好", "呀"]          # 逐 chunk 组帧
    assert last == "[DONE]"


def test_chat_persists_history_for_next_turn():
    client = make_client(_FakeChat(chunks=("第一", "轮答")))
    with client.stream("POST", "/api/chat", json={"session_id": "s2", "message": "第一问"}) as r:
        _collect_sse(r)
    with client.stream("POST", "/api/chat", json={"session_id": "s2", "message": "第二问"}) as r:
        _collect_sse(r)
    history = chat_api.store.get("s2")
    assert [m.type for m in history] == ["human", "ai", "human", "ai"]
    assert [m.content for m in history] == ["第一问", "第一轮答", "第二问", "第一轮答"]


def test_chat_error_streams_error_frame_and_no_store():
    client = make_client(_FakeChat(error=RuntimeError("boom")))
    with client.stream("POST", "/api/chat", json={"session_id": "s3", "message": "你好"}) as resp:
        lines = list(resp.iter_lines())
    assert any(line.startswith("event: error") for line in lines)
    assert chat_api.store.get("s3") == []          # 出错不写半条历史


def test_chat_validates_empty_message():
    client = make_client(_FakeChat(chunks=("x",)))
    assert client.post("/api/chat", json={"session_id": "s4", "message": ""}).status_code == 422
