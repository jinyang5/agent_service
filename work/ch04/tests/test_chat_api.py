import json

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.api import chat as chat_api
from app.db import repository as repo
from app.main import app
from tests.test_agent_orchestration import FakeModel


def _client(model) -> httpx.AsyncClient:
    """注入 Task 11 的 FakeModel 作为 /api/chat 的模型依赖(走真实 stream_agent_turn)。"""
    app.dependency_overrides[chat_api.get_model] = lambda: model
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _parse_sse(text: str) -> dict:
    tools: list[str] = []
    deltas: list[str] = []
    errors: list[str] = []
    citations: list[list] = []
    done: int | None = None
    saw_done_marker = False
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if data == "[DONE]":
            saw_done_marker = True
            continue
        if event == "error":
            errors.append(json.loads(data)["message"])
            continue
        payload = json.loads(data)
        if payload.get("event") == "tool":
            tools.append(payload["name"])
        elif payload.get("event") == "citations":
            citations.append(payload["items"])
        elif payload.get("event") == "done":
            done = payload["conversation_id"]
        elif "delta" in payload:
            deltas.append(payload["delta"])
    return {"tools": tools, "deltas": deltas, "done": done,
            "errors": errors, "done_marker": saw_done_marker,
            "citations": citations}


async def _stream_chat(model, payload: dict) -> dict:
    async with _client(model) as client:
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = (await resp.aread()).decode()
    return _parse_sse(text)


async def test_chat_with_tool_streams_tool_deltas_done(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["您的", "订单", "正在运输中。"])
    out = await _stream_chat(model, {"user_id": "u1", "message": "订单 1001 到哪了"})
    assert out["tools"] == ["query_logistics"]
    assert "".join(out["deltas"]) == "您的订单正在运输中。"
    assert isinstance(out["done"], int)
    assert out["done_marker"] is True
    assert out["errors"] == []


async def test_chat_without_tool_single_delta_and_persists(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好呀")])
    out = await _stream_chat(model, {"user_id": "u1", "message": "你好"})
    assert out["tools"] == []
    assert out["deltas"] == ["你好呀"]
    assert out["done_marker"] is True
    msgs = await repo.list_messages(out["done"])
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert [m.content for m in msgs] == ["你好", "你好呀"]


async def test_chat_unknown_conversation_emits_error_no_done(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    out = await _stream_chat(
        model, {"user_id": "u1", "message": "hi", "conversation_id": 999999})
    assert out["errors"] == ["会话不存在"]
    assert out["done_marker"] is False
    assert out["done"] is None


async def test_chat_rejects_empty_message(db_session_factory):
    async with _client(FakeModel([AIMessage(content="x")])) as client:
        r = await client.post("/api/chat", json={"user_id": "u1", "message": ""})
    assert r.status_code == 422


async def test_chat_rejects_blank_message(db_session_factory):
    async with _client(FakeModel([AIMessage(content="x")])) as client:
        r = await client.post("/api/chat", json={"user_id": "u1", "message": "   "})
    assert r.status_code == 422


async def test_chat_forwards_citations_event(db_session_factory, db_clean, monkeypatch):
    async def fake(_args):
        return {"sufficient": True, "evidence": "[1] q: a",
                "citations": [{"n": 1, "chunk_id": 5, "section_path": "运费",
                               "question": "q", "answer": "a"}]}

    fake_tool = type("T", (), {"name": "query_faq",
                               "ainvoke": staticmethod(fake)})()
    monkeypatch.setattr("app.tools.registry.get_tool", lambda name: fake_tool)
    first = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "q"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["答案"])
    out = await _stream_chat(model, {"user_id": "u1", "message": "邮费多少"})
    assert out["citations"][0][0]["chunk_id"] == 5
    assert "".join(out["deltas"]) == "答案"
