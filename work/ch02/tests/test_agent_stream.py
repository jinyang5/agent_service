# tests/test_agent_stream.py
import pytest
from langchain_core.messages import AIMessage

from app.core import agent
from app.db import repository as repo
from tests.test_agent_orchestration import FakeModel


async def test_stream_with_tool_emits_tool_then_deltas(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["您的", "订单", "正在运输中。"])
    events = [ev async for ev in agent.stream_agent_turn("u1", "订单 1001 到哪了", None, model=model)]
    types = [e["type"] for e in events]
    assert types[0] == "tool" and events[0]["name"] == "query_logistics"
    assert types[-1] == "done" and events[-1]["conversation_id"] == 1
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "您的订单正在运输中。"
    assert model.bind_calls == 1
    msgs = await repo.list_messages(1)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]


async def test_stream_without_tool_emits_single_delta(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好呀")])
    events = [ev async for ev in agent.stream_agent_turn("u1", "你好", None, model=model)]
    assert [e["type"] for e in events] == ["delta", "done"]
    assert events[0]["text"] == "你好呀"
    assert model.astream_messages == []                 # 无工具不 astream
    msgs = await repo.list_messages(1)
    assert [m.role for m in msgs] == ["user", "assistant"]


async def test_stream_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    with pytest.raises(agent.ConversationNotFound):
        async for _ in agent.stream_agent_turn("u1", "hi", 999999, model=model):
            pass
