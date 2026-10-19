# tests/test_agent_dsml.py
"""编排层回归:上游泄漏的 DSML 标记不许出现在回答里(流式与非流式都要盖住)。"""
from langchain_core.messages import AIMessage

from app.core import agent
from app.core.dsml import DSML_FALLBACK
from app.db import repository as repo
from tests.test_agent_orchestration import FakeModel
from tests.test_dsml import REAL_LEAK

TOOL_CALL = AIMessage(content="", tool_calls=[
    {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])


async def test_run_agent_turn_no_tool_strips_markup(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content=REAL_LEAK)])
    res = await agent.run_agent_turn("u1", "退货政策是啥", None, model=model)
    assert "DSML" not in res.answer
    assert res.answer == "亲，帮您查一下退货政策的具体内容哈～ 🐾"
    msgs = await repo.list_messages(res.conversation_id)
    assert "DSML" not in (msgs[-1].content or "")          # 历史里也不许留


async def test_run_agent_turn_pure_markup_returns_fallback(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content='<｜｜DSML｜｜ calls>\n</｜｜DSML｜｜ calls>')])
    res = await agent.run_agent_turn("u1", "退货政策是啥", None, model=model)
    assert res.answer == DSML_FALLBACK


async def test_run_agent_turn_converge_leak_stripped(db_session_factory, db_clean):
    model = FakeModel([TOOL_CALL, AIMessage(content=REAL_LEAK)])
    res = await agent.run_agent_turn("u1", "订单 1001 到哪了", None, model=model)
    assert [t["name"] for t in res.tool_calls] == ["query_logistics"]
    assert "DSML" not in res.answer


async def test_stream_never_emits_markup(db_session_factory, db_clean):
    model = FakeModel([TOOL_CALL], stream_tokens=[
        "亲，帮您查一下～\n\n", "<｜｜DSML｜｜ calls>\n",
        '<｜｜DSML｜｜ invoke name="query_faq">\n'])
    events = [ev async for ev in agent.stream_agent_turn("u1", "订单 1001 到哪了", None, model=model)]
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "DSML" not in text
    assert text.endswith(DSML_FALLBACK)                    # 一个字都没漏,补一句人话
    msgs = await repo.list_messages(1)
    assert "DSML" not in (msgs[-1].content or "")


async def test_stream_empty_converge_gets_fallback(db_session_factory, db_clean):
    # 收敛又返回结构化 tool_calls(正文为空)时,不能给用户一个空气泡
    model = FakeModel([TOOL_CALL], stream_tokens=[])
    events = [ev async for ev in agent.stream_agent_turn("u1", "订单 1001 到哪了", None, model=model)]
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert text == DSML_FALLBACK
