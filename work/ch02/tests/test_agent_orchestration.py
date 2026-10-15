# tests/test_agent_orchestration.py
from langchain_core.messages import AIMessage, SystemMessage

from app.core import agent
from app.core.prompts import AGENT_SYSTEM
from app.db import repository as repo


class FakeModel:
    """turn1 返回 scripted[0](可带 tool_calls);非流式收敛返回 scripted[1];
    流式收敛按 stream_tokens 逐个 yield AIMessageChunk。
    bind_calls 记录 bind_tools 次数(收敛不 bind → 应恒为 1)。"""

    def __init__(self, scripted, stream_tokens=None):
        self._scripted = list(scripted)
        self._stream_tokens = list(stream_tokens or [])
        self.bind_calls = 0
        self.invoke_messages = []
        self.astream_messages = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        self.invoke_messages.append(list(messages))
        return self._scripted.pop(0)

    async def astream(self, messages):
        from langchain_core.messages import AIMessageChunk

        self.astream_messages.append(list(messages))
        for t in self._stream_tokens:
            yield AIMessageChunk(content=t)


async def test_no_tool_calls_returns_direct_answer(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好,喵~ 有什么可以帮您?")])
    res = await agent.run_agent_turn("u1", "你好", None, model=model)
    assert res.answer.startswith("你好") and res.tool_calls == []
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"]


async def test_system_prompt_survives_trimming(db_session_factory, db_clean):
    # 回归:trim_history 内部 trim_messages(include_system=False) 会删掉 SystemMessage,
    # 必须在 trim 之后再前置,否则模型收到不到客服人设与工具策略。
    model = FakeModel([AIMessage(content="你好")])
    await agent.run_agent_turn("u1", "你好", None, model=model)
    first_msg = model.invoke_messages[0][0]
    assert isinstance(first_msg, SystemMessage)
    assert first_msg.content == AGENT_SYSTEM


async def test_tool_call_flow_executes_and_converges(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first, AIMessage(content="您的订单 1001 正在运输中。")])
    res = await agent.run_agent_turn("u1", "订单 1001 到哪了", None, model=model)
    assert res.answer == "您的订单 1001 正在运输中。"
    assert res.tool_calls[0]["name"] == "query_logistics"
    assert res.tool_runs[0].ok is True
    assert model.bind_calls == 1                       # 收敛不 bind
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[2].tool_call_id == "c1"


async def test_continue_conversation_replays_only_final_answers(db_session_factory, db_clean):
    m1 = FakeModel([
        AIMessage(content="我帮您查一下:", tool_calls=[
            {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]),
        AIMessage(content="订单 1001 已揽件。")])
    cid = (await agent.run_agent_turn("u1", "订单1001到哪了", None, model=m1)).conversation_id
    m2 = FakeModel([AIMessage(content="还有什么可以帮您?")])
    await agent.run_agent_turn("u1", "谢谢", cid, model=m2)
    ai_contents = [m.content for m in m2.invoke_messages[0] if isinstance(m, AIMessage)]
    assert "订单 1001 已揽件。" in ai_contents            # 最终答回放
    assert "我帮您查一下:" not in ai_contents             # tool-calling preamble 不跨轮


async def test_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    try:
        await agent.run_agent_turn("u1", "hi", 999999, model=model)
        assert False, "应抛 ConversationNotFound"
    except agent.ConversationNotFound:
        pass


async def test_multiple_tool_calls_all_executed(db_session_factory, db_clean):
    # Review Focus 2:一次返回多个 tool_calls → 全部并行执行、各落一条 tool
    first = AIMessage(content="", tool_calls=[
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"},
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c2"}])
    model = FakeModel([first, AIMessage(content="已为您查询。")])
    res = await agent.run_agent_turn("u1", "订单 1001 的情况和物流", None, model=model)
    assert {tc["name"] for tc in res.tool_calls} == {"query_order", "query_logistics"}
    assert len(res.tool_runs) == 2 and all(r.ok for r in res.tool_runs)
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "tool", "assistant"]


async def test_continuing_other_users_conversation_does_not_crash(db_session_factory, db_clean):
    # Review Focus 3:无鉴权,传存在的他人会话应照常续接,不误抛
    cid = (await agent.run_agent_turn(
        "u1", "你好", None, model=FakeModel([AIMessage(content="你好")]))).conversation_id
    res = await agent.run_agent_turn("u2", "在吗", cid, model=FakeModel([AIMessage(content="在的")]))
    assert res.conversation_id == cid
