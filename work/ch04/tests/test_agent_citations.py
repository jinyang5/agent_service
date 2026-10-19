import json

from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.core import agent
from app.core.prompts import EVIDENCE_REFUSAL
from app.db.models import LowConfidenceQuestion, Message
from tests.test_agent_orchestration import FakeModel


def _install_fake_faq(monkeypatch, payload):
    async def fake(_args):
        return payload

    fake_tool = type("T", (), {"name": "query_faq",
                               "ainvoke": staticmethod(fake)})()
    monkeypatch.setattr("app.tools.registry.get_tool",
                        lambda name: fake_tool if name == "query_faq" else None)


def _faq_call():
    return AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "q"}, "id": "c1"}])


def _faq_calls_two():
    return AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "a"}, "id": "c1"},
        {"name": "query_faq", "args": {"keyword": "b"}, "id": "c2"}])


def _install_fake_faq_by_keyword(monkeypatch, mapping):
    """按工具入参 keyword 返回不同 payload(值为 Exception 时模拟执行失败)。"""
    async def fake(args):
        value = mapping[args["keyword"]]
        if isinstance(value, Exception):
            raise value
        return value

    fake_tool = type("T", (), {"name": "query_faq",
                               "ainvoke": staticmethod(fake)})()
    monkeypatch.setattr("app.tools.registry.get_tool",
                        lambda name: fake_tool if name == "query_faq" else None)


def _cite(n, chunk_id):
    return {"n": n, "chunk_id": chunk_id, "section_path": "s",
            "question": f"q{chunk_id}", "answer": f"a{chunk_id}"}


def _sufficient(cites):
    return {"sufficient": True,
            "evidence": "\n".join(
                f"[{c['n']}] {c['question']}: {c['answer']}" for c in cites),
            "citations": cites}


def _insufficient(source, reason):
    return {"sufficient": False, "signal": "evidence_insufficient",
            "source": source, "reason": reason, "citations": []}


class _FakeRun:
    def __init__(self, name, payload):
        self.name = name
        self.ok = True
        self.signal = payload.get("signal")
        self.citations = payload.get("citations", [])
        self.tool_call_id = "c1"
        self.tool_message = type("M", (), {
            "content": json.dumps(payload, ensure_ascii=False)})()


def test_faq_payload_and_signal_helpers():
    ok = _FakeRun("query_faq", {"sufficient": True, "citations": [{"n": 1}]})
    bad = _FakeRun("query_faq", {"sufficient": False,
                                 "signal": "evidence_insufficient",
                                 "source": "self_check", "reason": "不够"})
    assert agent._faq_payload(ok)["sufficient"] is True
    assert agent._faq_signal([ok]) is None
    assert agent._faq_signal([bad]) is bad
    assert agent._faq_citations([ok]) == [{"n": 1}]


async def test_refusal_skips_convergence_run(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq(monkeypatch, {
        "sufficient": False, "signal": "evidence_insufficient",
        "source": "self_check", "reason": "证据不够", "citations": []})
    # 只给 turn1 脚本;若 agent 仍尝试收敛,FakeModel 会 IndexError,测试即失败
    model = FakeModel([_faq_call()])
    res = await agent.run_agent_turn("u1", "Pro 型号能自动铲屎吗", None, model=model)
    assert res.answer == EVIDENCE_REFUSAL
    assert model.bind_calls == 1
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert len(rows) == 1
    assert rows[0].source == "self_check"
    assert rows[0].raw_question == "Pro 型号能自动铲屎吗"
    assert rows[0].reason == "证据不够"


async def test_refusal_skips_convergence_stream(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq(monkeypatch, {
        "sufficient": False, "signal": "evidence_insufficient",
        "source": "self_check", "reason": "证据不够", "citations": []})
    model = FakeModel([_faq_call()])
    events = [ev async for ev in agent.stream_agent_turn(
        "u1", "问题的原话", None, model=model)]
    assert [e["type"] for e in events] == ["tool", "delta", "done"]
    assert events[1]["text"] == EVIDENCE_REFUSAL
    assert model.astream_messages == []          # 收敛模型一次都没调
    conversation_id = events[-1]["conversation_id"]
    async with db_session_factory() as s:
        msgs = list((await s.execute(
            select(Message).where(Message.conversation_id == conversation_id)
            .order_by(Message.id))).scalars())
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assistant = [m for m in msgs if m.role == "assistant"]
    assert assistant[-1].content == EVIDENCE_REFUSAL
    assert len(rows) == 1
    assert rows[0].conversation_id == conversation_id


async def test_non_faq_tool_signal_does_not_refuse(db_session_factory, db_clean, monkeypatch):
    # 同名 signal 来自非 query_faq 工具时,不得触发整体拒答,仍走收敛
    async def fake(_args):
        return {"signal": "evidence_insufficient", "reason": "other"}

    fake_tool = type("T", (), {"name": "query_order",
                               "ainvoke": staticmethod(fake)})()
    monkeypatch.setattr("app.tools.registry.get_tool", lambda name: fake_tool)
    first = AIMessage(content="", tool_calls=[
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}])
    model = FakeModel([first, AIMessage(content="订单已查到。")])
    res = await agent.run_agent_turn("u1", "订单 1 什么状态", None, model=model)
    assert res.answer == "订单已查到。"
    assert model.bind_calls == 2                 # turn1 + 收敛各一次,未被拒答短路
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert rows == []


async def test_mechanical_gate_does_not_pool(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq(monkeypatch, {
        "sufficient": False, "signal": "evidence_insufficient",
        "source": "retrieval_low_conf", "reason": "top=0.010", "citations": []})
    model = FakeModel([_faq_call()])
    res = await agent.run_agent_turn("u1", "你们卖火星车吗", None, model=model)
    assert res.answer == EVIDENCE_REFUSAL
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert rows == []                            # 机械闸只拒答,不落池


async def test_citations_event_before_deltas(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq(monkeypatch, {
        "sufficient": True, "evidence": "[1] q: a",
        "citations": [{"n": 1, "chunk_id": 5, "section_path": "运费",
                       "question": "q", "answer": "a"}]})
    # 首 token 撑满 StreamGate 的 40 字留尾(40 字不吐、再加 1 字吐 1 段),
    # 确保流式确实分成 2 个 delta,才能验证 citations 夹在工具帧与正文之间。
    model = FakeModel([_faq_call()], stream_tokens=["答" * 40, "。"])
    events = [ev async for ev in agent.stream_agent_turn(
        "u1", "邮费多少", None, model=model)]
    types = [e["type"] for e in events]
    assert types == ["tool", "citations", "delta", "delta", "done"]
    assert events[1]["items"][0]["chunk_id"] == 5
    # 闸门不得吞字:流式 delta 拼起来必须与模型输出一致
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "答" * 40 + "。"


async def test_multiple_faq_calls_renumber_citations(db_session_factory, db_clean, monkeypatch):
    # 一轮内两次 query_faq:第二次的证据编号必须全局偏移,不能从 [1] 重新开始
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": _sufficient([_cite(1, 1), _cite(2, 2)]),
        "b": _sufficient([_cite(1, 3), _cite(2, 4)]),
    })
    model = FakeModel([_faq_calls_two(), AIMessage(content="综合回答[1][3]。")])
    res = await agent.run_agent_turn(
        "u1", "退货政策是什么,另外运费怎么算的?", None, model=model)
    assert [c["n"] for c in res.citations] == [1, 2, 3, 4]
    assert [c["chunk_id"] for c in res.citations] == [1, 2, 3, 4]
    # 第二个 run 回灌给模型的 evidence 已从 [1][2] 偏移成 [3][4]
    second = json.loads(res.tool_runs[1].tool_message.content)
    assert second["evidence"] == "[3] q3: a3\n[4] q4: a4"
    async with db_session_factory() as s:
        msgs = list((await s.execute(
            select(Message).where(Message.conversation_id == res.conversation_id)
            .order_by(Message.id))).scalars())
    tool_rows = [m for m in msgs if m.role == "tool"]
    assert json.loads(tool_rows[1].content)["evidence"] == "[3] q3: a3\n[4] q4: a4"


async def test_multiple_faq_calls_stream_citations_merged(db_session_factory, db_clean, monkeypatch):
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": _sufficient([_cite(1, 1), _cite(2, 2)]),
        "b": _sufficient([_cite(1, 3), _cite(2, 4)]),
    })
    model = FakeModel([_faq_calls_two()], stream_tokens=["答案"])
    events = [ev async for ev in agent.stream_agent_turn(
        "u1", "复合问题", None, model=model)]
    assert [e["type"] for e in events] == ["tool", "tool", "citations", "delta", "done"]
    items = events[2]["items"]
    assert [c["n"] for c in items] == [1, 2, 3, 4]
    assert [c["chunk_id"] for c in items] == [1, 2, 3, 4]


async def test_one_sufficient_faq_call_prevents_refusal(db_session_factory, db_clean, monkeypatch):
    # 第一次不足 + 第二次充足 → 有可用证据,不拒答,仍走收敛
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": _insufficient("self_check", "不够"),
        "b": _sufficient([_cite(1, 5)]),
    })
    model = FakeModel([_faq_calls_two(), AIMessage(content="根据[1]回答。")])
    res = await agent.run_agent_turn("u1", "复合问题", None, model=model)
    assert res.answer == "根据[1]回答。"
    assert model.bind_calls == 2
    assert [c["chunk_id"] for c in res.citations] == [5]
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert rows == []                            # 未拒答,不写池


async def test_insufficient_second_faq_call_still_not_refuse(db_session_factory, db_clean, monkeypatch):
    # 第一次充足 + 第二次不足 → 同样不拒答
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": _sufficient([_cite(1, 5)]),
        "b": _insufficient("self_check", "不够"),
    })
    model = FakeModel([_faq_calls_two(), AIMessage(content="根据[1]回答。")])
    res = await agent.run_agent_turn("u1", "复合问题", None, model=model)
    assert res.answer == "根据[1]回答。"
    assert model.bind_calls == 2
    assert [c["chunk_id"] for c in res.citations] == [5]
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert rows == []


async def test_all_faq_calls_insufficient_refuse_and_pool(db_session_factory, db_clean, monkeypatch):
    # 两次都不足 → 拒答;落池不依赖 _faq_signal 选中的 run,而是找 self_check 的那条
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": _insufficient("retrieval_low_conf", "top=0.010"),
        "b": _insufficient("self_check", "证据不够"),
    })
    model = FakeModel([_faq_calls_two()])        # 若误收敛会 IndexError
    res = await agent.run_agent_turn("u1", "复合问题", None, model=model)
    assert res.answer == EVIDENCE_REFUSAL
    assert model.bind_calls == 1
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert len(rows) == 1
    assert rows[0].source == "self_check"
    assert rows[0].reason == "证据不够"


async def test_failed_faq_plus_insufficient_refuses(db_session_factory, db_clean, monkeypatch):
    # 执行失败 + 证据不足:没有任何可用结果 → 拒答(旧的 all(insufficient) 会漏掉失败 run)
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": RuntimeError("boom"),
        "b": _insufficient("self_check", "证据不够"),
    })
    model = FakeModel([_faq_calls_two()])        # 若误收敛会 IndexError
    res = await agent.run_agent_turn("u1", "复合问题", None, model=model)
    assert res.answer == EVIDENCE_REFUSAL
    assert model.bind_calls == 1
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert len(rows) == 1
    assert rows[0].source == "self_check"
    assert rows[0].reason == "证据不够"


async def test_failed_faq_plus_sufficient_does_not_refuse(db_session_factory, db_clean, monkeypatch):
    # 执行失败 + 充足:存在可用证据 → 正常收敛
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": RuntimeError("boom"),
        "b": _sufficient([_cite(1, 5)]),
    })
    model = FakeModel([_faq_calls_two(), AIMessage(content="根据[1]回答。")])
    res = await agent.run_agent_turn("u1", "复合问题", None, model=model)
    assert res.answer == "根据[1]回答。"
    assert model.bind_calls == 2
    assert [c["chunk_id"] for c in res.citations] == [5]


async def test_all_faq_calls_failed_refuses(db_session_factory, db_clean, monkeypatch):
    # 两次都执行失败:无可用证据 → 拒答;失败 run 不落池
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": RuntimeError("boom-a"),
        "b": RuntimeError("boom-b"),
    })
    model = FakeModel([_faq_calls_two()])        # 若误收敛会 IndexError
    res = await agent.run_agent_turn("u1", "复合问题", None, model=model)
    assert res.answer == EVIDENCE_REFUSAL
    assert model.bind_calls == 1
    async with db_session_factory() as s:
        rows = list((await s.execute(select(LowConfidenceQuestion))).scalars())
    assert rows == []


async def test_renumber_only_rewrites_line_start_brackets(db_session_factory, db_clean, monkeypatch):
    # 正文里的 [3](非行首)不是证据编号,重编号不能碰
    _install_fake_faq_by_keyword(monkeypatch, {
        "a": _sufficient([_cite(1, 1), _cite(2, 2)]),
        "b": {"sufficient": True,
              "evidence": "[1] q3: 正文引用 [3]\n[2] q4: a4",
              "citations": [_cite(1, 3), _cite(2, 4)]},
    })
    model = FakeModel([_faq_calls_two(), AIMessage(content="答[3]")])
    res = await agent.run_agent_turn("u1", "复合问题", None, model=model)
    second = json.loads(res.tool_runs[1].tool_message.content)
    assert second["evidence"] == "[3] q3: 正文引用 [3]\n[4] q4: a4"
