# tests/test_tools_infra.py
import asyncio

from app.tools import infra, registry


def test_registry_has_five_tools():
    names = {t.name for t in registry.get_all_tools()}
    assert names == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}


async def test_execute_unknown_tool_returns_error_run():
    run = await infra.execute_tool_call({"name": "nope", "args": {}, "id": "c1"}, conversation_id=1)
    assert run.ok is False and run.tool_call_id == "c1"
    assert "未知工具" in run.tool_message.content


async def test_timeout_then_error(monkeypatch):
    async def slow(_args):
        await asyncio.sleep(1)

    fake = type("T", (), {"name": "query_order", "ainvoke": staticmethod(slow)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
        conversation_id=1,
        timeout=0.05,
        max_retries=1,
    )
    assert run.ok is False and "失败" in run.tool_message.content


async def test_retry_succeeds_on_second_attempt(monkeypatch):
    calls = {"n": 0}

    async def flaky(_args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("瞬时错误")
        return {"ok": True}

    fake = type("T", (), {"name": "query_faq", "ainvoke": staticmethod(flaky)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call(
        {"name": "query_faq", "args": {"keyword": "x"}, "id": "c1"},
        conversation_id=1,
        timeout=1.0,
        max_retries=2,
    )
    assert run.ok is True and calls["n"] == 2


async def test_create_ticket_not_retried(monkeypatch):
    calls = {"n": 0}

    async def always_fail(_args):
        calls["n"] += 1
        raise RuntimeError("boom")

    fake = type("T", (), {"name": "create_ticket", "ainvoke": staticmethod(always_fail)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x", "ticket_type": "售后"}, "id": "c1"},
        conversation_id=1,
        timeout=1.0,
        max_retries=2,
    )
    assert run.ok is False and calls["n"] == 1  # 写类工具不重试


async def test_result_with_newlines_roundtrips(monkeypatch):
    # Review Focus 1:结果含换行/引号时,回灌的 content 必须仍是合法 JSON
    import json

    async def weird(_args):
        return {"answer": "第一行\n第二行", "quote": '他说"你好"'}

    fake = type("T", (), {"name": "query_faq", "ainvoke": staticmethod(weird)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call(
        {"name": "query_faq", "args": {"keyword": "x"}, "id": "c1"},
        conversation_id=1,
        timeout=1.0,
        max_retries=1,
    )
    assert run.ok is True
    loaded = json.loads(run.tool_message.content)
    assert loaded["answer"] == "第一行\n第二行" and loaded["quote"] == '他说"你好"'
