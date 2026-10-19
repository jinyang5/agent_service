import asyncio

import pytest
from langchain_core.exceptions import OutputParserException

from app.core import selfcheck


@pytest.mark.asyncio
async def test_check_sufficient_parses_flat(monkeypatch):
    class Fake:
        useful = False
        reason = "证据只讲运费,未涉及型号"

    class Chain:
        async def ainvoke(self, payload):
            assert "[1] 满 99 包邮" in payload["evidence"]
            assert "[2] 七天无理由" in payload["evidence"]
            return Fake()

    monkeypatch.setattr(selfcheck, "_chain", lambda model=None: Chain())
    out = await selfcheck.check_sufficient(
        "Pro 型号功能", ["满 99 包邮", "七天无理由"])
    assert out == {"useful": False, "reason": "证据只讲运费,未涉及型号"}


@pytest.mark.asyncio
async def test_check_sufficient_empty_evidence(monkeypatch):
    class Fake:
        useful = False
        reason = "无证据"

    class Chain:
        async def ainvoke(self, payload):
            assert payload["evidence"] == "(无证据)"
            return Fake()

    monkeypatch.setattr(selfcheck, "_chain", lambda model=None: Chain())
    assert (await selfcheck.check_sufficient("q", []))["useful"] is False


@pytest.mark.asyncio
async def test_check_sufficient_fail_open_on_upstream_error(monkeypatch):
    """自评上游报错不能把异常抛进 RAG 主链路:按 useful=True 放行,不误拒答。"""

    class Chain:
        async def ainvoke(self, payload):
            raise RuntimeError("上游挂了")

    monkeypatch.setattr(selfcheck, "_chain", lambda model=None: Chain())
    out = await selfcheck.check_sufficient("q", ["证据"])
    assert out["useful"] is True
    assert out["reason"] == "self_check_unavailable"


@pytest.mark.asyncio
async def test_check_sufficient_fail_open_on_timeout(monkeypatch):
    """自有超时:ainvoke 卡住时按 useful=True 降级,不外抛。"""

    class Chain:
        async def ainvoke(self, payload):
            await asyncio.sleep(5)

    monkeypatch.setattr(selfcheck, "_chain", lambda model=None: Chain())
    monkeypatch.setattr(
        selfcheck, "settings", type("S", (), {"selfcheck_timeout": 0.01})()
    )
    out = await selfcheck.check_sufficient("q", ["证据"])
    assert out["useful"] is True
    assert out["reason"] == "self_check_unavailable"


@pytest.mark.asyncio
async def test_check_sufficient_logs_type_only(monkeypatch, caplog):
    """日志只记异常类型,不带模型输出/用户问题/证据原文。"""

    class Chain:
        async def ainvoke(self, payload):
            raise OutputParserException("模型输出不是合法结构: 敏感内容")

    monkeypatch.setattr(selfcheck, "_chain", lambda model=None: Chain())
    with caplog.at_level("WARNING"):
        out = await selfcheck.check_sufficient("那个邮费多少钱啊", ["满 99 包邮"])
    assert out["useful"] is True
    assert "OutputParserException" in caplog.text
    assert "敏感内容" not in caplog.text
    assert "那个邮费多少钱啊" not in caplog.text
    assert "满 99 包邮" not in caplog.text
