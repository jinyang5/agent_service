import asyncio

import pytest
from langchain_core.exceptions import OutputParserException

from app.core import query_understanding as qu


@pytest.mark.asyncio
async def test_understand_parses_flat_fields(monkeypatch):
    class Fake:
        standard = "运费怎么算"
        expanded = ["邮费", "快递费"]

    class Chain:
        async def ainvoke(self, payload):
            assert payload == {"query": "那个邮费多少钱啊"}
            return Fake()

    monkeypatch.setattr(qu, "_chain", lambda model=None: Chain())
    out = await qu.understand("那个邮费多少钱啊")
    assert out == {"standard": "运费怎么算", "expanded": ["邮费", "快递费"]}


@pytest.mark.asyncio
async def test_understand_falls_back_to_raw_query(monkeypatch):
    class Fake:
        standard = ""
        expanded = ["", "  "]

    class Chain:
        async def ainvoke(self, payload):
            return Fake()

    monkeypatch.setattr(qu, "_chain", lambda model=None: Chain())
    out = await qu.understand("坏了咋整")
    assert out == {"standard": "坏了咋整", "expanded": []}


@pytest.mark.asyncio
async def test_understand_degrades_safely_on_upstream_error(monkeypatch):
    """上游报错/超时不能把异常抛进聊天主链路:回退原句。"""

    class Chain:
        async def ainvoke(self, payload):
            raise RuntimeError("上游挂了")

    monkeypatch.setattr(qu, "_chain", lambda model=None: Chain())
    out = await qu.understand("那个邮费多少钱啊")
    assert out == {"standard": "那个邮费多少钱啊", "expanded": []}


@pytest.mark.asyncio
async def test_understand_degrades_safely_on_timeout(monkeypatch):
    """自有超时:ainvoke 卡住时按原句降级,不外抛。"""

    class Chain:
        async def ainvoke(self, payload):
            await asyncio.sleep(5)

    monkeypatch.setattr(qu, "_chain", lambda model=None: Chain())
    monkeypatch.setattr(
        qu, "settings", type("S", (), {"query_understanding_timeout": 0.01})()
    )
    out = await qu.understand("坏了咋整")
    assert out == {"standard": "坏了咋整", "expanded": []}


@pytest.mark.asyncio
async def test_understand_degrades_safely_on_unparseable_output(monkeypatch, caplog):
    """parser 异常按原句降级;日志只记异常类型,不带模型输出/用户原文。"""

    class Chain:
        async def ainvoke(self, payload):
            raise OutputParserException("模型输出不是合法结构: 敏感内容")

    monkeypatch.setattr(qu, "_chain", lambda model=None: Chain())
    with caplog.at_level("WARNING"):
        out = await qu.understand("那个邮费多少钱啊")
    assert out == {"standard": "那个邮费多少钱啊", "expanded": []}
    assert "OutputParserException" in caplog.text
    assert "敏感内容" not in caplog.text
    assert "模型输出" not in caplog.text
    assert "那个邮费多少钱啊" not in caplog.text
