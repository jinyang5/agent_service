import asyncio

import pytest

from app.core import judge


@pytest.mark.asyncio
async def test_judge_clamps_and_dedupes_covered(monkeypatch):
    class Fake:
        covered = [1, 3, 99, 1]
        faithful = False
        reason = "第 2 个数字是编的"

    class Chain:
        async def ainvoke(self, payload):
            assert "1. 满99元包邮" in payload["points"]
            assert payload["answer"] == "答案"
            return Fake()

    monkeypatch.setattr(judge, "_chain", lambda model=None: Chain())
    out = await judge.judge_answer(
        "q", "[1] 满 99 包邮", "答案", ["满99元包邮", "10元运费", "每3周"])
    assert out == {"covered": [1, 3], "faithful": False,
                   "reason": "第 2 个数字是编的"}


@pytest.mark.asyncio
async def test_judge_empty_points(monkeypatch):
    class Fake:
        covered = [1]
        faithful = True
        reason = ""

    class Chain:
        async def ainvoke(self, payload):
            return Fake()

    monkeypatch.setattr(judge, "_chain", lambda model=None: Chain())
    out = await judge.judge_answer("q", "e", "a", [])
    assert out["covered"] == []


@pytest.mark.asyncio
async def test_judge_drops_negative_and_zero(monkeypatch):
    """covered 只保留 1..len(points):0、负数、越界一律丢弃。"""

    class Fake:
        covered = [-1, 0, 2, 4]
        faithful = True
        reason = ""

    class Chain:
        async def ainvoke(self, payload):
            return Fake()

    monkeypatch.setattr(judge, "_chain", lambda model=None: Chain())
    out = await judge.judge_answer("q", "e", "a", ["甲", "乙", "丙"])
    assert out["covered"] == [2]


@pytest.mark.asyncio
async def test_judge_timeout_propagates_logs_type_only(monkeypatch, caplog):
    """自有超时;裁判失败由调用方(评估脚本)跳过,日志只记异常类型。"""

    class Chain:
        async def ainvoke(self, payload):
            await asyncio.sleep(5)

    monkeypatch.setattr(judge, "_chain", lambda model=None: Chain())
    monkeypatch.setattr(
        judge, "settings", type("S", (), {"judge_timeout": 0.01})()
    )
    with caplog.at_level("WARNING"):
        with pytest.raises(TimeoutError):
            await judge.judge_answer(
                "敏感问题", "敏感证据", "敏感答案", ["敏感要点"])
    assert "TimeoutError" in caplog.text
    assert "敏感" not in caplog.text
