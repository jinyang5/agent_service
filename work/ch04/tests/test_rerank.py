import pytest

from app.core import rerank as rr


@pytest.mark.asyncio
async def test_rerank_orders_and_truncates(monkeypatch):
    monkeypatch.setattr(rr, "_score_pairs", lambda q, docs: [0.1, 0.9, 0.5])
    out = await rr.rerank("q", ["a", "b", "c"], top_n=2)
    assert out == [(1, 0.9), (2, 0.5)]


@pytest.mark.asyncio
async def test_rerank_empty_docs():
    assert await rr.rerank("q", []) == []


@pytest.mark.asyncio
async def test_rerank_top_n_zero_returns_empty(monkeypatch):
    """top_n=0 是「一条都不要」,不能退化成不截断。"""
    monkeypatch.setattr(rr, "_score_pairs", lambda q, docs: [0.1, 0.9])
    assert await rr.rerank("q", ["a", "b"], top_n=0) == []


def test_score_pairs_normalizes_single_float(monkeypatch):
    class FakeModel:
        def compute_score(self, pairs, normalize=True):
            assert normalize is True
            assert pairs == [["q", "a"]]
            return 0.7

    monkeypatch.setattr(rr, "_model", lambda: FakeModel())
    assert rr._score_pairs("q", ["a"]) == [0.7]


@pytest.mark.asyncio
async def test_rerank_uses_to_thread(monkeypatch):
    calls = {}

    async def fake_to_thread(fn, *args):
        calls["fn"] = fn
        calls["args"] = args
        return fn(*args)

    monkeypatch.setattr(rr.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(rr, "_score_pairs", lambda q, docs: [0.2, 0.8])
    out = await rr.rerank("q", ["a", "b"])
    assert calls["fn"] is rr._score_pairs
    assert calls["args"] == ("q", ["a", "b"])
    assert out == [(1, 0.8), (0, 0.2)]
