import pytest

from app.tools import business


def _hit(i, section, rerank=0.9, category="c"):
    return {"id": i, "score": 0.5, "rerank_score": rerank,
            "question": f"q{i}", "answer": f"a{i}",
            "section_path": section, "content_type": "faq", "category": category}


@pytest.fixture(autouse=True)
def _default_understanding(monkeypatch):
    async def fake_understand(q, model=None):
        return {"standard": q, "expanded": ["运费"]}

    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)


async def test_query_faq_sufficient_numbers_after_head_tail(monkeypatch):
    async def fake_search(query, **kw):
        assert kw["strategy"] == "hybrid_rerank"
        assert kw["bm25_text"] == "邮费多少 运费"
        return [_hit(1, "运费政策"), _hit(2, "退货政策", 0.7),
                _hit(3, "会员政策", 0.6)]

    async def fake_check(q, texts, model=None):
        return {"useful": True, "reason": "够"}

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)

    out = await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert out["sufficient"] is True
    # 首尾组装:1 首、2 尾、3 居中 → 编号顺序 1,3,2
    assert [c["chunk_id"] for c in out["citations"]] == [1, 3, 2]
    assert [c["n"] for c in out["citations"]] == [1, 2, 3]
    assert out["evidence"].startswith("[1] q1: a1")
    assert "[2] q3: a3" in out["evidence"]
    assert "[3] q2: a2" in out["evidence"]


async def test_query_faq_understanding_falls_back(monkeypatch):
    async def boom(q, model=None):
        raise RuntimeError("上游挂了")

    async def fake_search(query, **kw):
        assert query == "原话问题"
        return [_hit(1, "p")]

    async def fake_check(q, texts, model=None):
        return {"useful": True, "reason": "够"}

    monkeypatch.setattr("app.core.query_understanding.understand", boom)
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "原话问题"})
    assert out["sufficient"] is True


async def test_query_faq_no_hits_mechanical_gate(monkeypatch):
    async def fake_search(query, **kw):
        return []

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    out = await business.query_faq.ainvoke({"keyword": "火星车"})
    assert out["sufficient"] is False
    assert out["signal"] == "evidence_insufficient"
    assert out["source"] == "retrieval_low_conf"
    assert out["citations"] == []


async def test_query_faq_low_score_mechanical_gate(monkeypatch):
    async def fake_search(query, **kw):
        return [_hit(1, "p", rerank=0.1)]

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    out = await business.query_faq.ainvoke({"keyword": "火星车"})
    assert out["source"] == "retrieval_low_conf"
    assert "top=0.100" in out["reason"]


async def test_query_faq_self_check_gate(monkeypatch):
    async def fake_search(query, **kw):
        return [_hit(1, "运费", rerank=0.8)]

    async def fake_check(q, texts, model=None):
        return {"useful": False, "reason": "问型号但证据只讲运费"}

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "Pro 型号能自动铲屎吗"})
    assert out["sufficient"] is False
    assert out["source"] == "self_check"
    assert out["signal"] == "evidence_insufficient"
    assert "型号" in out["reason"]


async def test_query_faq_self_check_exception_fails_open(monkeypatch):
    async def fake_search(query, **kw):
        return [_hit(1, "运费", rerank=0.8)]

    async def boom(q, texts, model=None):
        raise RuntimeError("自评上游挂了")

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", boom)
    out = await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert out["sufficient"] is True


async def test_query_faq_passes_category(monkeypatch):
    captured = {}

    async def fake_search(query, **kw):
        captured.update(kw)
        return [_hit(1, "商品规格手册 / MH-W40")]

    async def fake_check(q, texts, model=None):
        return {"useful": True, "reason": "够"}

    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    await business.query_faq.ainvoke(
        {"keyword": "MH-W40 滤芯", "category": "商品规格手册"})
    assert captured["category"] == "商品规格手册"
