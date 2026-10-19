from app.tools.business import query_faq


async def test_query_faq_maps_hits_to_contract(monkeypatch):
    async def fake_search(keyword, **kw):
        return [{"id": 1, "score": 0.8, "question": "运费怎么算", "answer": "满99包邮"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "邮费是多少"})
    assert out == {"hits": [{"question": "运费怎么算", "answer": "满99包邮"}]}


async def test_query_faq_empty_returns_message(monkeypatch):
    async def fake_search(keyword, **kw):
        return []
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "无关问题"})
    assert out["hits"] == []
    assert "未找到" in out["message"]
