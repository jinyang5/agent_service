async def test_admin_overview_has_kb_card(client, db_session_factory, monkeypatch):
    from app.api import kb
    async def fake_milvus():
        return {"online": True, "count": 0, "collection": "knowledge"}
    monkeypatch.setattr(kb, "milvus_state", fake_milvus)
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    card = next(c for c in r.json()["modules"] if c["key"] == "kb")
    assert card["title"] == "知识库"
    assert card["page"] == "/kb"
