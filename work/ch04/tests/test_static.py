async def test_admin_pages_reachable_and_share_nav(client):
    for path in ("/kb", "/admin", "/rag-eval"):
        r = await client.get(path)
        assert r.status_code == 200
        assert "/static/admin.js" in r.text


async def test_rag_eval_page_has_report_ui(client):
    r = await client.get("/rag-eval")
    assert r.status_code == 200
    for text in ("重跑 RAG 评估", "编造个案台账"):
        assert text in r.text
