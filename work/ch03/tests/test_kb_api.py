from app.db import repository


async def test_preview_is_dry_run(client, db_session_factory):
    """预览只切不写:按完预览再看库存,一块都不该多。"""
    r = await client.post("/api/kb/preview",
                          json={"text": "# 售后\n\n## 运费说明\n\n满99包邮,未满10元运费。", "content_type": "policy"})
    assert r.status_code == 200
    assert r.json()["total"] >= 1
    assert await repository.count_chunks_by_status("pending") == 0


async def test_ingest_then_reingest_is_idempotent(client, db_session_factory):
    """Review Focus 4:同一份正文重录:第二次 inserted=0、skipped=全部,库里总数不变。"""
    body = {"text": "# 售后\n\n## 运费说明\n\n满99包邮,未满10元运费。", "content_type": "policy",
            "vectorize": False}
    r1 = await client.post("/api/kb/ingest", json=body)
    assert r1.status_code == 200 and r1.json()["inserted"] >= 1
    total1 = (await repository.knowledge_stats())["total"]
    r2 = await client.post("/api/kb/ingest", json=body)
    assert r2.json()["inserted"] == 0 and r2.json()["skipped"] == r1.json()["chunks"]
    assert (await repository.knowledge_stats())["total"] == total1


async def test_ingest_keeps_multiple_pieces_of_one_section(client, db_session_factory):
    """大表格按行拆出的多块共用节标题——只按问法查重会误杀,这里锁住不许。"""
    rows = "\n".join(f"| 商品{i} | {i} |" for i in range(1, 40))
    text = f"# 价目表\n\n## 价目\n\n| 商品 | 价格 |\n| --- | --- |\n{rows}"
    r = await client.post("/api/kb/ingest", json={"text": text, "content_type": "manual", "vectorize": False})
    assert r.status_code == 200
    assert r.json()["inserted"] >= 3           # 同一节多块都进了,没被误杀


async def test_ingest_empty_document_rejected(client, db_session_factory):
    """Review Focus 1:空正文 / 只有标题 → 400,且不落任何库。"""
    r = await client.post("/api/kb/ingest", json={"text": "# 只有标题\n", "content_type": "policy"})
    assert r.status_code == 400
    assert (await repository.knowledge_stats())["total"] == 0


async def test_preview_rejects_bad_input(client, db_session_factory):
    """空正文 / 非法 content_type / 不在材料清单里的文件名 一律 400。"""
    assert (await client.post("/api/kb/preview", json={"text": "   ", "content_type": "faq"})).status_code == 400
    assert (await client.post("/api/kb/preview",
                              json={"text": "正文。", "content_type": "nope"})).status_code == 400
    assert (await client.post("/api/kb/preview",
                              json={"file": "../secret.md"})).status_code == 400


async def test_overview_milvus_offline_is_state_not_error(client, db_session_factory, monkeypatch):
    """Review Focus 3:Milvus 离线 → overview 返回 online=false、consistent=None,不 500。"""
    def boom():
        raise RuntimeError("connection refused")
    monkeypatch.setattr("app.kb.milvus_client.get_client", boom)
    r = await client.get("/api/kb/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["milvus"]["online"] is False
    assert body["consistent"] is None
