import uuid

import pytest

from app.core import retrieval
from app.kb import milvus_client as mc


@pytest.fixture()
def milvus():
    try:
        c = mc.get_client()
        c.list_collections()
    except Exception as e:
        pytest.skip(f"Milvus 不可用,先 make milvus-up:{e}")
    coll = f"knowledge_test_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(c, collection=coll)
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (mc.DIM - 3)
    mc.upsert_vectors(c, [
        {"id": 1, "vector": v_hit, "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": v_other, "question": "发货时效", "answer": "48小时"},
    ], collection=coll)
    mc.flush(c, collection=coll)
    yield c, coll
    mc.drop(c, collection=coll)


async def _coro(v):
    return v


async def test_search_returns_top_hits_above_threshold(monkeypatch, milvus):
    c, coll = milvus
    qv = [1.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3)
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", lambda q: _coro(qv))
    hits = await retrieval.search_knowledge("邮费是多少", top_k=1, min_score=0.5,
                                            client=c, collection=coll)
    assert hits and hits[0]["question"] == "运费怎么算"


async def test_below_threshold_filtered(monkeypatch, milvus):
    c, coll = milvus
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query",
                        lambda q: _coro([1.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3)))
    hits = await retrieval.search_knowledge("邮费", top_k=2, min_score=0.999999,
                                            client=c, collection=coll)
    assert hits == [] or all(h["score"] >= 0.999999 for h in hits)
