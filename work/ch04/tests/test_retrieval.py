import uuid

import pytest

from app.core import retrieval
from app.kb import milvus_client as mc


def _hit(i: int, section: str, score: float = 0.5, rerank_score: float | None = None):
    h = {"id": i, "score": score, "question": f"q{i}", "answer": f"a{i}",
         "section_path": section, "content_type": "faq", "category": "c"}
    if rerank_score is not None:
        h["rerank_score"] = rerank_score
    return h


def test_arrange_head_tail():
    out = retrieval.arrange_head_tail(["a", "b", "c", "d", "e"])
    assert out[0] == "a" and out[-1] == "b"
    assert out[1:-1] == ["c", "d", "e"]


def test_arrange_head_tail_short():
    assert retrieval.arrange_head_tail(["a"]) == ["a"]
    assert retrieval.arrange_head_tail(["a", "b"]) == ["a", "b"]


def test_arrange_head_tail_does_not_mutate_input():
    items = ["a", "b", "c"]
    retrieval.arrange_head_tail(items)
    assert items == ["a", "b", "c"]


async def test_strategy_vector(monkeypatch):
    async def fake_embed(q):
        return [0.1] * 1024

    captured = {}

    def fake_dense(client, vec, top_k, category=None, collection="knowledge"):
        captured.update(vec=vec, top_k=top_k, category=category)
        return [_hit(1, "运费")]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.dense_search", fake_dense)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="vector", top_k=3,
                                           category="运费", client=object())
    assert out[0]["id"] == 1
    assert captured["top_k"] == 3 and captured["category"] == "运费"


async def test_strategy_bm25_uses_bm25_text(monkeypatch):
    captured = {}

    def fake_bm25(client, text, top_k, category=None, collection="knowledge"):
        captured["text"] = text
        return [_hit(2, "型号")]

    monkeypatch.setattr("app.kb.milvus_client.bm25_search", fake_bm25)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("Pro 型号", strategy="bm25",
                                           bm25_text="Pro 型号 猫砂盆",
                                           client=object())
    assert out[0]["id"] == 2
    assert captured["text"] == "Pro 型号 猫砂盆"


async def test_strategy_bm25_does_not_embed(monkeypatch):
    """BM25 路只走关键词:多调一次嵌入就是白烧一次编码。"""
    called = False

    async def fake_embed(q):
        nonlocal called
        called = True
        return [0.1] * 1024

    def fake_bm25(client, text, top_k, category=None, collection="knowledge"):
        return [_hit(2, "型号")]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.bm25_search", fake_bm25)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    await retrieval.search_knowledge("Pro 型号", strategy="bm25", client=object())
    assert called is False


async def test_strategy_hybrid(monkeypatch):
    async def fake_embed(q):
        return [0.1] * 1024

    captured = {}

    def fake_hybrid(client, vec, text, top_k, recall=None, category=None,
                    collection="knowledge"):
        captured.update(vec=vec, text=text, top_k=top_k, recall=recall)
        return [_hit(1, "运费", 0.9)]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="hybrid", top_k=5,
                                           client=object())
    assert out[0]["id"] == 1
    assert captured["text"] == "运费" and captured["recall"] == 50


async def test_strategy_hybrid_rerank(monkeypatch):
    async def fake_embed(q):
        return [0.1] * 1024

    def fake_hybrid(client, vec, text, top_k, recall=None, category=None,
                    collection="knowledge"):
        return [_hit(1, "运费", 0.5), _hit(2, "型号", 0.4)]

    async def fake_rerank(q, docs, top_n=None):
        assert len(docs) == 2
        return [(1, 0.95), (0, 0.2)]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("Pro 型号", strategy="hybrid_rerank",
                                           client=object())
    assert out[0]["id"] == 2 and out[0]["rerank_score"] == 0.95
    assert out[1]["id"] == 1


async def test_hybrid_rerank_top_k_feeds_both_stages(monkeypatch):
    """top_k 既是 hybrid 召回后的条数上限,也是重排的 top_n;每条 hit 都带 rerank_score。"""
    async def fake_embed(q):
        return [0.1] * 1024

    captured = {}

    def fake_hybrid(client, vec, text, top_k, recall=None, category=None,
                    collection="knowledge"):
        captured["hybrid"] = (top_k, recall, category)
        return [_hit(1, "运费", 0.5), _hit(2, "型号", 0.4)]

    async def fake_rerank(q, docs, top_n=None):
        captured["top_n"] = top_n
        return [(1, 0.9), (0, 0.8)]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("运费", strategy="hybrid_rerank", top_k=3,
                                           category="运费", client=object())
    assert captured["hybrid"] == (3, 50, "运费")
    assert captured["top_n"] == 3
    # 重排给的是原始索引:分数必须跟着对应 hit 走,不能串位(引用编号错位的直接防线)。
    assert [(h["id"], h["rerank_score"]) for h in out] == [(2, 0.9), (1, 0.8)]


async def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        await retrieval.search_knowledge("x", strategy="nope")


# --- 真实 Milvus 端到端:临时集合 + try/finally drop,绝不碰真实 knowledge ---

_QUERY_VEC = [1.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3)


def _rows() -> list[dict]:
    return [
        {"id": 1, "dense": _QUERY_VEC, "text": "运费 满 99 包邮",
         "question": "运费怎么算", "answer": "满 99 包邮",
         "section_path": "运费", "content_type": "faq", "category": "运费"},
        {"id": 2, "dense": [0.0, 1.0, 0.0] + [0.0] * (mc.DIM - 3),
         "text": "发货时效 48 小时",
         "question": "发货时效", "answer": "48 小时",
         "section_path": "发货", "content_type": "faq", "category": "发货"},
    ]


@pytest.fixture()
def milvus():
    """真实 Standalone 上的独立测试集合;连不上就 skip(先 make milvus-up)。"""
    try:
        c = mc.get_client()
        c.list_collections()
    except Exception as e:
        pytest.skip(f"Milvus 不可用,先 make milvus-up:{e}")
    coll = f"knowledge_test_{uuid.uuid4().hex[:8]}"
    try:
        mc.ensure_collection(c, collection=coll)
        mc.upsert_vectors(c, _rows(), collection=coll)
        mc.flush(c, collection=coll)
        yield c, coll
    finally:
        try:
            mc.drop(c, collection=coll)
        except Exception:
            pass  # drop 幂等;失败别掩盖 setup/用例的原始异常


async def test_real_milvus_vector_hits_expected_id(milvus, monkeypatch):
    c, coll = milvus

    async def fake_embed(q):
        return _QUERY_VEC

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    hits = await retrieval.search_knowledge("运费怎么算", strategy="vector", top_k=1,
                                            client=c, collection=coll)
    assert [h["id"] for h in hits] == [1]


async def test_real_milvus_hybrid_rerank_maps_scores_to_hits(milvus, monkeypatch):
    c, coll = milvus
    captured = {}

    async def fake_embed(q):
        return _QUERY_VEC

    async def fake_rerank(q, docs, top_n=None):
        captured["docs"] = docs
        return [(1, 0.9), (0, 0.8)]

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.core.retrieval.rerank.rerank", fake_rerank)
    hits = await retrieval.search_knowledge("运费 包邮", strategy="hybrid_rerank",
                                            client=c, collection=coll)
    assert len(hits) == 2 and {h["id"] for h in hits} == {1, 2}
    assert [h["rerank_score"] for h in hits] == [0.9, 0.8]
    # 重排返回原始索引:score 必须落在原命中上,question/answer 不能串位。
    assert f"{hits[0]['question']} {hits[0]['answer']}" == captured["docs"][1]
    assert f"{hits[1]['question']} {hits[1]['answer']}" == captured["docs"][0]
