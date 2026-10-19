import uuid

import pytest

from app.kb import milvus_client as mc

URI = "http://localhost:19530"


@pytest.fixture()
def milvus():
    """真实 Standalone 上的独立测试集合;连不上就 skip(先 make milvus-up)。"""
    try:
        c = mc.get_client(uri=URI)
        c.list_collections()
    except Exception as e:
        pytest.skip(f"Milvus 不可用,先 make milvus-up:{e}")
    coll = f"knowledge_test_{uuid.uuid4().hex[:8]}"
    try:
        mc.ensure_collection(c, collection=coll)
        yield c, coll
    finally:
        try:
            mc.drop(c, collection=coll)
        except Exception:
            pass  # drop 幂等;失败别掩盖 setup/用例的原始异常


def _vec(seed: float) -> list[float]:
    v = [0.0] * mc.DIM
    v[0] = seed
    v[1] = 1.0 - seed
    return v


def _rows() -> list[dict]:
    return [
        {"id": 1, "dense": _vec(1.0), "text": "运费 满 99 元包邮 未满收取 10 元运费",
         "question": "运费怎么算", "answer": "满 99 元包邮,未满收取 10 元运费",
         "section_path": "商品与购物 FAQ / 运费怎么算", "content_type": "faq",
         "category": "商品与购物 FAQ"},
        {"id": 2, "dense": _vec(0.0), "text": "智能猫砂盆 MH-LP100 废砂盒每 5 至 7 天倾倒一次",
         "question": "智能猫砂盆 Pro(型号 MH-LP100)",
         "answer": "废砂盒每 5 至 7 天倾倒一次",
         "section_path": "商品规格手册 / 智能猫砂盆 Pro(型号 MH-LP100)",
         "content_type": "spec", "category": "商品规格手册"},
    ]


def test_upsert_then_dense_search_returns_fields(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.dense_search(c, _vec(0.98), top_k=1, collection=coll)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "运费怎么算"
    assert hits[0]["section_path"] == "商品与购物 FAQ / 运费怎么算"
    assert hits[0]["content_type"] == "faq"
    assert hits[0]["category"] == "商品与购物 FAQ"
    assert isinstance(hits[0]["score"], float)


def test_bm25_hits_model_number(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.bm25_search(c, "MH-LP100 废砂盒", top_k=2, collection=coll)
    assert hits and hits[0]["id"] == 2
    assert "MH-LP100" in hits[0]["section_path"]


def test_hybrid_returns_scored_hits(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.hybrid_search(c, _vec(1.0), "MH-LP100", top_k=2, recall=10,
                            collection=coll)
    assert hits
    assert {"id", "score", "question", "answer", "section_path",
            "content_type", "category"} <= hits[0].keys()


def test_category_filter(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.bm25_search(c, "运费 型号", top_k=5, category="商品与购物 FAQ",
                          collection=coll)
    assert hits
    assert all(h["category"] == "商品与购物 FAQ" for h in hits)


def test_dense_category_filter(milvus):
    """dense 路的 category 过滤不能只写在 bm25 路上。"""
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.dense_search(c, _vec(0.5), top_k=5, category="商品与购物 FAQ",
                           collection=coll)
    assert hits
    assert all(h["category"] == "商品与购物 FAQ" for h in hits)


def test_hybrid_category_filter(milvus):
    """hybrid 的两路 AnnSearchRequest 都要带 category 过滤,否则另一类会漏进来。"""
    c, coll = milvus
    mc.upsert_vectors(c, _rows(), collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.hybrid_search(c, _vec(0.5), "运费 型号 MH-LP100", top_k=5, recall=10,
                            category="商品规格手册", collection=coll)
    assert hits
    assert all(h["category"] == "商品规格手册" for h in hits)


def test_upsert_is_idempotent_by_pk(milvus):
    c, coll = milvus
    row = _rows()[0]
    mc.upsert_vectors(c, [row], collection=coll)
    mc.upsert_vectors(c, [row], collection=coll)
    mc.flush(c, collection=coll)
    assert mc.count(c, collection=coll) == 1


def test_upsert_truncates_overlong_text(milvus):
    c, coll = milvus
    long_answer = "满 99 包邮。" * 2000       # 远超 MAX_ANSWER(8192)
    mc.upsert_vectors(c, [{
        "id": 7, "dense": _vec(1.0), "text": "运费", "question": "运费怎么算",
        "answer": long_answer, "section_path": "运费", "content_type": "faq",
        "category": "运费",
    }], collection=coll)
    mc.flush(c, collection=coll)
    assert mc.count(c, collection=coll) == 1
    hits = mc.dense_search(c, _vec(1.0), top_k=1, collection=coll)
    assert len(hits[0]["answer"]) <= mc.MAX_ANSWER


def test_identical_vector_scores_near_one(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, [_rows()[0]], collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.dense_search(c, _vec(1.0), top_k=1, collection=coll)
    assert hits[0]["score"] > 0.99
