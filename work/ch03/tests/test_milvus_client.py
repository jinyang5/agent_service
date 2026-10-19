import uuid

import pytest

from app.kb import milvus_client as mc


@pytest.fixture()
def milvus():
    """真实 Milvus Standalone 上的独立测试集合;连不上就 skip(先 make milvus-up)。"""
    try:
        c = mc.get_client()
        c.list_collections()
    except Exception as e:  # 连不上是「环境没起」,不是被测逻辑错
        pytest.skip(f"Milvus 不可用,先 make milvus-up:{e}")
    coll = f"knowledge_test_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(c, collection=coll)
    yield c, coll
    mc.drop(c, collection=coll)


def _vec(seed: float) -> list[float]:
    v = [0.0] * mc.DIM
    v[0] = seed
    v[1] = 1.0 - seed
    return v


def test_upsert_then_search_returns_fields(milvus):
    c, coll = milvus
    mc.upsert_vectors(c, [
        {"id": 1, "vector": _vec(1.0), "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": _vec(0.0), "question": "发货时效", "answer": "48小时内发货"},
    ], collection=coll)
    mc.flush(c, collection=coll)
    hits = mc.search(c, _vec(0.98), top_k=1, collection=coll)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "运费怎么算"
    assert hits[0]["answer"] == "满99包邮"
    assert isinstance(hits[0]["score"], float)


def test_upsert_is_idempotent_by_pk(milvus):
    c, coll = milvus
    row = {"id": 1, "vector": _vec(1.0), "question": "q", "answer": "a"}
    mc.upsert_vectors(c, [row], collection=coll)
    mc.upsert_vectors(c, [row], collection=coll)   # 同 id 重写
    mc.flush(c, collection=coll)
    assert mc.count(c, collection=coll) == 1


def test_upsert_truncates_overlong_text(milvus):
    """Review Focus 2:超 max_length 的 question/answer 要裁剪后再 upsert,不许整批被 Milvus 拒。"""
    c, coll = milvus
    long_answer = "满99包邮。" * 2000        # 10000 字 > MAX_ANSWER(8192)
    mc.upsert_vectors(c, [
        {"id": 7, "vector": _vec(1.0), "question": "运费怎么算", "answer": long_answer},
    ], collection=coll)
    mc.flush(c, collection=coll)
    assert mc.count(c, collection=coll) == 1
    hits = mc.search(c, _vec(1.0), top_k=1, collection=coll)
    assert len(hits[0]["answer"]) <= mc.MAX_ANSWER
