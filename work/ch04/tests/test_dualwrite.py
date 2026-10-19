import uuid

import pytest

from app.db import repository
from app.kb import dualwrite
from app.kb import milvus_client as mc
from app.kb.documents import Chunk


def _chunk(q, a):
    return Chunk(category="c", questions=q, answer=a, section_path="c / " + q, content_type="policy")


@pytest.fixture()
def milvus():
    try:
        c = mc.get_client()
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


async def test_write_pending_links_neighbors(db_session_factory):
    ids = await dualwrite.write_pending([_chunk("q1", "a1"), _chunk("q2", "a2"), _chunk("q3", "a3")])
    assert len(ids) == 3
    by_id = {c.id: c for c in await repository.list_pending_chunks()}
    assert by_id[ids[1]].prev_chunk_id == ids[0]
    assert by_id[ids[1]].next_chunk_id == ids[2]
    assert by_id[ids[0]].prev_chunk_id is None


async def test_vectorize_resumes_after_crash(db_session_factory, milvus, monkeypatch):
    c, coll = milvus
    await dualwrite.write_pending([_chunk(f"q{i}", f"a{i}") for i in range(4)])

    # 第一趟:embed 在第 2 批抛错(batch_size=2 → 前 2 条 done,后 2 条仍 pending)
    calls = {"n": 0}

    async def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("嵌入中断")
        return [[float(i), 0.0, 1.0] + [0.0] * (mc.DIM - 3) for i, _ in enumerate(texts)]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", flaky_embed)
    with pytest.raises(RuntimeError):
        await dualwrite.vectorize_pending(c, batch_size=2, collection=coll)
    assert await repository.count_chunks_by_status("done") == 2
    assert await repository.count_chunks_by_status("pending") == 2

    # 第二趟:恢复正常 → 只捡剩下 2 条 pending,补齐;Milvus 无重(按 id upsert)
    async def ok_embed(texts):
        return [[9.0, 0.0, 1.0] + [0.0] * (mc.DIM - 3) for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok_embed)
    done_now = await dualwrite.vectorize_pending(c, batch_size=2, collection=coll)
    assert done_now == 2
    assert await repository.count_chunks_by_status("pending") == 0
    assert await repository.count_chunks_by_status("done") == 4
    assert mc.count(c, collection=coll) == 4  # 无重无漏


async def test_vectorize_writes_text_and_metadata(db_session_factory, milvus, monkeypatch):
    c, coll = milvus

    async def fake_embed(texts):
        return [[0.05] * mc.DIM for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", fake_embed)
    await dualwrite.write_pending([
        Chunk(category="运费", questions="运费怎么算", answer="满 99 元包邮",
              section_path="商品与购物 FAQ / 运费怎么算", content_type="faq"),
    ])
    done = await dualwrite.vectorize_pending(c, collection=coll)
    assert done == 1
    hits = mc.bm25_search(c, "运费 包邮", top_k=1, collection=coll)
    assert hits and hits[0]["section_path"] == "商品与购物 FAQ / 运费怎么算"
    assert hits[0]["category"] == "运费"
