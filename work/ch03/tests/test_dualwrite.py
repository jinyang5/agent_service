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
    mc.ensure_collection(c, collection=coll)
    yield c, coll
    mc.drop(c, collection=coll)


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
