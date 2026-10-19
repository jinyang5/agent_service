from app.core import embeddings
from app.db import repository
from app.kb import milvus_client
from app.kb.documents import Chunk


async def write_pending(chunks: list[Chunk]) -> list[int]:
    """按顺序把一份文档的 chunks 写入 MySQL(status=pending),并在同文档内连 prev/next 指针。"""
    ids: list[int] = []
    for c in chunks:
        cid = await repository.insert_knowledge_chunk(
            c.category, c.questions, c.answer,
            section_path=c.section_path, content_type=c.content_type,
            is_key_clause=c.is_key_clause,
        )
        ids.append(cid)
    for i, cid in enumerate(ids):
        prev_id = ids[i - 1] if i > 0 else None
        next_id = ids[i + 1] if i < len(ids) - 1 else None
        await repository.set_chunk_neighbors(cid, prev_id, next_id)
    return ids


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def vectorize_pending(client=None, batch_size: int = 64,
                            collection: str = milvus_client.COLLECTION) -> int:
    """幂等可重跑:取 pending → 拼 category+questions+answer → 嵌入 + BM25 text
    → Milvus upsert(PK=id)→ flush → 回填 vector_id、status=done。
    崩在任意批,重跑只捡剩余 pending(Milvus 按 id upsert 无重)。"""
    pending = await repository.list_pending_chunks()
    done = 0
    c = client or milvus_client.get_client()
    milvus_client.ensure_collection(c, collection=collection)
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        rows = [
            {
                "id": r.id, "dense": v, "text": t,
                "question": r.questions, "answer": r.answer,
                "section_path": r.section_path or "",
                "content_type": r.content_type or "",
                "category": r.category or "",
            }
            for r, v, t in zip(batch, vectors, texts)
        ]
        milvus_client.upsert_vectors(c, rows, collection=collection)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    if done:
        milvus_client.flush(c, collection=collection)
    return done
