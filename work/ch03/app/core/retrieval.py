from app.config import settings
from app.core import embeddings
from app.kb import milvus_client


async def search_knowledge(
    query: str, top_k: int | None = None, min_score: float | None = None,
    client=None, collection: str = milvus_client.COLLECTION,
) -> list[dict]:
    """dense 单路:embed(query) → Milvus Top-K → 按 min_score 过滤。
    Standalone COSINE 的 score 即相似度(越大越像),故直接 score >= min_score。"""
    top_k = top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if min_score is None else min_score
    vector = await embeddings.embed_query(query)
    c = client or milvus_client.get_client()
    milvus_client.ensure_collection(c, collection=collection)
    hits = milvus_client.search(c, vector, top_k, collection=collection)
    return [h for h in hits if h["score"] >= min_score]
