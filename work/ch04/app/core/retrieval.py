"""确定性检索核心(无 LLM):vector | bm25 | hybrid | hybrid_rerank。

Query 理解(口语归一/同义词扩展)在 query_faq 里做,不进入这里——否则检索段评估
就依赖 LLM,违反 spec §5「检索段与证据覆盖度不依赖 LLM」。评估脚本直接复用本模块。
"""
from app.config import settings
from app.core import embeddings, rerank
from app.kb import milvus_client

STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def arrange_head_tail(items: list) -> list:
    """最相关放首、次相关放尾,其余按序居中(缓解 lost-in-the-middle)。"""
    if len(items) <= 2:
        return list(items)
    return [items[0], *items[2:], items[1]]


async def search_knowledge(
    query: str,
    strategy: str = "hybrid_rerank",
    top_k: int | None = None,
    category: str | None = None,
    client=None,
    collection: str = milvus_client.COLLECTION,
    bm25_text: str | None = None,
) -> list[dict]:
    """确定性检索核心(无 LLM):vector | bm25 | hybrid | hybrid_rerank。

    - query:标准问法,dense 路用。
    - bm25_text:关键词路文本(可为 standard + 同义词扩展);缺省用 query。
    - 返回最终排序的 hit;hybrid_rerank 额外带 rerank_score。
    - 不做首尾组装,调用侧按需 arrange_head_tail。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知检索策略:{strategy}")
    # 显式 top_k=0 会回落到默认值;调用方须保证 ≥1(HTTP 层 ge=1 已挡)。
    top_k = top_k or settings.rerank_top_k
    c = client or milvus_client.get_client()
    milvus_client.ensure_collection(c, collection=collection)

    if strategy == "vector":
        vec = await embeddings.embed_query(query)
        return milvus_client.dense_search(c, vec, top_k, category=category,
                                          collection=collection)
    if strategy == "bm25":
        return milvus_client.bm25_search(c, bm25_text or query, top_k,
                                         category=category, collection=collection)

    vec = await embeddings.embed_query(query)
    hits = milvus_client.hybrid_search(
        c, vec, bm25_text or query, top_k, recall=settings.recall_top_k,
        category=category, collection=collection,
    )
    if strategy == "hybrid":
        return hits

    docs = [f"{h['question']} {h['answer']}" for h in hits]
    ranked = await rerank.rerank(query, docs, top_n=top_k)
    out = []
    for idx, score in ranked:
        h = dict(hits[idx])
        h["rerank_score"] = score
        out.append(h)
    return out
