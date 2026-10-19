"""本地重排:进程内 FlagReranker(BAAI/bge-reranker-v2-m3),CPU 推理。

模型只在首次调用时加载(惰性单例);加载与 compute_score 都是阻塞的,
统一丢进 asyncio.to_thread,别卡住事件循环。
"""
import asyncio

from app.config import settings

_MODEL = None


def _model():
    """惰性单例。供测试 monkeypatch 成带 compute_score 的假对象。"""
    global _MODEL
    if _MODEL is None:
        from FlagEmbedding import FlagReranker

        _MODEL = FlagReranker(
            settings.rerank_model,
            use_fp16=settings.rerank_device != "cpu",
        )
    return _MODEL


def _score_pairs(query: str, docs: list[str]) -> list[float]:
    pairs = [[query, d] for d in docs]
    scores = _model().compute_score(pairs, normalize=True)
    if isinstance(scores, (int, float)):
        return [float(scores)]
    return [float(s) for s in scores]


async def rerank(query: str, docs: list[str],
                 top_n: int | None = None) -> list[tuple[int, float]]:
    """返回 [(原始索引, 相关分)] 按分降序,截断 top_n。空 docs 直接返回 []。
    normalize=True 的分落在 0~1,与 settings.rerank_min_score 同一个量纲。"""
    if not docs:
        return []
    scores = await asyncio.to_thread(_score_pairs, query, docs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n] if top_n is not None else ranked
