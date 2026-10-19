"""后台首页聚合:知识库一张卡。当 Milvus/MySQL 没起时,只让它自己这张卡显示读数失败,不连坐整页。"""
from fastapi import APIRouter

from app.api import kb, rageval
from app.db import repository

router = APIRouter(prefix="/api/admin")


async def _kb_card() -> dict:
    card = {"key": "kb", "title": "知识库", "page": "/kb",
            "lede": "文档与对话挖出来的知识 → 切块 → 双写 MySQL 与 Milvus",
            "status": "error", "headline": "读数失败", "metrics": [], "note": None}
    try:
        stats = await repository.knowledge_stats()
    except Exception as e:
        card["note"] = f"{type(e).__name__}: {e}"
        return card
    milvus = await kb.milvus_state()
    card["metrics"] = [
        {"label": "知识块", "value": stats["total"]},
        {"label": "待向量化", "value": stats["pending"]},
        {"label": "Milvus", "value": milvus["count"] if milvus["online"] else "离线"},
        {"label": "关键条款", "value": stats["key_clause"]},
    ]
    if not stats["total"]:
        card["status"], card["headline"] = "missing", "库是空的,先录入或跑一次离线建库"
    elif not milvus["online"]:
        card["status"], card["headline"] = "attention", f"MySQL 有 {stats['total']} 块,Milvus 离线"
    elif stats["pending"] or stats["done"] != milvus["count"]:
        card["status"] = "attention"
        card["headline"] = (f"{stats['pending']} 块待向量化,"
                            f"MySQL 已向量化 {stats['done']} 对 Milvus {milvus['count']}")
    else:
        card["status"], card["headline"] = "ok", f"{stats['total']} 块双写一致,可被语义检索"
    return card


def _rag_eval_card_skeleton() -> dict:
    return {"key": "rag-eval", "title": "RAG 评估", "page": "/rag-eval",
            "lede": "四策略 × 四桶:检索、证据覆盖、忠实度与库外拒答全景",
            "status": "error", "headline": "读数失败", "metrics": [], "note": None}


async def _fill_rag_eval_card(card: dict) -> None:
    snapshot = rageval.report_snapshot()
    data = rageval.overview_payload(snapshot)

    if snapshot["kind"] == "missing":
        card["status"], card["headline"] = "missing", "还没跑过评估"
        card["note"] = "make eval-rag"
        return
    if snapshot["kind"] == "error":
        card["status"], card["headline"] = "error", "评估报告读不出来"
        card["note"] = snapshot["detail"]
        return

    meta = data["meta"] or {}
    generation = data["generation"] or {}
    # 合法 JSON 也可能形状漂移;这里不做防御式归一,让下面的隔离兜住并报 error
    refusal = generation.get("refusal", {}).get("D_offtopic", {})
    best = data["best"] or {}
    best_mrr = best.get("mrr")
    rate = refusal.get("rate")
    card["metrics"] = [
        {"label": "最佳 MRR",
         "value": round(best_mrr, 3) if best_mrr is not None else "—"},
        {"label": "评估题数", "value": meta.get("eval_set_size", "—")},
        {"label": "库外拒答",
         "value": round(rate, 3) if isinstance(rate, (int, float)) else "—"},
        {"label": "编造个案", "value": "—"},
    ]

    unresolved = None
    try:
        counts = await repository.faith_case_counts()
        unresolved = counts["未解决"]
        card["metrics"][3]["value"] = unresolved
    except Exception as e:                       # 台账库挂了:读数留空,不拖垮整卡
        card["note"] = f"个案台账读数失败:{type(e).__name__}: {e}"

    if not data["generation_done"]:
        card["status"], card["headline"] = "attention", "生成段未完成,只有检索与覆盖度"
    elif best.get("strategy") is not None:
        card["status"], card["headline"] = "ok", f"最佳 {best['strategy']} MRR {best_mrr:.3f}"
    else:
        card["status"], card["headline"] = "attention", "检索指标缺失"

    if card["note"] is None and unresolved:
        card["note"] = (f"{unresolved} 条编造个案未解决"
                        f"(共 {sum(counts.values())} 条)")


async def _rag_eval_card() -> dict:
    """RAG 评估卡:只读报告 + 台账未解决数。

    整张卡的构建都在隔离里:报告读坏、JSON 形状漂移、台账库挂了,都只让本卡
    显示 status=error,不冒泡成整页 500,更不连坐知识库卡。
    """
    card = _rag_eval_card_skeleton()
    try:
        await _fill_rag_eval_card(card)
    except Exception as e:
        card = _rag_eval_card_skeleton()
        card["note"] = f"{type(e).__name__}: {e}"
    return card


@router.get("/overview")
async def overview() -> dict:
    return {"modules": [await _kb_card(), await _rag_eval_card()]}
