"""后台首页聚合:知识库一张卡。当 Milvus/MySQL 没起时,只让它自己这张卡显示读数失败,不连坐整页。"""
from fastapi import APIRouter

from app.api import kb
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


@router.get("/overview")
async def overview() -> dict:
    return {"modules": [await _kb_card()]}
