"""知识库录入 API:材料清单、切块预览、录入入库、向量化、检索自测。

手工录入(POST /ingest):正文在浏览器里贴,预览什么样就入库什么样——切块用
app.kb.documents.build_chunks,与离线 CLI 同一个函数。查重按「问法 + 正文」的指纹
(app.kb.dedup.fingerprint,离线建库的幂等守卫共用这一个定义),
同一份正文重复录入天然幂等(指纹全撞、全跳过)。"""
import datetime as dt
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core import jobs, retrieval
from app.db import repository
from app.kb import chunking, dedup, documents, dualwrite, milvus_client
from app.kb.sources import CONTENT_TYPE_DESC, CONTENT_TYPES, KB_DIR, SOURCE_TYPES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb")

MAX_TEXT_CHARS = 40000


def _chunk_view(seq: int, c: documents.Chunk, dup: bool | None) -> dict:
    return {"seq": seq, "section_path": c.section_path or "(无标题层级)", "category": c.category,
            "questions": c.questions, "answer": c.answer, "chars": len(c.answer),
            "is_key_clause": bool(c.is_key_clause), "is_table": chunking.is_table_block(c.answer),
            "duplicate": dup}


def _features(chunks: list[documents.Chunk]) -> dict:
    groups: dict[str, list[documents.Chunk]] = {}
    for c in chunks:
        groups.setdefault(c.section_path, []).append(c)
    table_split = any(len(g) > 1 and chunking.is_table_block(g[0].answer) for g in groups.values())
    overlap = any(len(g) > 1 and not chunking.is_table_block(g[0].answer) for g in groups.values())
    return {"sections": len(groups), "table_split": table_split, "overlap": overlap,
            "multi_piece_sections": [p for p, g in groups.items() if len(g) > 1]}


async def _existing_fingerprints() -> set[str] | None:
    """库里已有的指纹集合;库连不上回 None——预览照样能看,只是查重那列显示未知。"""
    try:
        return dedup.fingerprints(await repository.list_chunk_pairs())
    except Exception:
        return None


async def milvus_state() -> dict:
    """Milvus 探活 + 条数。连不上不是错误,是一种状态:页面显示离线,双写一致那一栏不下结论。"""
    try:
        c = milvus_client.get_client()
        milvus_client.ensure_collection(c)
        return {"online": True, "count": milvus_client.count(c), "collection": milvus_client.COLLECTION}
    except Exception as e:
        return {"online": False, "count": None, "detail": f"{type(e).__name__}: {e}"}


def _sources() -> list[dict]:
    out = []
    for fname, ctype in SOURCE_TYPES.items():
        path = KB_DIR / fname
        item = {"file": fname, "content_type": ctype, "present": path.exists(),
                "path": f"data/kb/{fname}"}
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            chunks = documents.build_chunks(raw, content_type=ctype)
            st = path.stat()
            item.update({"chars": len(raw), "lines": raw.count("\n") + 1,
                         "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                         "chunks": len(chunks), "key_clause": sum(c.is_key_clause for c in chunks),
                         "features": _features(chunks)})
        out.append(item)
    return out


@router.get("/overview")
async def overview() -> dict:
    """录入页总览:库存 + 双写一致 + 暂存表 + 材料清单 + 作业状态。"""
    milvus = await milvus_state()
    try:
        stats = await repository.knowledge_stats()
        recent = [{"id": r.id, "questions": r.questions, "answer": r.answer[:120],
                   "category": r.category, "section_path": r.section_path,
                   "content_type": r.content_type, "is_key_clause": bool(r.is_key_clause),
                   "status": r.vectorize_status,
                   "created_at": r.created_at.isoformat(timespec="seconds") if r.created_at else None}
                  for r in await repository.list_recent_chunks(12)]
        staging = await repository.staging_stats()
        db_err = None
    except Exception as e:                     # mysql 没起时不连坐整页
        stats = {"total": None, "pending": None, "done": None, "by_content_type": {}, "key_clause": None}
        recent, staging, db_err = [], None, f"{type(e).__name__}: {e}"

    # 双写一致 = 没有块卡在 pending,且 MySQL 已向量化数与 Milvus 条数对得上。
    # 任一边读不到就是 None(不知道),不当成不一致——「读不到」和「对不上」是两回事。
    consistent = None
    if db_err is None and milvus["online"]:
        consistent = stats["pending"] == 0 and stats["done"] == milvus["count"]

    return {"chunks": stats, "recent": recent, "staging": staging, "db_error": db_err,
            "milvus": milvus, "consistent": consistent, "sources": _sources(),
            "content_types": [{"key": k, "desc": CONTENT_TYPE_DESC[k]} for k in CONTENT_TYPES],
            "jobs": jobs.status_all()}


class PreviewIn(BaseModel):
    text: str | None = None
    file: str | None = None                    # 只能是材料清单里的文件名,不接受路径
    content_type: str = "faq"


def _resolve_preview(body: PreviewIn) -> tuple[str, str, str]:
    """回 (正文, content_type, 来源标注)。file 走白名单精确匹配,传不进任何路径。"""
    if body.file:
        if body.file not in SOURCE_TYPES:
            raise HTTPException(status_code=400, detail=f"不在建库材料清单里:{body.file}")
        path = KB_DIR / body.file
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"材料文件不存在:data/kb/{body.file}")
        return path.read_text(encoding="utf-8"), SOURCE_TYPES[body.file], f"data/kb/{body.file}"
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="正文是空的,贴一段 Markdown 再预览")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"正文 {len(text)} 字,超过单次上限 {MAX_TEXT_CHARS} 字,拆开录")
    if body.content_type not in CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"content_type 只能是 {'/'.join(CONTENT_TYPES)}")
    return text, body.content_type, "手工录入"


@router.post("/preview")
async def preview(body: PreviewIn) -> dict:
    """切块预览(dry-run):不写库、不碰 Milvus、不调上游。"""
    text, ctype, source = _resolve_preview(body)
    chunks = documents.build_chunks(text, content_type=ctype)
    seen = await _existing_fingerprints()
    views, dups, batch = [], 0, set()
    for i, c in enumerate(chunks, 1):
        fp = dedup.fingerprint(c.questions, c.answer)
        dup = None if seen is None else (fp in seen or fp in batch)
        dups += bool(dup)
        batch.add(fp)
        views.append(_chunk_view(i, c, dup))
    return {"source": source, "content_type": ctype, "chars": len(text), "total": len(chunks),
            "duplicates": dups, "key_clause": sum(c.is_key_clause for c in chunks),
            "features": _features(chunks), "chunks": views, "dedup_known": seen is not None}


class IngestIn(BaseModel):
    text: str
    content_type: str = "faq"
    vectorize: bool = True


@router.post("/ingest")
async def ingest(body: IngestIn) -> dict:
    """录入:切块 → 按指纹查重 → 写 MySQL(pending)→ 可选立刻向量化。
    先写 MySQL 再进 Milvus 的顺序不能反:MySQL 是原文权威源,挂了重跑捡 pending 补齐。"""
    text, ctype, _ = _resolve_preview(PreviewIn(text=body.text, content_type=body.content_type))
    chunks = documents.build_chunks(text, content_type=ctype)
    if not chunks:
        raise HTTPException(status_code=400, detail="这段正文切不出块,检查是不是只有标题没有正文")
    seen = await _existing_fingerprints()
    if seen is None:
        raise HTTPException(status_code=503, detail="连不上 MySQL,录入这条路走不通(查重与落库都要它)")

    kept, skipped = [], []
    for c in chunks:
        fp = dedup.fingerprint(c.questions, c.answer)
        if fp in seen:
            skipped.append({"questions": c.questions, "answer": c.answer[:80]})
            continue
        seen.add(fp)
        kept.append(c)

    ids = await dualwrite.write_pending(kept) if kept else []
    vectorized = None
    if body.vectorize and ids:
        try:
            vectorized = await dualwrite.vectorize_pending()
        except Exception as e:      # 向量化失败不回滚:块已在 MySQL 记 pending,补跑一次就齐
            raise HTTPException(
                status_code=502,
                detail=(f"已入库 {len(ids)} 块(pending),向量化失败:{type(e).__name__}: {e}。"
                        "修好嵌入/Milvus 后按「向量化待补块」补齐,不必重录")) from e
    return {"content_type": ctype, "chunks": len(chunks), "inserted": len(ids), "skipped": len(skipped),
            "skipped_samples": skipped[:5], "ids": ids, "vectorized": vectorized,
            "milvus": await milvus_state(), "chunk_stats": await repository.knowledge_stats()}


@router.post("/vectorize")
async def vectorize() -> dict:
    """把库里所有 pending 块补齐向量(幂等可重跑)。与 make kb-vectorize 同一个函数。"""
    try:
        n = await dualwrite.vectorize_pending()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"向量化失败:{type(e).__name__}: {e}") from e
    return {"vectorized": n, "chunk_stats": await repository.knowledge_stats(),
            "milvus": await milvus_state()}


class SearchIn(BaseModel):
    q: str
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/search")
async def search(body: SearchIn) -> dict:
    """检索自测(dense 单路):换个说法问一句,看召回的是不是该召回的那块。"""
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="问一句话再检索")
    try:
        hits = await retrieval.search_knowledge(q, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"检索失败({type(e).__name__}: {e});Milvus 要在线") from e
    return {"q": q, "top_k": body.top_k,
            "hits": [{"id": h.get("id"), "question": h.get("question"), "answer": h.get("answer"),
                      "score": h.get("score")} for h in hits]}


@router.get("/staging")
async def staging(limit: int = 30) -> dict:
    """抽 QA 暂存表:分批抽出 → 整体去重 → kept 待人审。"""
    stats = await repository.staging_stats()
    rows = {}
    for st in ("extracted", "kept", "discarded"):
        rows[st] = [{"id": r.id, "batch_no": r.batch_no, "source_ref": r.source_ref,
                     "question": r.question, "answer": r.answer[:160]}
                    for r in (await repository.list_staging_by_status(st))[:limit]]
    return {"stats": stats, "rows": rows}
