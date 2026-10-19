# app/db/repository.py
from datetime import datetime

from sqlalchemy import func, select, update

import app.db.base as db          # 用模块属性引用,便于测试 monkeypatch async_session
from app.db.models import (
    Conversation,
    FaithCase,
    Faq,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    Ticket,
)

_TICKET_SEQ = 0

def _gen_ticket_no() -> str:
    global _TICKET_SEQ
    _TICKET_SEQ += 1
    return f"T{datetime.now():%Y%m%d%H%M%S}{_TICKET_SEQ:03d}"

async def create_conversation(user_id: str) -> int:
    async with db.async_session() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        await s.commit()
        return conv.id

async def get_conversation(conversation_id: int) -> Conversation | None:
    async with db.async_session() as s:
        return await s.get(Conversation, conversation_id)

async def append_message(
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with db.async_session() as s:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        s.add(msg)
        await s.commit()
        return msg.id

async def list_messages(conversation_id: int) -> list[Message]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())

def _escape_like(keyword: str) -> str:
    """转义 LIKE 通配符,避免用户输入被当模式(如 "%" 匹配整表)。"""
    return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def search_faq(keyword: str) -> list[Faq]:
    async with db.async_session() as s:
        pattern = f"%{_escape_like(keyword)}%"
        result = await s.execute(
            select(Faq).where(Faq.question.like(pattern, escape="\\"))
        )
        return list(result.scalars())

async def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
    ticket_no = _gen_ticket_no()
    async with db.async_session() as s:
        s.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "已转人工"
        await s.commit()
    return ticket_no


async def insert_knowledge_chunk(
    category: str, questions: str, answer: str,
    section_path: str | None = None, content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session() as s:
        row = KnowledgeChunk(
            category=category, questions=questions, answer=answer,
            section_path=section_path, content_type=content_type,
            is_key_clause=is_key_clause,
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"
            await s.commit()


async def set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id
            await s.commit()


async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())


async def list_all_questions() -> list[str]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions))
        return list(result.scalars())


async def insert_staging(batch_no: str, source_ref: str | None, question: str, answer: str) -> int:
    async with db.async_session() as s:
        row = QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())


async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(QaExtractionStaging, i)
            if row is not None:
                row.status = status
        await s.commit()


async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    async with db.async_session() as s:
        conv_ids = list((await s.execute(select(Conversation.id).order_by(Conversation.id))).scalars())
        out: list[tuple[int, list[Message]]] = []
        for cid in conv_ids:
            msgs = list((await s.execute(
                select(Message).where(Message.conversation_id == cid).order_by(Message.id)
            )).scalars())
            out.append((cid, msgs))
        return out


async def knowledge_stats() -> dict:
    """知识库盘点:总块数 + 双写状态分布 + 内容类型分布 + 关键条款数。"""
    async with db.async_session() as s:
        total = (await s.execute(select(func.count()).select_from(KnowledgeChunk))).scalar_one()
        by_status = {k: int(v) for k, v in (await s.execute(
            select(KnowledgeChunk.vectorize_status, func.count())
            .group_by(KnowledgeChunk.vectorize_status))).all()}
        by_type = {(k or "未标注"): int(v) for k, v in (await s.execute(
            select(KnowledgeChunk.content_type, func.count())
            .group_by(KnowledgeChunk.content_type))).all()}
        key_clause = (await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.is_key_clause == 1))).scalar_one()
    return {"total": int(total), "pending": by_status.get("pending", 0),
            "done": by_status.get("done", 0), "by_content_type": by_type,
            "key_clause": int(key_clause)}


async def list_recent_chunks(limit: int = 20) -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit))
        return list(result.scalars())


async def list_chunk_pairs() -> list[tuple[str, str]]:
    """全表的 (questions, answer) 对,给录入查重算指纹(只按问法会误杀同节多块)。"""
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions, KnowledgeChunk.answer))
        return [(q, a) for q, a in result.all()]


async def staging_stats() -> dict:
    """抽 QA 暂存表盘点:三个状态各几条 + 批次数 + 最近批次号。"""
    async with db.async_session() as s:
        counts = {k: int(v) for k, v in (await s.execute(
            select(QaExtractionStaging.status, func.count())
            .group_by(QaExtractionStaging.status))).all()}
        batches = (await s.execute(
            select(func.count(func.distinct(QaExtractionStaging.batch_no))))).scalar_one()
    return {"counts": {k: counts.get(k, 0) for k in ("extracted", "kept", "discarded")},
            "total": sum(counts.values()), "batches": int(batches)}


async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None
) -> int:
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


async def upsert_faith_case(
    eval_id: str, bucket: str, query: str, strategy: str, answer: str,
    reason: str, citations: list[dict] | None, judge_model: str | None,
) -> dict:
    """一题一行:新题插入;复发只更新并 seen_count+1,已处置的退回「未解决」。"""
    async with db.async_session() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id)
        )).scalar_one_or_none()
        now = datetime.now()
        if row is None:
            row = FaithCase(
                eval_id=eval_id, bucket=bucket, query=query, strategy=strategy,
                answer=answer, reason=reason, citations=citations,
                judge_model=judge_model, seen_count=1,
                first_seen_at=now, last_seen_at=now,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return {"id": row.id, "seen_count": row.seen_count,
                    "relapsed": False, "status": row.status}

        relapsed = row.status != "未解决"
        row.bucket = bucket
        row.query = query
        row.strategy = strategy
        row.answer = answer
        row.reason = reason
        row.citations = citations
        row.judge_model = judge_model
        row.seen_count = (row.seen_count or 0) + 1
        row.last_seen_at = now
        if relapsed:
            row.status = "未解决"
            row.resolution = None  # DDL 注释:退回未解决时清空;resolved_at 保留,用来标复发
        await s.commit()
        await s.refresh(row)
        return {"id": row.id, "seen_count": row.seen_count,
                "relapsed": relapsed, "status": row.status}


async def list_faith_cases(limit: int = 100, status: str | None = None) -> list[FaithCase]:
    async with db.async_session() as s:
        stmt = select(FaithCase)
        if status is not None:
            stmt = stmt.where(FaithCase.status == status)
        result = await s.execute(
            stmt
            .order_by(FaithCase.last_seen_at.desc(), FaithCase.id.desc())
            .limit(limit)
        )
        return list(result.scalars())


async def faith_case_counts() -> dict[str, int]:
    """台账三个状态各几条;没有的状态补 0,前端直接当角标用。"""
    async with db.async_session() as s:
        rows = (await s.execute(
            select(FaithCase.status, func.count()).group_by(FaithCase.status))).all()
    counts = {k: 0 for k in ("未解决", "已解决", "无需解决")}
    for status, n in rows:
        if status in counts:
            counts[status] = int(n)
    return counts


async def get_faith_case(eval_id: str) -> FaithCase | None:
    async with db.async_session() as s:
        return (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id)
        )).scalar_one_or_none()


async def set_faith_case_status(
    eval_id: str, status: str, resolution: str | None = None
) -> FaithCase | None:
    """人工处置一题:终态写处置说明与首次处置时间;重开清说明但保留 resolved_at。

    单条 UPDATE 完成状态流转,避免 SELECT→改→commit 的并发丢更新。
    resolved_at 用 COALESCE 只在原本为空时写 now:已处置的题再次被标终态,
    保留首次处置时间;退回「未解决」不碰它——它和 status 一起构成「复发」标记。
    """
    values: dict = {
        "status": status,
        "resolution": None if status == "未解决" else resolution,
    }
    if status != "未解决":
        values["resolved_at"] = func.coalesce(FaithCase.resolved_at, datetime.now())

    async with db.async_session() as s:
        result = await s.execute(
            update(FaithCase).where(FaithCase.eval_id == eval_id).values(**values))
        if result.rowcount == 0:
            # MySQL 的 rowcount 是「实际变更行数」:同值重复提交也是 0,别误判 404
            exists = (await s.execute(
                select(FaithCase.id).where(FaithCase.eval_id == eval_id)
            )).scalar_one_or_none()
            if exists is None:
                return None
        await s.commit()
        return (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id)
        )).scalar_one_or_none()
