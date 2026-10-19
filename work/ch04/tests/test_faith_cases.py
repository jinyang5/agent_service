from datetime import datetime

from sqlalchemy import select

from app.db import repository
from app.db.models import FaithCase

CITES = [{"n": 1, "chunk_id": 5, "section_path": "运费",
          "question": "运费怎么算", "answer": "满 99 包邮"}]


async def test_upsert_new_row(db_session_factory):
    r = await repository.upsert_faith_case(
        "A01", "A_policy", "满多少包邮", "hybrid_rerank",
        "答案", "编造了数值", CITES, "deepseek-v4-flash")
    assert r["seen_count"] == 1
    assert r["relapsed"] is False
    assert r["status"] == "未解决"
    async with db_session_factory() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == "A01")
        )).scalar_one()
    assert row.citations[0]["chunk_id"] == 5
    assert row.answer == "答案"


async def test_recurrence_updates_one_row(db_session_factory):
    first = await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a1", "r1", CITES, "m")
    second = await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a2", "r2", CITES, "m")
    assert second["id"] == first["id"]
    assert second["seen_count"] == 2
    assert second["relapsed"] is False
    rows = await repository.list_faith_cases()
    assert len(rows) == 1 and rows[0].answer == "a2"


async def test_resolved_then_recurrence_resets_status(db_session_factory):
    r = await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a1", "r1", CITES, "m")
    async with db_session_factory() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.id == r["id"])
        )).scalar_one()
        row.status = "已解决"
        row.resolution = "补充了知识"
        row.resolved_at = datetime.now()
        await s.commit()

    again = await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a2", "r2", CITES, "m")
    assert again["relapsed"] is True
    assert again["status"] == "未解决"
    assert again["seen_count"] == 2
    async with db_session_factory() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.id == r["id"])
        )).scalar_one()
    assert row.resolved_at is not None          # 复发后仍保留,用来标「复发」
    assert row.resolution is None               # DDL:退回未解决时清空
