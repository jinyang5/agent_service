from sqlalchemy import select

from app.db.models import KnowledgeChunk, QaExtractionStaging


async def test_knowledge_chunk_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = KnowledgeChunk(category="售后政策", questions="运费说明", answer="满99包邮")
        s.add(row)
        await s.commit()
        await s.refresh(row)
        assert row.id is not None
        assert row.vectorize_status == "pending"
        assert row.is_key_clause == 0


async def test_defaults_readable_after_commit_without_refresh(db_session_factory):
    """MySQL 没有 INSERT ... RETURNING:新建 mapper 必须带 eager_defaults=True,
    否则 commit 后直接读服务端默认列会触发 greenlet 之外的惰性加载 → MissingGreenlet。
    """
    async with db_session_factory() as s:
        row = KnowledgeChunk(category="售后政策", questions="运费说明", answer="满99包邮")
        s.add(row)
        await s.commit()
        # 故意不 refresh:三个服务端默认列的值必须已随 INSERT 取回
        assert row.vectorize_status == "pending"
        assert row.is_key_clause == 0
        assert row.created_at is not None


async def test_staging_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = QaExtractionStaging(batch_no="b1", question="邮费多少", answer="满99包邮")
        s.add(row)
        await s.commit()
        got = (await s.execute(select(QaExtractionStaging))).scalars().all()
        assert len(got) == 1
        assert got[0].status == "extracted"
