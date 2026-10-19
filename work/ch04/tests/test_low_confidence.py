from sqlalchemy import select

from app.db import repository
from app.db.models import LowConfidenceQuestion


async def test_insert_low_confidence(db_session_factory):
    cid = await repository.create_conversation("u1")
    lid = await repository.insert_low_confidence(
        cid, "邮费到底多少啊", "self_check", "证据不足")
    assert lid > 0
    async with db_session_factory() as s:
        row = (await s.execute(
            select(LowConfidenceQuestion).where(LowConfidenceQuestion.id == lid)
        )).scalar_one()
    assert row.raw_question == "邮费到底多少啊"
    assert row.source == "self_check"
    assert row.reason == "证据不足"
    assert row.conversation_id == cid


async def test_insert_low_confidence_without_conversation(db_session_factory):
    lid = await repository.insert_low_confidence(
        None, "库外问题", "retrieval_low_conf", None)
    assert lid > 0
