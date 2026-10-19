from datetime import datetime

from sqlalchemy import select

from app.db import repository as repo
from app.db.models import Conversation, FaithCase, Faq, Ticket

async def test_create_and_get_conversation(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv is not None and conv.user_id == "u1"
    assert await repo.get_conversation(999999) is None

async def test_append_and_list_messages_in_order(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="订单 1001 到哪了")
    await repo.append_message(cid, "assistant", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    await repo.append_message(cid, "tool", content='{"status":"运输中"}', tool_call_id="c1")
    msgs = await repo.list_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[2].tool_call_id == "c1"

async def test_search_faq_like_hit_and_miss(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="退货政策", answer="7 天无理由退货", category="售后"))
        await s.commit()
    assert len(await repo.search_faq("退货")) == 1          # 命中
    assert await repo.search_faq("鞋子") == []              # 漏召回(字面不含)


async def test_search_faq_escapes_like_wildcards(db_session_factory, db_clean):
    # "%" 应作为字面量检索,而非通配符(否则会命中整表)
    async with db_session_factory() as s:
        s.add(Faq(question="退货政策", answer="7 天无理由退货", category="售后"))
        await s.commit()
    assert await repo.search_faq("%") == []

async def test_create_ticket_writes_and_flips_conversation_status(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    no = await repo.create_ticket(cid, "要退货", "售后")
    assert no.startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, no)
        assert t.ticket_type == "售后" and t.status == "待处理"
        conv = await s.get(Conversation, cid)
        assert conv.status == "已转人工"          # 会话状态流转


# ---------- ch04 RAG 评估页:编造个案台账 ----------

async def test_faith_case_listing_filter_and_counts(db_session_factory):
    for eval_id in ("A01", "A02", "A03"):
        await repo.upsert_faith_case(
            eval_id, "A_policy", f"q-{eval_id}", "hybrid_rerank", "a", "r", None, "m")
    await repo.set_faith_case_status("A02", "已解决", "补知识")
    await repo.set_faith_case_status("A03", "无需解决", "题面歧义")

    assert await repo.faith_case_counts() == {"未解决": 1, "已解决": 1, "无需解决": 1}
    rows = await repo.list_faith_cases(status="已解决")
    assert [r.eval_id for r in rows] == ["A02"]
    assert len(await repo.list_faith_cases()) == 3
    assert len(await repo.list_faith_cases(status="未解决")) == 1


async def test_set_faith_case_status_keeps_resolved_at_and_reopens(db_session_factory):
    assert await repo.get_faith_case("NOPE") is None
    assert await repo.set_faith_case_status("NOPE", "已解决", "x") is None

    await repo.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a", "r", None, "m")
    row = await repo.set_faith_case_status("A01", "已解决", "补了知识")
    assert row is not None
    assert row.status == "已解决" and row.resolution == "补了知识" and row.resolved_at

    # 回填成过去时间:后续任何操作都不许覆盖
    past = datetime(2026, 1, 2, 3, 4, 5)
    async with db_session_factory() as s:
        stored = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == "A01")
        )).scalar_one()
        stored.resolved_at = past
        await s.commit()

    # 同值重复提交:MySQL rowcount=0,不许误判 404
    same = await repo.set_faith_case_status("A01", "已解决", "补了知识")
    assert same is not None and same.resolved_at == past

    row = await repo.set_faith_case_status("A01", "无需解决", "题面歧义")
    assert row.resolved_at == past                   # 终态切换不覆盖首次处置时间

    row = await repo.set_faith_case_status("A01", "未解决", None)
    assert row.status == "未解决"
    assert row.resolution is None                    # 重开清空处置说明
    assert row.resolved_at == past                   # 保留,用来标「复发」
