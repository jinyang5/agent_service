from app.db.models import Conversation, Ticket
from app.tools.business import create_ticket


def test_conversation_id_hidden_from_model_schema():
    # 断言的是「暴露给 LLM 的 schema」= tool_call_schema。
    # get_input_schema() 是内部全量 schema,会(刻意)包含被注入的参数。
    schema = create_ticket.tool_call_schema.model_json_schema()
    props = schema.get("properties", {})
    assert "description" in props and "ticket_type" in props
    assert "conversation_id" not in props        # 关键:模型看不到会话主键


async def test_create_ticket_injects_conversation_id_and_writes(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id
    r = await create_ticket.ainvoke(
        {"description": "商品损坏要退货", "ticket_type": "售后", "conversation_id": cid}
    )
    assert r["ticket_no"].startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, r["ticket_no"])
        assert t is not None and t.conversation_id == cid
