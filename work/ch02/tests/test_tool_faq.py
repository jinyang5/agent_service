from app.db.models import Faq
from app.tools.business import query_faq


async def test_query_faq_hit(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="退货政策", answer="7 天无理由", category="售后"))
        await s.commit()
    r = await query_faq.ainvoke({"keyword": "退货"})
    assert r["hits"] and r["hits"][0]["question"] == "退货政策"


async def test_query_faq_miss_returns_message(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="退货政策", answer="7 天无理由", category="售后"))
        await s.commit()
    r = await query_faq.ainvoke({"keyword": "鞋子"})
    assert r["hits"] == [] and "未找到" in r["message"]
