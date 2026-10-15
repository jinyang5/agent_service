"""标注样例评估:核对上游模型是否按预期选中工具。需服务运行中。"""
import asyncio

import httpx

BASE = "http://localhost:8000"

# (用户问法, 期望命中的工具名集合之一;None 表示期望不调用工具)
SAMPLES = [
    ("订单 1001 的物流到哪了", {"query_logistics"}),
    ("退货政策是什么", {"query_faq"}),
    ("邮费是多少", {"query_faq"}),                 # 关注漏召回:答案在「运费怎么算」但字面 LIKE 对不上
    ("iPhone 还有货吗", {"query_product"}),
    ("订单 2002 多少钱", {"query_order"}),
    ("我要投诉,给我登记一下", {"create_ticket"}),
    ("今天天气怎么样", None),                       # 超范围,期望不调工具
]

async def main() -> None:
    passed = 0
    async with httpx.AsyncClient(timeout=60) as client:
        for msg, expect in SAMPLES:
            r = await client.post(f"{BASE}/api/agent", json={"user_id": "eval", "message": msg})
            body = r.json()
            names = {tc["name"] for tc in body["tool_calls"]}
            ok = (not names) if expect is None else bool(names & expect)
            passed += ok
            note = ""
            if msg == "邮费是多少":
                faq = [tr for tr in body["tool_results"] if tr["name"] == "query_faq"]
                if not faq:
                    note = " [未调用 query_faq]"
                else:
                    miss = '"hits": []' in faq[0]["content"]
                    note = f" [漏召回={'是' if miss else '否'}]"
            print(f"{'✅' if ok else '❌'} {msg!r} -> {names or '(未调用)'} 期望={expect}{note}")
            print(f"    answer: {body['answer'][:60]}")
    print(f"\n通过 {passed}/{len(SAMPLES)}")

if __name__ == "__main__":
    asyncio.run(main())
