"""CLI:把 data/kb/*.md 切块写入 knowledge_chunks(pending)。之后跑 vectorize_kb.py。
幂等:与录入(POST /api/kb/ingest)共用 app.kb.dedup 的「问法 + 正文」指纹,逐块跳过已在库的。
运行:PYTHONPATH=. uv run python scripts/build_kb.py(或在录入页 /kb 上按「离线建库」)"""
import asyncio

from app.db import repository
from app.kb import dedup, documents, dualwrite
from app.kb.sources import KB_DIR, SOURCE_TYPES as DOCS


async def main() -> None:
    # 幂等守卫:与录入同源的指纹(问法 + 正文归一化),逐块比对——已有的跳过,缺的补上。
    # 不能只按 content_type 计块:页面手工录过一段 faq/policy,整批就会被当成「文档已建」而拒建,
    # 冷库里按「离线建库」永远建不出东西,唯一的逃生口 make kb-reset 还会连手工内容一起清掉。
    seen = dedup.fingerprints(await repository.list_chunk_pairs())
    inserted = skipped = 0
    for fname, ctype in DOCS.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        todo, dup = [], 0
        for c in chunks:
            fp = dedup.fingerprint(c.questions, c.answer)
            if fp in seen:
                dup += 1
                continue
            seen.add(fp)                    # 同批内也去重:指纹撞上的块只进一份
            todo.append(c)
        ids = await dualwrite.write_pending(todo)
        inserted += len(ids)
        skipped += dup
        print(f"  {fname}: 新增 {len(ids)} 块,跳过 {dup} 块(指纹已在库)")
    tail = "下一步:scripts/vectorize_kb.py" if inserted else "没有新增;改过材料要整份重建走 make kb-reset"
    print(f"✅ 建库(pending):新增 {inserted} 块,跳过 {skipped} 块。{tail}")


if __name__ == "__main__":
    asyncio.run(main())
