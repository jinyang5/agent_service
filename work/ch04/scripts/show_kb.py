"""验收展示(dry-run):列出建库材料 + 预览结构化切块结果,不写库、不碰 Milvus、不调上游。
运行:PYTHONPATH=. uv run python scripts/show_kb.py"""
import pathlib

from app.kb import chunking, documents
from app.kb.sources import KB_DIR, SOURCE_TYPES as DOCS

ROOT = pathlib.Path(__file__).resolve().parents[3]
SEED_SQL = ROOT / "sql" / "ch03-seed.sql"


def _preview(text: str, n: int = 46) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[:n] + "…"


def main() -> None:
    print("=== 离线建库材料清单 ===")
    for fname, ctype in DOCS.items():
        raw = (KB_DIR / fname).read_text(encoding="utf-8")
        print(f"  data/kb/{fname:<24} [{ctype:<6}] {len(raw):>4} 字符 / {raw.count(chr(10)) + 1} 行")
    print(f"历史会话种子:{SEED_SQL.relative_to(ROOT)}(喂 make kb-mine)\n")
    print("=== 切块预览(dry-run,不写库)===")
    tally: dict[str, int] = {}
    key_total = 0
    for fname, ctype in DOCS.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        tally[ctype] = tally.get(ctype, 0) + len(chunks)
        by_path: dict[str, list[documents.Chunk]] = {}
        for c in chunks:
            by_path.setdefault(c.section_path, []).append(c)
        print(f"\n▼ {fname} [{ctype}]  →  {len(chunks)} 块")
        for path, group in by_path.items():
            multi = f"  (该节切成 {len(group)} 块)" if len(group) > 1 else ""
            print(f"  ┌ 节:{path}{multi}")
            for j, c in enumerate(group, 1):
                key_total += c.is_key_clause
                flag = " ★关键条款" if c.is_key_clause else ""
                kind = "表格块" if chunking.is_table_block(c.answer) else "文本块"
                seq = f"{j}/{len(group)}" if len(group) > 1 else "-"
                print(f"  │ [{kind} {seq}]{flag}  Q={c.questions}  category={c.category}")
                print(f"  │   A({len(c.answer)}字): {_preview(c.answer)}")
    print("\n=== 汇总 ===")
    print(f"总块数 {sum(tally.values())};类型分布 " + " ".join(f"{k}={v}" for k, v in tally.items()))
    print(f"关键条款(is_key_clause=1)共 {key_total} 块")


if __name__ == "__main__":
    main()
