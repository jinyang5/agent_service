"""对 Query 改写跑标注样例(真实 DeepSeek 上游,需要 .env)。

命中判定:expect_any 中任一词出现在 standard 或 expanded 里;两者分开统计。
通过线:>= 5/6(brief);目标 6/6。
"""
import asyncio
import json
import pathlib
import sys

from app.core.llm import get_chat_model
from app.core.query_understanding import understand

SAMPLES = pathlib.Path(__file__).parent.parent / "tests/data/query_rewrite_samples.jsonl"
EXPECTED_TOTAL = 6


async def main() -> int:
    samples = [
        json.loads(ln)
        for ln in SAMPLES.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    tot = len(samples)
    if tot != EXPECTED_TOTAL:
        print(f"样本数不符:期望 {EXPECTED_TOTAL},实际 {tot}", file=sys.stderr)
        return 2

    model = get_chat_model(temperature=0)  # 消除重放抖动
    ok = std_ok = exp_ok = 0
    for s in samples:
        r = await understand(s["query"], model=model)
        std_hit = any(w in r["standard"] for w in s["expect_any"])
        exp_hit = any(w in " ".join(r["expanded"]) for w in s["expect_any"])
        hit = std_hit or exp_hit
        print(
            "OK" if hit else "MISS",
            s["query"],
            "->",
            r,
            f"[standard={'HIT' if std_hit else '-'} expanded={'HIT' if exp_hit else '-'}]",
        )
        ok += hit
        std_ok += std_hit
        exp_ok += exp_hit
    print(f"standard 命中 {std_ok}/{tot};expanded 命中 {exp_ok}/{tot};合计 {ok}/{tot}")
    return 1 if ok < 5 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
