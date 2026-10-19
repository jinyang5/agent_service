"""校验评估集:结构 + ground truth 必须能在 data/kb 的真实切块里落地。
数据任务,非 TDD。运行:make eval-check。

直接跑时中文 Windows(默认 GBK)必须显式 UTF-8,否则读带中文的 JSONL/材料会
UnicodeDecodeError:
  PYTHONUTF8=1 PYTHONPATH=. uv run python scripts/build_eval_set.py
"""
import json
import pathlib
import re
import sys

from app.kb import documents
from app.kb.sources import KB_DIR, SOURCE_TYPES

ROOT = pathlib.Path(__file__).resolve().parents[1]       # work/ch04
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"
BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi", "D_offtopic"]


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def load_chunks() -> list:
    out = []
    for fname, ctype in SOURCE_TYPES.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        out.extend(documents.build_chunks(md, content_type=ctype))
    return out


def main() -> int:
    errors: list[str] = []
    rows = [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    if len(rows) != 100:
        errors.append(f"题数应为 100,实际 {len(rows)}")

    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        errors.append("id 有重复")

    counts = {b: 0 for b in BUCKETS}
    chunks = load_chunks()

    for r in rows:
        rid, bucket = r.get("id"), r.get("bucket")
        if bucket not in counts:
            errors.append(f"{rid}: 未知 bucket {bucket}")
            continue
        counts[bucket] += 1
        if not r.get("query"):
            errors.append(f"{rid}: query 为空")
        if r.get("difficulty") not in (1, 2, 3):
            errors.append(f"{rid}: difficulty 必须是 1/2/3")

        if bucket in ("A_policy", "B_model", "C_colloquial"):
            aliases = r.get("expect_section") or []
            points = r.get("expect_points") or []
            if not aliases:
                errors.append(f"{rid}: 缺 expect_section")
            if not points:
                errors.append(f"{rid}: 缺 expect_points")
            matched = [c for c in chunks
                       if any(a in (c.section_path or "") for a in aliases)]
            if not matched:
                errors.append(f"{rid}: expect_section 在真实 section_path 里没命中: {aliases}")
                continue
            blob = norm("\n".join(c.questions + c.answer for c in matched))
            for p in points:
                if norm(p) not in blob:
                    errors.append(f"{rid}: expect_point 未在目标小节原文中出现: {p!r}")
        elif bucket == "E_multi":
            groups = r.get("expect_sections_all") or []
            if len(groups) < 2:
                errors.append(f"{rid}: E 桶至少两组 expect_sections_all")
            hit_paths: list[set[str]] = []
            for group in groups:
                paths = {c.section_path for c in chunks
                         if any(a in (c.section_path or "") for a in group)}
                if not paths:
                    errors.append(f"{rid}: E 组在真实 section_path 里没命中: {group}")
                hit_paths.append(paths)
            # 至少两组,且组间真实 section_path 集合必须 disjoint:只要两组共享
            # 任一小节,就说明「跨小节」是别名重叠造成的假象,不是真正的多跳。
            for i, left in enumerate(hit_paths):
                for right in hit_paths[i + 1:]:
                    overlap = left & right
                    if overlap:
                        errors.append(
                            f"{rid}: E 题组间 section_path 有重叠(非 disjoint): "
                            f"{sorted(overlap)}")
            if sum(1 for paths in hit_paths if paths) < 2:
                errors.append(f"{rid}: E 题没有跨小节(至少要有两组命中)")
            if r.get("expect_section") or r.get("expect_points"):
                errors.append(f"{rid}: E 桶不应带 expect_section/expect_points")
        else:  # D_offtopic
            if r.get("expect_section") or r.get("expect_points"):
                errors.append(f"{rid}: D 桶不应带 ground truth")
            if r.get("expect_sections_all"):
                errors.append(f"{rid}: D 桶不应带 expect_sections_all")

    for b, n in counts.items():
        if n != 20:
            errors.append(f"{b}: 应为 20 题,实际 {n}")

    print(f"评估集 {len(rows)} 题;分桶 {counts};KB 切块 {len(chunks)} 块")
    if errors:
        print(f"\n❌ {len(errors)} 个错误:")
        for e in errors:
            print("  -", e)
        return 1
    print("✅ 结构、section 命中、points 原文、E 多跳、D 无 ground truth 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
