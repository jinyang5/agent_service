"""评估指标:确定性、可复现,不依赖 LLM。"""
import re

GRADED_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
ANSWERABLE_BUCKETS = ("A_policy", "B_model", "C_colloquial")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def section_hit(section_path: str, aliases: list[str]) -> bool:
    return any(a in (section_path or "") for a in aliases)


def _single_rank(sample: dict, hits: list[dict]) -> int | None:
    for i, h in enumerate(hits):
        if section_hit(h.get("section_path", ""), sample.get("expect_section") or []):
            return i + 1
    return None


def _group_rank(group: list[str], hits: list[dict]) -> int | None:
    for i, h in enumerate(hits):
        if section_hit(h.get("section_path", ""), group):
            return i + 1
    return None


def recall_at_k(sample: dict, hits: list[dict], k: int = 5) -> float:
    top = hits[:k]
    if sample["bucket"] == "E_multi":
        groups = sample.get("expect_sections_all") or []
        if not groups:
            return 0.0
        hit = sum(1 for g in groups
                  if any(section_hit(h.get("section_path", ""), g) for h in top))
        return hit / len(groups)
    return 1.0 if any(section_hit(h.get("section_path", ""), sample.get("expect_section") or [])
                      for h in top) else 0.0


def mrr(sample: dict, hits: list[dict]) -> float:
    if sample["bucket"] == "E_multi":
        groups = sample.get("expect_sections_all") or []
        if not groups:
            return 0.0
        scores = []
        for g in groups:
            rank = _group_rank(g, hits)
            scores.append(1.0 / rank if rank else 0.0)
        return sum(scores) / len(scores)
    rank = _single_rank(sample, hits)
    return 1.0 / rank if rank else 0.0


def evidence_coverage(sample: dict, hits: list[dict]) -> float:
    points = sample.get("expect_points") or []
    if not points:
        return 0.0
    blob = norm("\n".join((h.get("question") or "") + (h.get("answer") or "")
                          for h in hits))
    return sum(1 for p in points if norm(p) in blob) / len(points)


def sample_per_bucket(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return list(rows)
    picked, counts = [], {}
    for r in rows:
        b = r["bucket"]
        if counts.get(b, 0) < limit:
            picked.append(r)
            counts[b] = counts.get(b, 0) + 1
    return picked
