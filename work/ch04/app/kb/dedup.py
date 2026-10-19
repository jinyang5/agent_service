import re
from collections.abc import Iterable

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)  # \W 不含中文(中文属 \w),故只去空白/标点


def normalize_question(q: str) -> str:
    return _STRIP_RE.sub("", q.strip().lower())


def fingerprint(questions: str, answer: str) -> str:
    """「问法 + 正文」的归一化指纹:录入查重与离线建库守卫共用同一把尺。

    只按问法会误杀同节切出的多块(表格按行拆、长文递归切,标题相同);
    带上正文,同一份内容无论从哪条路入库(页面录入 / make kb-build),指纹都撞得上。
    """
    return normalize_question(questions) + "|" + normalize_question(answer)


def fingerprints(pairs: Iterable[tuple[str, str]]) -> set[str]:
    """全表 (questions, answer) → 指纹集合:说清「库里已有哪些内容」。"""
    return {fingerprint(q, a) for q, a in pairs}


def dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]:
    seen = {normalize_question(q) for q in existing_questions}
    kept, discarded = [], []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            discarded.append(item)
        else:
            seen.add(key)
            kept.append(item)
    return kept, discarded
