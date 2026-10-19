"""编号证据 ↔ chunk 元数据。n 从 1 开始,与证据文本、SSE citations 一致。"""


def build_citations(hits: list[dict]) -> list[dict]:
    return [
        {
            "n": i + 1,
            "chunk_id": h["id"],
            "section_path": h.get("section_path") or "",
            "question": h.get("question") or "",
            "answer": h.get("answer") or "",
        }
        for i, h in enumerate(hits)
    ]


def build_evidence(citations: list[dict]) -> str:
    return "\n".join(
        f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations
    )
