from app.core import citations


def _hit(i, section):
    return {"id": i, "question": f"q{i}", "answer": f"a{i}",
            "section_path": section, "content_type": "faq", "category": "c",
            "score": 0.5}


def test_build_citations_numbers_and_maps():
    out = citations.build_citations([_hit(5, "运费"), _hit(9, "退货")])
    assert [c["n"] for c in out] == [1, 2]
    assert out[0]["chunk_id"] == 5 and out[1]["chunk_id"] == 9
    assert out[0]["section_path"] == "运费"


def test_build_citations_empty():
    assert citations.build_citations([]) == []


def test_build_evidence_format():
    cites = citations.build_citations([_hit(1, "运费"), _hit(2, "退货")])
    assert citations.build_evidence(cites) == "[1] q1: a1\n[2] q2: a2"
