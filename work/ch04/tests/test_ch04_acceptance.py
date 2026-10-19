"""验收四条的快速测试:评估集/报告结构 + 真实 KB 的 BM25 型号命中。

慢查询(citations/拒答落池)留在 scripts/verify_acceptance.py;这里保证
`uv run pytest` 也能在最少的依赖下发现结构漂移。
"""
import importlib.util
import json
import pathlib

import pytest

from app.core import retrieval
from app.kb import milvus_client

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"
REPORT = ROOT.parent.parent / "data/ch04/reports/rag_eval.json"
BUCKETS = {"A_policy", "B_model", "C_colloquial", "E_multi", "D_offtopic"}
ANSWERABLE = ("A_policy", "B_model", "C_colloquial", "E_multi")
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def _load_verify_script():
    path = ROOT / "scripts" / "verify_acceptance.py"
    spec = importlib.util.spec_from_file_location("verify_acceptance_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _valid_report() -> dict:
    bucket = {"recall_at_5": 1.0, "mrr": 1.0, "n": 20}
    return {
        "retrieval": {st: {b: dict(bucket) for b in ANSWERABLE}
                      for st in STRATEGIES},
        "evidence_coverage": {st: {b: 1.0 for b in ANSWERABLE}
                              for st in STRATEGIES},
    }


def _write_report(tmp_path, monkeypatch, payload) -> None:
    script = _load_verify_script()
    path = tmp_path / "rag_eval.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(script, "REPORT", path)
    return script


def test_verify_report_shape_drift_fails_cleanly(tmp_path, monkeypatch, capsys):
    script = _write_report(tmp_path, monkeypatch, {"retrieval": []})
    assert script.check_report() is False
    assert "❌" in capsys.readouterr().out


def test_verify_report_top_level_list_fails_cleanly(tmp_path, monkeypatch, capsys):
    script = _write_report(tmp_path, monkeypatch, [])
    assert script.check_report() is False
    assert "❌" in capsys.readouterr().out


def test_verify_report_rejects_bool_n(tmp_path, monkeypatch, capsys):
    report = _valid_report()
    report["retrieval"]["bm25"]["A_policy"]["n"] = True
    script = _write_report(tmp_path, monkeypatch, report)
    assert script.check_report() is False
    assert "❌" in capsys.readouterr().out


def test_verify_report_requires_all_coverage_strategies(tmp_path, monkeypatch, capsys):
    report = _valid_report()
    report["evidence_coverage"].pop("hybrid")
    script = _write_report(tmp_path, monkeypatch, report)
    assert script.check_report() is False
    assert "❌" in capsys.readouterr().out


def test_eval_set_structure():
    rows = [json.loads(line) for line in
            EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 100
    counts = {}
    ids = set()
    for row in rows:
        assert row["id"] not in ids
        ids.add(row["id"])
        counts[row["bucket"]] = counts.get(row["bucket"], 0) + 1
        assert row["difficulty"] in (1, 2, 3)
        if row["bucket"] == "D_offtopic":
            assert row["expect_section"] == [] and row["expect_points"] == []
        elif row["bucket"] == "E_multi":
            assert len(row["expect_sections_all"]) >= 2
        else:
            assert row["expect_section"] and row["expect_points"]
    assert counts == {bucket: 20 for bucket in BUCKETS}


def test_report_schema_when_present():
    if not REPORT.exists():
        pytest.skip("报告还没跑;先 make eval-rag")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = {"vector", "bm25", "hybrid", "hybrid_rerank"}
    assert set(report["retrieval"]) == expected
    assert set(report["evidence_coverage"]) == expected
    for strategy in report["retrieval"]:
        assert {"A_policy", "B_model", "C_colloquial", "E_multi"} <= set(
            report["retrieval"][strategy])
    assert "meta" in report and report["meta"]["recall_k"] == 5


async def test_bm25_model_hit_on_real_kb():
    try:
        client = milvus_client.get_client()
        milvus_client.ensure_collection(client)
        if milvus_client.count(client) == 0:
            pytest.skip("知识库为空;先 make kb-vectorize")
    except Exception as e:  # noqa: BLE001 - 环境没起时跳过,不让单测红
        pytest.skip(f"Milvus 不可用:{e}")
    hits = await retrieval.search_knowledge(
        "MH-W40 滤芯多久换一次", strategy="bm25", top_k=5)
    assert hits and "MH-W40" in hits[0]["section_path"]
