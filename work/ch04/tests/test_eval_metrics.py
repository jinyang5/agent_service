import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from app.core import eval_metrics as m


def _hit(section):
    return {"id": 1, "section_path": section, "question": "q", "answer": "a"}


def test_norm_and_section_hit():
    assert m.norm("满 99 元\n包邮") == "满99元包邮"
    assert m.section_hit("商品规格手册 / MH-W40", ["MH-W40"]) is True
    assert m.section_hit("商品规格手册 / MH-W20", ["MH-W40"]) is False


def test_recall_at_5_single_bucket():
    s = {"bucket": "A_policy", "expect_section": ["运费"]}
    assert m.recall_at_k(s, [_hit("x"), _hit("运费")], k=5) == 1.0
    assert m.recall_at_k(s, [_hit("x")] * 5 + [_hit("运费")], k=5) == 0.0


def test_recall_at_5_e_multi_partial():
    s = {"bucket": "E_multi", "expect_sections_all": [["A"], ["B"]]}
    hits = [_hit("A")] + [_hit("x")] * 5 + [_hit("B")]
    assert m.recall_at_k(s, hits, k=5) == 0.5


def test_mrr_single_and_multi():
    s = {"bucket": "A_policy", "expect_section": ["运费"]}
    assert m.mrr(s, [_hit("x"), _hit("运费")]) == 0.5
    e = {"bucket": "E_multi", "expect_sections_all": [["A"], ["B"]]}
    assert m.mrr(e, [_hit("A"), _hit("x"), _hit("x"), _hit("B")]) == (1 + 1 / 4) / 2
    assert m.mrr(e, [_hit("A"), _hit("x")]) == 0.5


def test_evidence_coverage_mechanical():
    s = {"expect_points": ["满99元包邮", "10元运费"]}
    hits = [{"question": "运费", "answer": "满 99 元包邮"}]
    assert m.evidence_coverage(s, hits) == 0.5


def test_sample_per_bucket_is_per_bucket():
    rows = ([{"id": f"A{i}", "bucket": "A_policy"} for i in range(3)] +
            [{"id": f"D{i}", "bucket": "D_offtopic"} for i in range(3)])
    out = m.sample_per_bucket(rows, limit=2)
    assert [r["id"] for r in out] == ["A0", "A1", "D0", "D1"]
    assert m.sample_per_bucket(rows, limit=None) == rows
    # 0/负数是"每桶取 0 条",不能回退成全量
    assert m.sample_per_bucket(rows, limit=0) == []
    assert m.sample_per_bucket(rows, limit=-1) == []


def _load_eval_script():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_ch04.py"
    spec = importlib.util.spec_from_file_location("eval_ch04_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def test_eval_script_report_structure(tmp_path, monkeypatch):
    """mock 检索 + --skip-generation:不碰 Milvus/LLM,只验证 JSON 契约。"""
    script = _load_eval_script()

    async def fake_search(query, strategy="hybrid_rerank", top_k=None, **kwargs):
        return [{"id": 1, "section_path": "运费怎么算",
                 "question": "q", "answer": "a"}]

    monkeypatch.setattr(script.retrieval, "search_knowledge", fake_search)
    monkeypatch.setattr(script, "_preflight", lambda: 42)
    monkeypatch.setattr(script, "OUT_DIR", tmp_path)
    monkeypatch.setattr(script, "OUT_TXT", tmp_path / "rag_eval.txt")
    monkeypatch.setattr(script, "OUT_JSON", tmp_path / "rag_eval.json")
    monkeypatch.setattr(sys, "argv",
                        ["eval_ch04.py", "--limit", "1", "--skip-generation"])

    assert await script.main() == 0

    report = json.loads((tmp_path / "rag_eval.json").read_text(encoding="utf-8"))
    assert set(report) == {"meta", "retrieval", "evidence_coverage", "generation"}

    meta = report["meta"]
    expected_meta = {"generated_at", "eval_set_size", "sampled", "limit",
                     "strategies", "buckets", "retrieval_depth", "recall_k",
                     "embed_model", "rerank_model", "judge_model", "kb_chunks"}
    assert expected_meta <= set(meta)
    assert meta["eval_set_size"] == 100 and meta["sampled"] == 5
    assert meta["limit"] == 1 and meta["kb_chunks"] == 42

    strategies = ["vector", "bm25", "hybrid", "hybrid_rerank"]
    buckets = ["A_policy", "B_model", "C_colloquial", "E_multi"]
    for st in strategies:
        for b in buckets:
            cell = report["retrieval"][st][b]
            assert set(cell) == {"recall_at_5", "mrr", "n"}
            assert cell["n"] == 1
        assert set(report["evidence_coverage"][st]) == {
            "A_policy", "B_model", "C_colloquial", "overall"}

    gen = report["generation"]
    assert gen["completed"] is False
    assert gen["answer_coverage"] == {} and gen["faithfulness"] == {}
    assert gen["refusal"] == {} and gen["faithfulness_cases"] == []
    # 跳过/失败必须显式进报告,不能静默剔除
    assert any("--skip-generation" in e for e in gen["errors"])

    text = (tmp_path / "rag_eval.txt").read_text(encoding="utf-8")
    assert "Ch04 RAG 评估报告" in text
    assert "--skip-generation" in text


class _FakeChain:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, inputs):
        self.calls += 1
        return SimpleNamespace(content="答案")


class _FakePrompt:
    def __init__(self):
        self.chain = _FakeChain()

    def __or__(self, other):
        return self.chain


def _patch_llm(script, monkeypatch):
    monkeypatch.setattr(script, "get_chat_model", lambda **kwargs: object())
    prompt = _FakePrompt()
    monkeypatch.setattr(script, "RAG_ANSWER_PROMPT", prompt)
    return prompt


def _patch_paths(script, monkeypatch, tmp_path):
    monkeypatch.setattr(script, "OUT_DIR", tmp_path)
    monkeypatch.setattr(script, "OUT_TXT", tmp_path / "rag_eval.txt")
    monkeypatch.setattr(script, "OUT_JSON", tmp_path / "rag_eval.json")


async def test_retrieval_failure_skips_generation(monkeypatch):
    """某策略检索失败时,不得用空证据生成/裁判(避免污染 faith_cases)。"""
    script = _load_eval_script()
    prompt = _patch_llm(script, monkeypatch)
    judge_calls = []

    async def fake_judge(*args, **kwargs):
        judge_calls.append(1)
        return {"covered": [], "faithful": True, "reason": ""}

    monkeypatch.setattr(script, "judge_answer", fake_judge)

    sample = {"id": "A01", "bucket": "A_policy", "query": "q",
              "expect_points": ["p"]}
    hits = {(sample["id"], st): [] for st in script.STRATEGIES}
    out = await script._generation([sample], hits, skip=False)

    assert prompt.chain.calls == 0 and judge_calls == []
    assert out["completed"] is False
    assert sum("跳过生成" in e for e in out["errors"]) == len(script.STRATEGIES)
    assert out["answer_coverage"]["vector"]["A_policy"] is None
    assert out["faithfulness"]["hybrid_rerank"]["overall"] == {"rate": None, "n": 0}


async def test_generation_success_fills_faithfulness(monkeypatch):
    """回归:faithful 桶容器必须是 list,不能是嵌套 defaultdict(否则 .append 崩)。"""
    script = _load_eval_script()
    _patch_llm(script, monkeypatch)

    async def fake_judge(*args, **kwargs):
        return {"covered": [1], "faithful": True, "reason": ""}

    monkeypatch.setattr(script, "judge_answer", fake_judge)

    sample = {"id": "A01", "bucket": "A_policy", "query": "q",
              "expect_points": ["p"]}
    hit = {"id": 1, "section_path": "s", "question": "q", "answer": "a"}
    hits = {(sample["id"], st): [hit] for st in script.STRATEGIES}
    out = await script._generation([sample], hits, skip=False)

    assert out["completed"] is True and out["errors"] == []
    for st in script.STRATEGIES:
        assert out["answer_coverage"][st]["A_policy"] == 1.0
    f = out["faithfulness"]["hybrid_rerank"]
    assert f["A_policy"] == {"rate": 1.0, "n": 1}
    assert f["overall"] == {"rate": 1.0, "n": 1}


async def test_refusal_rate_excludes_infra_errors(tmp_path, monkeypatch):
    """1 条 error + 1 条真拒答 → rate=1/1,n 只算有效检查。"""
    script = _load_eval_script()
    _patch_llm(script, monkeypatch)
    _patch_paths(script, monkeypatch, tmp_path)
    samples = [
        {"id": "D01", "bucket": "D_offtopic", "query": "err-query",
         "expect_points": []},
        {"id": "D02", "bucket": "D_offtopic", "query": "ok-query",
         "expect_points": []},
    ]

    async def fake_ainvoke(payload):
        if payload["keyword"].startswith("err"):
            raise RuntimeError("faq down")
        return {"sufficient": False, "source": "retrieval_low_conf",
                "reason": "low"}

    monkeypatch.setattr(script, "_load", lambda limit: (samples, 100))
    monkeypatch.setattr(script, "_preflight", lambda: 42)
    monkeypatch.setattr(script, "query_faq",
                        SimpleNamespace(ainvoke=fake_ainvoke))
    monkeypatch.setattr(sys, "argv", ["eval_ch04.py"])

    assert await script.main() == 0

    report = json.loads((tmp_path / "rag_eval.json").read_text(encoding="utf-8"))
    refusal = report["generation"]["refusal"]["D_offtopic"]
    assert refusal["rate"] == 1.0
    assert refusal["n"] == 1
    assert len(refusal["details"]) == 2
    assert refusal["sources"] == {"error": 1, "retrieval_low_conf": 1}
    assert any("D 拒答检查失败" in e for e in report["generation"]["errors"])


async def test_d_refusal_alone_does_not_complete_generation(monkeypatch):
    """R42:D 桶拒答成功不等于生成段完成;A/B/C 一条都没成时必须 completed=False。"""
    script = _load_eval_script()
    _patch_llm(script, monkeypatch)
    samples = [
        {"id": "A01", "bucket": "A_policy", "query": "q", "expect_points": ["p"]},
        {"id": "D01", "bucket": "D_offtopic", "query": "库外问题", "expect_points": []},
    ]
    hits = {(s["id"], st): [] for s in samples if s["bucket"] != "D_offtopic"
            for st in script.STRATEGIES}

    async def fake_ainvoke(payload):
        return {"sufficient": False, "source": "retrieval_low_conf", "reason": "low"}

    monkeypatch.setattr(script, "query_faq",
                        SimpleNamespace(ainvoke=fake_ainvoke))

    out = await script._generation(samples, hits, skip=False)

    assert out["completed"] is False
    assert out["refusal"]["D_offtopic"]["rate"] == 1.0
    assert out["refusal"]["D_offtopic"]["n"] == 1
    assert sum("跳过生成" in e for e in out["errors"]) == len(script.STRATEGIES)


async def test_all_retrieval_fail_report_not_completed(tmp_path, monkeypatch):
    """四策略全抛异常:completed=False,_write_report 走未完成分支。"""
    script = _load_eval_script()
    prompt = _patch_llm(script, monkeypatch)
    _patch_paths(script, monkeypatch, tmp_path)
    samples = [{"id": "A01", "bucket": "A_policy", "query": "q",
                "expect_points": ["p"]}]

    async def boom(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(script, "_load", lambda limit: (samples, 100))
    monkeypatch.setattr(script, "_preflight", lambda: 42)
    monkeypatch.setattr(script.retrieval, "search_knowledge", boom)
    monkeypatch.setattr(sys, "argv", ["eval_ch04.py"])

    assert await script.main() == 0

    report = json.loads((tmp_path / "rag_eval.json").read_text(encoding="utf-8"))
    gen = report["generation"]
    assert gen["completed"] is False
    assert prompt.chain.calls == 0
    assert sum("跳过生成" in e for e in gen["errors"]) == len(script.STRATEGIES)
    text = (tmp_path / "rag_eval.txt").read_text(encoding="utf-8")
    assert "本次未完成" in text


async def test_script_rejects_non_positive_limit(monkeypatch):
    script = _load_eval_script()
    monkeypatch.setattr(sys, "argv", ["eval_ch04.py", "--limit", "0"])
    with pytest.raises(SystemExit) as exc:
        await script.main()
    assert exc.value.code != 0
