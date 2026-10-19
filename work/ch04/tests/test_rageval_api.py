"""RAG 评估页后端:只读报告 API + 编造个案台账 + 状态流转。

报告产物只读、不重算;retrieval_overall 只做四桶 MRR 的算术平均(响应里注明)。
"""
import json
from datetime import datetime

import pytest
from sqlalchemy import select

from app.api import rageval
from app.db import repository
from app.db.models import FaithCase

BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi"]


def _bucket(mrr: float) -> dict:
    return {"recall_at_5": 1.0, "mrr": mrr, "n": 20}


def _retrieval(**means: float) -> dict:
    return {st: {b: _bucket(m) for b in BUCKETS} for st, m in means.items()}


def _report(**overrides) -> dict:
    report = {
        "meta": {
            "generated_at": "2026-10-19T16:40:49",
            "eval_set_size": 100, "sampled": 100, "limit": None,
            "strategies": ["vector", "bm25", "hybrid", "hybrid_rerank"],
            "buckets": BUCKETS + ["D_offtopic"],
            "retrieval_depth": 10, "recall_k": 5,
            "embed_model": "BAAI/bge-m3",
            "rerank_model": "BAAI/bge-reranker-v2-m3",
            "judge_model": "deepseek-v4-flash", "kb_chunks": 82,
        },
        "retrieval": _retrieval(vector=0.75, bm25=0.5, hybrid=0.85, hybrid_rerank=0.95),
        "evidence_coverage": {"vector": {"A_policy": 1.0, "B_model": 1.0,
                                         "C_colloquial": 1.0, "overall": 1.0}},
        "generation": {
            "completed": True,
            "answer_coverage": {},
            "faithfulness": {"hybrid_rerank": {"overall": {"rate": 1.0, "n": 60}}},
            "refusal": {"D_offtopic": {"rate": 1.0, "n": 20, "sources": {}, "details": []}},
            "faithfulness_cases": [],
            "errors": [],
        },
    }
    report.update(overrides)
    return report


def _install(tmp_path, monkeypatch, report: dict | str | None) -> None:
    """把产物路径指到 tmp_path;report=None 表示文件不存在。"""
    path = tmp_path / "rag_eval.json"
    if report is not None:
        payload = report if isinstance(report, str) else json.dumps(report, ensure_ascii=False)
        path.write_text(payload, encoding="utf-8")
    txt = tmp_path / "rag_eval.txt"
    if report is not None:
        txt.write_text("人读报告", encoding="utf-8")
    monkeypatch.setattr(rageval, "REPORT_JSON", path)
    monkeypatch.setattr(rageval, "REPORT_TXT", txt)


async def _seed_cases() -> None:
    """三条个案,覆盖三个状态;每条都走既有的 upsert 入口。"""
    for eval_id in ("A01", "B01", "C01"):
        await repository.upsert_faith_case(
            eval_id, "A_policy", f"q-{eval_id}", "hybrid_rerank",
            "a", "r", None, "deepseek-v4-flash")
    await repository.set_faith_case_status("B01", "已解决", "补了知识条款")
    await repository.set_faith_case_status("C01", "无需解决", "题面歧义,无需改库")


# ---------- overview ----------

async def test_overview_present_passes_through_and_aggregates(client, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, _report())
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200
    body = r.json()

    assert body["present"] is True
    assert body["retrieval_overall"]["vector"] == pytest.approx(0.75)
    assert body["retrieval_overall"]["bm25"] == pytest.approx(0.5)
    assert body["retrieval_overall"]["hybrid"] == pytest.approx(0.85)
    assert body["retrieval_overall"]["hybrid_rerank"] == pytest.approx(0.95)
    assert body["overall_note"] == "聚合自 4 桶 MRR,不重算底层指标"

    # 产物原样透传,不重算
    assert body["retrieval"]["vector"]["E_multi"]["mrr"] == 0.75
    assert body["meta"]["kb_chunks"] == 82
    assert body["generation_done"] is True
    assert body["best"]["strategy"] == "hybrid_rerank"
    assert body["best"]["mrr"] == pytest.approx(0.95)
    assert body["strategies"] == ["vector", "bm25", "hybrid", "hybrid_rerank"]
    assert body["buckets"] == BUCKETS + ["D_offtopic"]
    assert body["make"] == "make eval-rag"

    assert body["job"]["spec"]["name"] == "eval-rag"
    assert body["job"]["spec"]["cmd"] == "make eval-rag"
    assert body["artifacts"]["json"]["present"] is True
    assert body["artifacts"]["json"]["bytes"] > 0
    assert body["artifacts"]["json"]["mtime"]
    assert body["artifacts"]["text"]["present"] is True


async def test_overview_missing_report_is_not_a_500(client, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, None)
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["present"] is False
    assert body["make"] == "make eval-rag"
    assert body["hint"]
    assert body["job"]["spec"]["name"] == "eval-rag"
    assert body["artifacts"]["json"]["present"] is False
    assert body["artifacts"]["json"]["bytes"] is None
    assert body["artifacts"]["json"]["mtime"] is None


async def test_overview_broken_json_is_present_false(client, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, "{ this is not json")
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["present"] is False
    assert body["make"] == "make eval-rag"
    assert body["hint"]


async def test_overview_best_is_max_mean(client, tmp_path, monkeypatch):
    report = _report(retrieval=_retrieval(vector=0.1, bm25=0.2, hybrid=0.3,
                                          hybrid_rerank=0.25))
    _install(tmp_path, monkeypatch, report)
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["best"]["strategy"] == "hybrid"
    assert body["best"]["mrr"] == pytest.approx(0.3)


async def test_overview_best_tie_breaks_by_strategy_order(client, tmp_path, monkeypatch):
    # 产物里的键顺序故意把 bm25 放前面;并列时应取固定顺序靠前的 vector
    report = _report(retrieval=_retrieval(bm25=0.5, vector=0.5))
    _install(tmp_path, monkeypatch, report)
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["best"]["strategy"] == "vector"
    assert body["best"]["mrr"] == pytest.approx(0.5)


async def test_overview_overall_is_explicit_four_bucket_mean(client, tmp_path, monkeypatch):
    retrieval = {"hybrid_rerank": {
        "A_policy": _bucket(1.0),
        "B_model": _bucket(0.8),
        "C_colloquial": _bucket(0.6),
        "E_multi": _bucket(0.4),
        "D_offtopic": _bucket(9.9),          # 拒答桶不参与检索均值
    }}
    _install(tmp_path, monkeypatch, _report(retrieval=retrieval))
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["retrieval_overall"]["hybrid_rerank"] == pytest.approx(0.7)
    assert body["overall_note"] == "聚合自 4 桶 MRR,不重算底层指标"


async def test_overview_overall_skips_null_and_missing_buckets(client, tmp_path, monkeypatch):
    null_bucket = {"recall_at_5": None, "mrr": None, "n": 0}
    retrieval = {
        "vector": {"A_policy": null_bucket, "B_model": _bucket(0.9),
                   "C_colloquial": _bucket(0.5)},          # E_multi 缺失
        "bm25": {"A_policy": null_bucket, "B_model": _bucket(1.0),
                 "C_colloquial": _bucket(0.4)},            # E_multi 缺失
    }
    _install(tmp_path, monkeypatch, _report(retrieval=retrieval))
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["retrieval_overall"]["vector"] == pytest.approx(0.7)
    assert body["retrieval_overall"]["bm25"] == pytest.approx(0.7)
    assert body["overall_note"] == "聚合自 2 桶 MRR,不重算底层指标"


@pytest.mark.parametrize("report", [
    {"meta": {}, "retrieval": [1]},
    {"retrieval": {"vector": [1]}},
    {"retrieval": {"vector": {"A_policy": 1.0}}},
])
async def test_overview_shape_drift_is_present_false(client, tmp_path, monkeypatch, report):
    _install(tmp_path, monkeypatch, report)
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200                   # 合法 JSON 但形状漂移,不许 500
    body = r.json()
    assert body["present"] is False
    assert body["make"] == "make eval-rag"
    assert body["hint"]


async def test_overview_generation_incomplete_flags_done_false(client, tmp_path, monkeypatch):
    report = _report(generation={"completed": False, "answer_coverage": {},
                                 "faithfulness": {}, "refusal": {},
                                 "faithfulness_cases": [], "errors": ["上游超时"]})
    _install(tmp_path, monkeypatch, report)
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["present"] is True
    assert body["generation_done"] is False


# ---------- faith cases 台账 ----------

async def test_faith_cases_list_counts_and_filter(client, db_session_factory):
    await _seed_cases()
    r = await client.get("/api/rag-eval/faith-cases")
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"未解决": 1, "已解决": 1, "无需解决": 1}
    assert len(body["items"]) == 3
    item = next(i for i in body["items"] if i["eval_id"] == "B01")
    assert item["status"] == "已解决"
    assert item["resolution"] == "补了知识条款"
    assert item["resolved_at"]
    assert item["reopened"] is False
    assert item["first_seen_at"] and item["last_seen_at"]
    assert item["citations"] is None

    r2 = await client.get("/api/rag-eval/faith-cases", params={"status": "已解决"})
    body2 = r2.json()
    assert [i["eval_id"] for i in body2["items"]] == ["B01"]
    # 计数是全局台账口径,不随筛选变化
    assert body2["counts"] == {"未解决": 1, "已解决": 1, "无需解决": 1}


async def test_faith_cases_reopened_flag(client, db_session_factory):
    await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a", "r", None, "m")
    await repository.set_faith_case_status("A01", "已解决", "补了知识")
    await repository.set_faith_case_status("A01", "未解决", None)

    body = (await client.get("/api/rag-eval/faith-cases")).json()
    item = body["items"][0]
    assert item["status"] == "未解决"
    assert item["resolution"] is None
    assert item["resolved_at"]                       # 保留复发标记
    assert item["reopened"] is True


async def test_status_resolved_requires_non_blank_resolution(client, db_session_factory):
    await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a", "r", None, "m")
    for payload in ({"status": "已解决"},
                    {"status": "无需解决", "resolution": "   "}):
        r = await client.post("/api/rag-eval/faith-cases/A01/status", json=payload)
        assert r.status_code == 400, payload


async def test_faith_cases_invalid_status_query_is_422(client):
    r = await client.get("/api/rag-eval/faith-cases", params={"status": "已处理"})
    assert r.status_code == 422


async def test_status_resolution_max_length(client, db_session_factory):
    await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a", "r", None, "m")
    r = await client.post("/api/rag-eval/faith-cases/A01/status",
                          json={"status": "已解决", "resolution": "x" * 301})
    assert r.status_code == 422
    r = await client.post("/api/rag-eval/faith-cases/A01/status",
                          json={"status": "已解决", "resolution": "x" * 300})
    assert r.status_code == 200


async def test_status_invalid_enum_is_422(client, db_session_factory):
    r = await client.post("/api/rag-eval/faith-cases/A01/status",
                          json={"status": "已处理", "resolution": "x"})
    assert r.status_code == 422


async def test_status_update_resolve_and_reopen_keeps_resolved_at(client, db_session_factory):
    await repository.upsert_faith_case(
        "A01", "A_policy", "q", "hybrid_rerank", "a", "r", None, "m")

    r = await client.post("/api/rag-eval/faith-cases/A01/status",
                          json={"status": "已解决", "resolution": " 补了知识 "})
    assert r.status_code == 200
    item = r.json()["item"]
    assert r.json()["ok"] is True
    assert item["status"] == "已解决"
    assert item["resolution"] == "补了知识"          # 去空白后入库
    assert item["resolved_at"]

    # 把 resolved_at 回填成过去时间,后续操作都不许覆盖它
    past = datetime(2026, 1, 2, 3, 4, 5)
    async with db_session_factory() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == "A01")
        )).scalar_one()
        row.resolved_at = past
        await s.commit()
    past_iso = past.isoformat(timespec="seconds")

    # 同值重复提交(MySQL rowcount=0)不许误判 404
    same = await client.post("/api/rag-eval/faith-cases/A01/status",
                             json={"status": "已解决", "resolution": "补了知识"})
    assert same.status_code == 200
    assert same.json()["item"]["resolved_at"] == past_iso

    # 再标一次别的终态:resolved_at 保留首次时间,不覆盖
    r2 = await client.post("/api/rag-eval/faith-cases/A01/status",
                           json={"status": "无需解决", "resolution": "题面歧义"})
    assert r2.json()["item"]["resolved_at"] == past_iso

    # 重开:resolution 清空,resolved_at 保留(用来标复发)
    r3 = await client.post("/api/rag-eval/faith-cases/A01/status",
                           json={"status": "未解决", "resolution": "ignored"})
    assert r3.status_code == 200
    reopened = r3.json()["item"]
    assert reopened["status"] == "未解决"
    assert reopened["resolution"] is None
    assert reopened["resolved_at"] == past_iso
    assert reopened["reopened"] is True


async def test_status_unknown_eval_id_is_404(client, db_session_factory):
    r = await client.post("/api/rag-eval/faith-cases/NOPE/status",
                          json={"status": "未解决"})
    assert r.status_code == 404


# ---------- 后台卡片 ----------

async def _fake_milvus(monkeypatch):
    from app.api import kb

    async def fake():
        return {"online": True, "count": 0, "collection": "knowledge"}

    monkeypatch.setattr(kb, "milvus_state", fake)


async def test_admin_overview_has_rag_eval_card(client, db_session_factory,
                                                tmp_path, monkeypatch):
    await _fake_milvus(monkeypatch)
    _install(tmp_path, monkeypatch, _report())
    await _seed_cases()

    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    card = next(c for c in r.json()["modules"] if c["key"] == "rag-eval")
    assert card["title"] == "RAG 评估"
    assert card["page"] == "/rag-eval"
    assert card["status"] == "ok"
    metrics = {m["label"]: m["value"] for m in card["metrics"]}
    assert metrics["最佳 MRR"] == pytest.approx(0.95)
    assert metrics["评估题数"] == 100
    assert metrics["库外拒答"] == pytest.approx(1.0)
    assert metrics["编造个案"] == 1                  # 未解决数
    assert "1 条编造个案未解决" in card["note"]


async def test_admin_rag_eval_card_missing_and_corrupt(client, db_session_factory,
                                                       tmp_path, monkeypatch):
    await _fake_milvus(monkeypatch)
    _install(tmp_path, monkeypatch, None)
    card = next(c for c in (await client.get("/api/admin/overview")).json()["modules"]
                if c["key"] == "rag-eval")
    assert (card["status"], card["headline"]) == ("missing", "还没跑过评估")

    _install(tmp_path, monkeypatch, "{ broken")
    card = next(c for c in (await client.get("/api/admin/overview")).json()["modules"]
                if c["key"] == "rag-eval")
    assert card["status"] == "error"
    assert card["note"]


async def test_admin_rag_eval_card_attention_when_generation_incomplete(
        client, db_session_factory, tmp_path, monkeypatch):
    await _fake_milvus(monkeypatch)
    _install(tmp_path, monkeypatch,
             _report(generation={"completed": False, "errors": ["上游超时"]}))
    card = next(c for c in (await client.get("/api/admin/overview")).json()["modules"]
                if c["key"] == "rag-eval")
    assert card["status"] == "attention"
    assert "生成段未完成" in card["headline"]


async def test_admin_rag_eval_card_isolated_from_report_shape_drift(
        client, db_session_factory, tmp_path, monkeypatch):
    """合法 JSON 但 generation.refusal 类型漂移:本卡 error,知识库卡不受影响。"""
    await _fake_milvus(monkeypatch)
    report = _report()
    report["generation"]["refusal"] = []             # list 而不是 dict
    _install(tmp_path, monkeypatch, report)

    r = await client.get("/api/admin/overview")
    assert r.status_code == 200                       # 整页不许 500
    cards = {c["key"]: c for c in r.json()["modules"]}
    assert cards["rag-eval"]["status"] == "error"
    assert cards["rag-eval"]["note"]
    assert cards["kb"]["status"] != "error"


async def test_admin_rag_eval_card_failure_does_not_break_kb_card(client, db_session_factory,
                                                                  tmp_path, monkeypatch):
    await _fake_milvus(monkeypatch)

    def boom():
        raise RuntimeError("模拟:报告读数炸了")

    monkeypatch.setattr(rageval, "report_snapshot", boom)
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    cards = {c["key"]: c for c in r.json()["modules"]}
    assert cards["kb"]["status"] != "error"          # 知识库卡不被连坐
    assert cards["rag-eval"]["status"] == "error"
    assert "模拟:报告读数炸了" in cards["rag-eval"]["note"]
