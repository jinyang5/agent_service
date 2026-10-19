"""RAG 评估页后端:只读报告产物 + 编造个案台账 + 重跑入口(作业在 app/core/jobs.py)。

只读产物、不重算:overview 原样透传 retrieval / evidence_coverage / generation,
只把 retrieval 的四桶 MRR 做算术平均得到 retrieval_overall,并在响应里注明口径。
产物缺失或读坏都返回 present=false(不是 500),页面据此显示「还没跑过评估」。
"""
import datetime as dt
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException

from app.core import jobs
from app.db import repository
from app.schemas.rageval import FaithCaseStatusRequest

ROOT = Path(__file__).resolve().parents[2]              # work/ch04
REPORT_JSON = ROOT.parent.parent / "data/ch04/reports/rag_eval.json"
REPORT_TXT = ROOT.parent.parent / "data/ch04/reports/rag_eval.txt"

# 均值只算四个答案桶;D_offtopic 不做四策略检索,不能混进来
RETRIEVAL_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
# 并列时的先后:与评估脚本 STRATEGIES 一致(vector 在前)
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")

router = APIRouter(prefix="/api/rag-eval")


def _iso(value: dt.datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value is not None else None


def _artifact(path: Path) -> dict:
    try:
        st = path.stat()
    except OSError:
        return {"present": False, "bytes": None, "mtime": None, "path": str(path)}
    return {
        "present": True,
        "bytes": st.st_size,
        "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "path": str(path),
    }


def _shape_problem(report: dict) -> str | None:
    """合法 JSON 也可能形状漂移(如 retrieval 是 list):确认到用到的层级。"""
    for key in ("meta", "retrieval", "evidence_coverage", "generation"):
        if key in report and not isinstance(report[key], dict):
            return (f"报告字段 {key} 形状异常:应为对象,"
                    f"实际为 {type(report[key]).__name__}")
    retrieval = report.get("retrieval") or {}
    for strategy, buckets in retrieval.items():
        if not isinstance(buckets, dict):
            return (f"报告字段 retrieval.{strategy} 形状异常:应为对象,"
                    f"实际为 {type(buckets).__name__}")
        for bucket, metrics in buckets.items():
            if not isinstance(metrics, dict):
                return (f"报告字段 retrieval.{strategy}.{bucket} 形状异常:"
                        f"应为对象,实际为 {type(metrics).__name__}")
    return None


def report_snapshot() -> dict:
    """读产物,不抛异常:kind = ok | missing | error。"""
    try:
        text = REPORT_JSON.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"kind": "missing", "report": None,
                "detail": f"报告不存在:{REPORT_JSON};先跑 make eval-rag"}
    except (OSError, ValueError) as e:
        return {"kind": "error", "report": None,
                "detail": f"报告读不出来:{type(e).__name__}: {e}"}
    try:
        report = json.loads(text)
    except ValueError as e:
        return {"kind": "error", "report": None,
                "detail": f"报告不是合法 JSON:{e}"}
    if not isinstance(report, dict):
        return {"kind": "error", "report": None,
                "detail": "报告顶层不是对象,无法解读"}
    problem = _shape_problem(report)
    if problem is not None:
        return {"kind": "error", "report": None,
                "detail": f"报告不可用:{problem}"}
    return {"kind": "ok", "report": report, "detail": None}


def _retrieval_overall(retrieval: dict) -> tuple[dict[str, float], dict[str, int]]:
    """四个答案桶 MRR 的算术平均;缺桶或 mrr 非数值时跳过该桶。

    返回 (均值, 每策略实际参与桶数),note 据后者如实写「聚合自 N 桶」。
    """
    overall: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    for strategy, buckets in retrieval.items():
        if not isinstance(buckets, dict):          # 形状已在 report_snapshot 挡过,再兜一层
            continue
        values: list[float] = []
        for bucket in RETRIEVAL_BUCKETS:           # 只认固定四桶
            entry = buckets.get(bucket)
            if not isinstance(entry, dict):
                continue
            mrr = entry.get("mrr")
            if isinstance(mrr, (int, float)) and not isinstance(mrr, bool):
                values.append(float(mrr))
        if values:
            overall[strategy] = sum(values) / len(values)
            bucket_counts[strategy] = len(values)
    return overall, bucket_counts


def _overall_note(bucket_counts: dict[str, int]) -> str:
    counts = set(bucket_counts.values())
    if len(counts) == 1:
        n = counts.pop()
    elif counts:
        n = f"{min(counts)}–{max(counts)}"
    else:
        n = "0"
    return f"聚合自 {n} 桶 MRR,不重算底层指标"


def _ordered_strategies(overall: dict[str, float], report: dict) -> list[str]:
    """固定策略顺序在前(并列判定用),报告里多出来的策略追加在后。"""
    listed = (report.get("meta") or {}).get("strategies") or []
    ordered = [s for s in STRATEGIES if s in overall]
    ordered += [s for s in listed if s in overall and s not in ordered]
    ordered += [s for s in overall if s not in ordered]
    return ordered


def overview_payload(snapshot: dict | None = None) -> dict:
    """GET /overview 与后台卡片共用的读数(产物缺失/读坏也返回结构化结果)。

    snapshot 可由调用方先取一次传入,避免后台卡为了区分 missing/error 重复读文件。
    """
    artifacts = {"json": _artifact(REPORT_JSON), "text": _artifact(REPORT_TXT)}
    job = {"spec": jobs.status("eval-rag")}
    if snapshot is None:
        snapshot = report_snapshot()
    if snapshot["kind"] != "ok":
        return {
            "present": False,
            "make": "make eval-rag",
            "hint": snapshot["detail"],
            "job": job,
            "artifacts": artifacts,
        }

    report = snapshot["report"]
    meta = report.get("meta") or {}
    retrieval = report.get("retrieval") or {}
    generation = report.get("generation") or {}
    overall, bucket_counts = _retrieval_overall(retrieval)
    ordered = _ordered_strategies(overall, report)
    buckets = meta.get("buckets")
    if not isinstance(buckets, list):
        buckets = []

    best_strategy: str | None = None
    best_mrr: float | None = None
    for strategy in ordered:                 # 顺序即并列规则:靠前者胜出
        mrr = overall[strategy]
        if best_mrr is None or mrr > best_mrr:
            best_strategy, best_mrr = strategy, mrr

    return {
        "present": True,
        "meta": meta,
        "retrieval": retrieval,
        "retrieval_overall": overall,
        "overall_note": _overall_note(bucket_counts),
        "evidence_coverage": report.get("evidence_coverage") or {},
        "generation": generation,
        "generation_done": bool(generation.get("completed")),
        "best": {"strategy": best_strategy, "mrr": best_mrr},
        "strategies": ordered,
        "buckets": buckets,
        "job": job,
        "artifacts": artifacts,
        "make": "make eval-rag",
    }


def _case(row) -> dict:
    return {
        "id": row.id,
        "eval_id": row.eval_id,
        "bucket": row.bucket,
        "query": row.query,
        "strategy": row.strategy,
        "answer": row.answer,
        "reason": row.reason,
        "citations": row.citations,
        "judge_model": row.judge_model,
        "status": row.status,
        "seen_count": row.seen_count,
        "resolution": row.resolution,
        "reopened": row.resolved_at is not None and row.status == "未解决",
        "first_seen_at": _iso(row.first_seen_at),
        "last_seen_at": _iso(row.last_seen_at),
        "resolved_at": _iso(row.resolved_at),
    }


@router.get("/overview")
async def overview() -> dict:
    return overview_payload()


@router.get("/faith-cases")
async def faith_cases(
    status: Literal["未解决", "已解决", "无需解决"] | None = None,
) -> dict:
    rows = await repository.list_faith_cases(status=status)
    counts = await repository.faith_case_counts()
    return {"items": [_case(r) for r in rows], "counts": counts}


@router.post("/faith-cases/{eval_id}/status")
async def set_faith_case_status(eval_id: str, req: FaithCaseStatusRequest) -> dict:
    if req.status != "未解决" and not req.resolution:
        raise HTTPException(status_code=400,
                            detail="标为已解决/无需解决必须填写处置说明")
    row = await repository.set_faith_case_status(
        eval_id, req.status, None if req.status == "未解决" else req.resolution)
    if row is None:
        raise HTTPException(status_code=404, detail=f"台账里没有这道题:{eval_id}")
    return {"ok": True, "item": _case(row)}
