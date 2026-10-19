"""四策略 RAG 评估:检索段 + 证据覆盖度(确定性)+ 生成段(DeepSeek 裁判)。

用法:
  PYTHONPATH=. uv run python scripts/eval_ch04.py                 # 全量 100 题
  PYTHONPATH=. uv run python scripts/eval_ch04.py --limit 5       # 每桶 5 题
  PYTHONPATH=. uv run python scripts/eval_ch04.py --skip-generation
产物:
  data/ch04/reports/rag_eval.txt  人读
  data/ch04/reports/rag_eval.json 机读
"""
import argparse
import asyncio
import datetime as dt
import json
import logging
import pathlib
import sys
from collections import defaultdict

from app.config import settings
from app.core import citations, eval_metrics, retrieval
from app.core.judge import judge_answer
from app.core.llm import get_chat_model
from app.core.prompts import RAG_ANSWER_PROMPT
from app.db import repository
from app.kb import milvus_client
from app.tools.business import query_faq

logger = logging.getLogger("eval_ch04")

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"
OUT_DIR = ROOT.parent.parent / "data/ch04/reports"
OUT_TXT = OUT_DIR / "rag_eval.txt"
OUT_JSON = OUT_DIR / "rag_eval.json"

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
RETRIEVAL_BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi"]
ANSWERABLE = ["A_policy", "B_model", "C_colloquial"]
RETRIEVAL_DEPTH = 10
RECALL_K = 5
RETR_CONCURRENCY = 4
GEN_CONCURRENCY = 2
GEN_TIMEOUT = 60.0
REFUSAL_TIMEOUT = 120.0

_LINES: list[str] = []
_ERRORS: list[str] = []


def _log(line: str = "") -> None:
    print(line, flush=True)
    _LINES.append(line)


def _positive_limit(value: str) -> int:
    limit = int(value)
    if limit < 1:
        raise argparse.ArgumentTypeError("--limit 必须 >= 1")
    return limit


def _load(limit: int | None) -> tuple[list[dict], int]:
    rows = [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    return eval_metrics.sample_per_bucket(rows, limit), len(rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _preflight() -> int:
    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    n = milvus_client.count(client)
    if n == 0:
        raise SystemExit("Milvus 集合为空:先 make kb-revectorize && make kb-vectorize")
    return n


async def _retrieve_all(samples: list[dict]) -> dict[tuple[str, str], list[dict]]:
    sem = asyncio.Semaphore(RETR_CONCURRENCY)

    async def one(sample: dict, strategy: str) -> list[dict]:
        async with sem:
            return await retrieval.search_knowledge(
                sample["query"], strategy=strategy, top_k=RETRIEVAL_DEPTH)

    tasks = {}
    for s in samples:
        if s["bucket"] == "D_offtopic":
            continue          # D 只用 query_faq 的线上拒答路径,不做四策略检索
        for st in STRATEGIES:
            tasks[(s["id"], st)] = asyncio.create_task(one(s, st))
    out: dict[tuple[str, str], list[dict]] = {}
    for key, task in tasks.items():
        try:
            out[key] = await task
        except Exception as e:
            out[key] = []
            _ERRORS.append(f"检索失败 {key}: {type(e).__name__}: {e}")
    return out


def _retrieval_reports(samples: list[dict],
                       hits: dict[tuple[str, str], list[dict]]) -> dict:
    buckets = defaultdict(lambda: defaultdict(list))
    for s in samples:
        if s["bucket"] not in RETRIEVAL_BUCKETS:
            continue
        for st in STRATEGIES:
            h = hits[(s["id"], st)]
            buckets[st][s["bucket"]].append(
                (eval_metrics.recall_at_k(s, h, RECALL_K), eval_metrics.mrr(s, h)))
    report = {}
    for st in STRATEGIES:
        report[st] = {}
        for b in RETRIEVAL_BUCKETS:
            vals = buckets[st][b]
            report[st][b] = {
                "recall_at_5": _mean([v[0] for v in vals]),
                "mrr": _mean([v[1] for v in vals]),
                "n": len(vals),
            }
    return report


def _coverage_reports(samples: list[dict],
                      hits: dict[tuple[str, str], list[dict]]) -> dict:
    report = {}
    for st in STRATEGIES:
        per_bucket = defaultdict(list)
        for s in samples:
            if s["bucket"] in ANSWERABLE:
                per_bucket[s["bucket"]].append(
                    eval_metrics.evidence_coverage(s, hits[(s["id"], st)]))
        all_vals = [v for b in ANSWERABLE for v in per_bucket[b]]
        report[st] = {b: _mean(per_bucket[b]) for b in ANSWERABLE}
        report[st]["overall"] = _mean(all_vals)
    return report


def _answer_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))


async def _generation(samples: list[dict],
                      hits: dict[tuple[str, str], list[dict]],
                      skip: bool) -> dict:
    out = {
        "completed": False,
        "answer_coverage": {},
        "faithfulness": {},
        "refusal": {},
        "faithfulness_cases": [],
        "errors": [],
    }
    if skip:
        out["errors"].append("--skip-generation:生成段未跑")
        return out

    answer_model = get_chat_model(temperature=0)
    judge_model = get_chat_model(temperature=0)
    gen_sem = asyncio.Semaphore(GEN_CONCURRENCY)
    answer_chain = RAG_ANSWER_PROMPT | answer_model

    async def gen_one(s: dict, st: str) -> dict | None:
        h = hits[(s["id"], st)]
        cites = citations.build_citations(h)
        evidence = citations.build_evidence(cites)
        async with gen_sem:
            try:
                msg = await asyncio.wait_for(
                    answer_chain.ainvoke({"query": s["query"], "evidence": evidence}),
                    timeout=GEN_TIMEOUT)
                answer = _answer_text(msg)
                verdict = await asyncio.wait_for(
                    judge_answer(s["query"], evidence, answer,
                                 s.get("expect_points") or [], model=judge_model),
                    timeout=GEN_TIMEOUT)
            except Exception as e:
                out["errors"].append(
                    f"生成/裁判失败 {s['id']}/{st}: {type(e).__name__}: {e}")
                return None
        coverage = (len(verdict["covered"]) / len(s["expect_points"])
                    if s.get("expect_points") else None)
        return {"answer": answer, "evidence": evidence, "cites": cites,
                "verdict": verdict, "coverage": coverage}

    async def d_refusal(s: dict) -> dict:
        async with gen_sem:
            try:
                payload = await asyncio.wait_for(
                    query_faq.ainvoke({"keyword": s["query"]}),
                    timeout=REFUSAL_TIMEOUT)
            except Exception as e:
                out["errors"].append(
                    f"D 拒答检查失败 {s['id']}: {type(e).__name__}: {e}")
                return {"id": s["id"], "refused": False, "source": "error",
                        "reason": str(e)}
        return {"id": s["id"], "refused": payload.get("sufficient") is False,
                "source": payload.get("source") or ("answerable" if payload.get("sufficient") else "unknown"),
                "reason": payload.get("reason") or ""}

    tasks: list[tuple[tuple[str, str] | str, object]] = []
    for s in samples:
        if s["bucket"] in ANSWERABLE:
            for st in STRATEGIES:
                if not hits.get((s["id"], st)):
                    # 检索失败或零命中:空证据会生成"答案"并被判不忠实、写进
                    # faith_cases 台账,污染评估;这里跳过生成并显式记录。
                    out["errors"].append(
                        f"检索失败/无结果,跳过生成 {s['id']}/{st}")
                    continue
                tasks.append(((s["id"], st), gen_one(s, st)))
        elif s["bucket"] == "D_offtopic":
            tasks.append((s["id"], d_refusal(s)))
    keys = [k for k, _ in tasks]
    results = await asyncio.gather(*(c for _, c in tasks), return_exceptions=True)
    res_by_key = dict(zip(keys, results))
    # completed 只认 A/B/C 生成任务的落地成功;D 桶拒答成功不能冒充生成段完成。
    answerable_keys = {(s["id"], st) for s in samples if s["bucket"] in ANSWERABLE
                       for st in STRATEGIES}

    # 聚合成四策略 × 桶的答案覆盖度
    per = defaultdict(lambda: defaultdict(list))
    faithful = defaultdict(list)          # 桶 → [0/1];只一层分组,值必须是 list
    cases, seen = [], set()
    for s in samples:
        if s["bucket"] in ANSWERABLE:
            for st in STRATEGIES:
                r = res_by_key.get((s["id"], st))
                if isinstance(r, Exception):
                    out["errors"].append(
                        f"生成任务异常 {s['id']}/{st}: {type(r).__name__}: {r}")
                    continue
                if r is None:
                    continue
                if r["coverage"] is not None:
                    per[st][s["bucket"]].append(r["coverage"])
                if st == "hybrid_rerank":
                    faithful[s["bucket"]].append(1.0 if r["verdict"]["faithful"] else 0.0)
                    if not r["verdict"]["faithful"] and s["id"] not in seen:
                        seen.add(s["id"])
                        cases.append({
                            "eval_id": s["id"], "bucket": s["bucket"],
                            "query": s["query"], "reason": r["verdict"]["reason"],
                            "answer": r["answer"],
                        })
                        try:
                            await repository.upsert_faith_case(
                                s["id"], s["bucket"], s["query"], "hybrid_rerank",
                                r["answer"], r["verdict"]["reason"],
                                r["cites"], settings.chat_model)
                        except Exception as e:
                            logger.warning("faith_cases 写入失败 %s: %s", s["id"], e)
                            out["errors"].append(
                                f"faith_cases 写入失败 {s['id']}: "
                                f"{type(e).__name__}: {e}")
        elif s["bucket"] == "D_offtopic":
            r = res_by_key.get(s["id"])
            if isinstance(r, Exception):
                out["errors"].append(
                    f"D 拒答任务异常 {s['id']}: {type(r).__name__}: {r}")
                continue
            if r is not None:
                out["refusal"].setdefault("details", []).append(r)

    out["answer_coverage"] = {}
    for st in STRATEGIES:
        vals = [v for b in ANSWERABLE for v in per[st][b]]
        out["answer_coverage"][st] = {b: _mean(per[st][b]) for b in ANSWERABLE}
        out["answer_coverage"][st]["overall"] = _mean(vals)

    out["faithfulness"] = {"hybrid_rerank": {}}
    all_f = [v for b in ANSWERABLE for v in faithful[b]]
    for b in ANSWERABLE:
        out["faithfulness"]["hybrid_rerank"][b] = {
            "rate": _mean(faithful[b]), "n": len(faithful[b])}
    out["faithfulness"]["hybrid_rerank"]["overall"] = {
        "rate": _mean(all_f), "n": len(all_f)}
    out["faithfulness_cases"] = cases

    details = out["refusal"].pop("details", [])
    sources = defaultdict(int)
    for d in details:
        sources[d["source"]] += 1
    # 基础设施故障(source="error")不是「没拒答」:rate/n 只看有效检查;
    # 故障仍保留在 details/sources/errors 里可追溯。
    ok_details = [d for d in details if d["source"] != "error"]
    out["refusal"]["D_offtopic"] = {
        "rate": (_mean([1.0 if d["refused"] else 0.0 for d in ok_details])
                 if ok_details else None),
        "n": len(ok_details),
        "sources": dict(sources),
        "details": details,
    }
    # 按 A/B/C 生成任务的实际完成判定:全部失败/跳过(含检索失败)时为 False。
    # 不能把 D 桶拒答也算进来——拒答成功只说明 D 段跑了,生成段可能一条都没成。
    gen_results = [r for k, r in zip(keys, results) if k in answerable_keys]
    out["completed"] = any(
        r is not None and not isinstance(r, Exception)
        and not (isinstance(r, dict) and r.get("source") == "error")
        for r in gen_results)
    return out


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _write_report(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    lines = ["Ch04 RAG 评估报告", "=" * 72, ""]
    meta = report["meta"]
    lines.append(f"时间:{meta['generated_at']}")
    lines.append(f"题数:{meta['eval_set_size']}(本轮 {meta['sampled']};limit={meta['limit']})")
    lines.append(f"KB 块数:{meta['kb_chunks']};嵌入:{meta['embed_model']};"
                 f"重排:{meta['rerank_model']};裁判:{meta['judge_model']}")
    lines.append("")
    lines.append("## 检索(Recall@5 / MRR)")
    for st in STRATEGIES:
        lines.append(f"- {st}")
        for b in RETRIEVAL_BUCKETS:
            m = report["retrieval"][st][b]
            lines.append(f"    {b:14s} Recall@5={_fmt(m['recall_at_5'])} "
                         f"MRR={_fmt(m['mrr'])} n={m['n']}")
    lines.append("")
    lines.append("## 证据覆盖度(机械匹配 expect_points)")
    for st in STRATEGIES:
        m = report["evidence_coverage"][st]
        lines.append(f"- {st}: " + " ".join(
            f"{b}={_fmt(m[b])}" for b in ANSWERABLE + ["overall"]))
    lines.append("")
    lines.append("## 生成段")
    g = report["generation"]
    if g.get("completed"):
        lines.append("答案覆盖度:")
        for st in STRATEGIES:
            m = g["answer_coverage"][st]
            lines.append(f"- {st}: " + " ".join(
                f"{b}={_fmt(m[b])}" for b in ANSWERABLE + ["overall"]))
        f = g["faithfulness"]["hybrid_rerank"]
        lines.append("Faithfulness(hybrid_rerank): " + " ".join(
            f"{b}={_fmt(f[b]['rate'])}(n={f[b]['n']})"
            for b in ANSWERABLE + ["overall"]))
        r = g["refusal"]["D_offtopic"]
        lines.append(f"D 桶拒答率:{_fmt(r['rate'])} (n={r['n']}, sources={r['sources']})")
        lines.append(f"Faithfulness 个案:{len(g['faithfulness_cases'])} 条")
    else:
        lines.append("本次未完成(--skip-generation 或上游失败);检索段与证据覆盖度仍然有效")
    if g.get("errors"):
        lines.append("")
        lines.append("## 错误")
        lines.extend(f"- {e}" for e in g["errors"])
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=_positive_limit, default=None,
                        help="每个桶最多取 N 题(默认全量,须 >= 1)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="只跑确定性检索段与证据覆盖度")
    args = parser.parse_args()

    _LINES.clear()
    _ERRORS.clear()

    samples, total = _load(args.limit)
    kb_chunks = _preflight()
    _log(f"评估集 {total} 题,本轮 {len(samples)} 题;KB {kb_chunks} 块")

    hits = await _retrieve_all(samples)
    retrieval_report = _retrieval_reports(samples, hits)
    coverage_report = _coverage_reports(samples, hits)
    _log("检索段与证据覆盖度完成(零 LLM)")

    try:
        generation = await _generation(samples, hits, args.skip_generation)
    except Exception as e:
        _ERRORS.append(f"生成段整体失败:{type(e).__name__}: {e}")
        # 检索段错误统一在下方与 generation.errors 合并,这里留空避免重复。
        generation = {"completed": False, "answer_coverage": {},
                      "faithfulness": {}, "refusal": {},
                      "faithfulness_cases": [], "errors": []}

    report = {
        "meta": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "eval_set_size": total,
            "sampled": len(samples),
            "limit": args.limit,
            "strategies": STRATEGIES,
            "buckets": RETRIEVAL_BUCKETS + ["D_offtopic"],
            "retrieval_depth": RETRIEVAL_DEPTH,
            "recall_k": RECALL_K,
            "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model,
            "judge_model": settings.chat_model,
            "kb_chunks": kb_chunks,
        },
        "retrieval": retrieval_report,
        "evidence_coverage": coverage_report,
        "generation": generation,
    }
    if _ERRORS:
        report["generation"].setdefault("errors", []).extend(_ERRORS)
    _write_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
