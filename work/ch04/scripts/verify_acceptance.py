"""验收四条的程序化检查(1 报告 / 2 BM25 / 3 引用 / 4 拒答+落池)。

不跑 9 分钟的全量评估:读现有产物 + 单次查询。需要 Milvus + 已建库 + MySQL +
.env 里的聊天上游。浏览器里点引用、👍/👎 反馈和落池 UI 仍由人工步骤确认。

运行:
  make verify-ch04
  # 直接跑时中文 Windows 显式 UTF-8(Makefile 已 export PYTHONUTF8=1):
  PYTHONUTF8=1 PYTHONPATH=. uv run python scripts/verify_acceptance.py

退出码:0 全部通过;1 任一项失败或前置依赖不可用。
"""
import asyncio
import json
import pathlib
import sys
from collections import Counter

from langchain_core.messages import ToolMessage
from sqlalchemy import select

from app.core import retrieval
from app.core.agent import _record_low_confidence
from app.db import base
from app.db.models import LowConfidenceQuestion
from app.kb import milvus_client
from app.tools.business import query_faq
from app.tools.infra import ToolRun

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"
REPORT = ROOT.parent.parent / "data/ch04/reports/rag_eval.json"
BUCKETS = {"A_policy", "B_model", "C_colloquial", "E_multi", "D_offtopic"}
ANSWERABLE = ("A_policy", "B_model", "C_colloquial", "E_multi")
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")

BTC_QUERY = "你们支持比特币付款吗"
PLUS_QUERY = "这边的plus会员有什么权益吗"
# plus 被机械闸先拦时的备用问法(README 同款):目标是无答案但能过分数闸。
FALLBACK_QUERY = "猫砂盆的滤芯多久换一次"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_eval_set() -> bool:
    """1a:评估集 100 题、五桶各 20 题(与报告口径对得上)。"""
    try:
        parsed = [
            json.loads(line)
            for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as e:  # noqa: BLE001 - 验收脚本要把失败讲清楚,不裸崩
        print(f"❌ 验收1a 评估集读取失败:{type(e).__name__}: {e}")
        return False
    rows = []
    for lineno, row in enumerate(parsed, 1):
        if not isinstance(row, dict):
            print(f"❌ 验收1a 评估集第 {lineno} 行不是对象:"
                  f"{type(row).__name__}")
            return False
        rows.append(row)
    counts = Counter(row.get("bucket") for row in rows)
    ok = (len(rows) == 100 and set(counts) == BUCKETS
          and all(counts[bucket] == 20 for bucket in BUCKETS))
    print(f"{'✅' if ok else '❌'} 验收1a 评估集: {len(rows)} 题,"
          f"分桶={dict(sorted(counts.items()))}")
    return ok


def _report_mrr_means(report: dict) -> dict[str, float]:
    means: dict[str, float] = {}
    retrieval = report.get("retrieval")
    if not isinstance(retrieval, dict):
        return means
    for strategy in STRATEGIES:
        buckets = retrieval.get(strategy)
        if not isinstance(buckets, dict):
            continue
        values = []
        for bucket in ANSWERABLE:
            cell = buckets.get(bucket)
            values.append(cell.get("mrr") if isinstance(cell, dict) else None)
        if all(_is_number(v) for v in values):
            means[strategy] = sum(values) / len(values)
    return means


def check_report() -> bool:
    """1b:四策略 × 四个可作答桶都有数字(不重跑评估)。"""
    if not REPORT.exists():
        print(f"❌ 验收1b 报告不存在:{REPORT};先跑 make eval-rag")
        return False
    try:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"❌ 验收1b 报告读取失败:{type(e).__name__}: {e}")
        return False
    if not isinstance(report, dict):
        print(f"❌ 验收1b 报告顶层不是对象:{type(report).__name__}")
        return False
    retrieval = report.get("retrieval")
    if not isinstance(retrieval, dict):
        print(f"❌ 验收1b retrieval 形状异常:"
              f"{type(retrieval).__name__}(应为对象)")
        return False

    problems: list[str] = []
    coverage = report.get("evidence_coverage")
    if not isinstance(coverage, dict):
        problems.append(f"evidence_coverage 形状异常:{type(coverage).__name__}")
    elif set(coverage) != set(STRATEGIES):
        problems.append(f"evidence_coverage 策略不符:{sorted(coverage)}")
    for strategy in STRATEGIES:
        buckets = retrieval.get(strategy)
        if not isinstance(buckets, dict):
            problems.append(f"{strategy}:无 retrieval 数据")
            continue
        for bucket in ANSWERABLE:
            cell = buckets.get(bucket)
            if not isinstance(cell, dict):
                problems.append(f"{strategy}/{bucket}:缺桶")
                continue
            for metric in ("recall_at_5", "mrr"):
                if not _is_number(cell.get(metric)):
                    problems.append(f"{strategy}/{bucket}:{metric} 非数字")
            if type(cell.get("n")) is not int or cell["n"] <= 0:
                problems.append(f"{strategy}/{bucket}:n={cell.get('n')!r}")

    ok = not problems
    means = _report_mrr_means(report)
    mean_text = " ".join(f"{st}={means[st]:.3f}" for st in STRATEGIES if st in means)
    if ok:
        print(f"✅ 验收1b 四策略报告: 4 策略 × 4 桶均有数字;MRR均值 {mean_text}")
    else:
        print(f"❌ 验收1b 四策略报告: " + "; ".join(problems[:6])
              + (f" 等 {len(problems)} 处" if len(problems) > 6 else ""))
    return ok


async def check_bm25() -> bool:
    """2:B 桶型号问题 BM25 Top-1 必须命中 MH-W40。"""
    query = "MH-W40 滤芯多久换一次"
    try:
        hits = await retrieval.search_knowledge(query, strategy="bm25", top_k=5)
    except Exception as e:  # noqa: BLE001
        print(f"❌ 验收2 BM25:{type(e).__name__}: {e}")
        return False
    if not isinstance(hits, list) or not all(isinstance(h, dict) for h in hits):
        print(f"❌ 验收2 BM25:检索结果形状异常:{type(hits).__name__}")
        return False
    top1 = (hits[0].get("section_path") or "") if hits else ""
    ok = "MH-W40" in top1
    print(f"{'✅' if ok else '❌'} 验收2 B 桶 BM25: Top-1={top1!r}, hits={len(hits)}")
    return ok


async def check_citations() -> bool:
    """3:可回答问题 → sufficient=True,citations 从 1 连续编号且可定位原文。"""
    query = "邮费是多少"
    try:
        out = await query_faq.ainvoke({"keyword": query})
    except Exception as e:  # noqa: BLE001
        print(f"❌ 验收3 引用:{type(e).__name__}: {e}")
        return False
    if not isinstance(out, dict):
        print(f"❌ 验收3 引用:query_faq 返回形状异常:{type(out).__name__}")
        return False
    cites = out.get("citations") or []
    if not isinstance(cites, list) or not all(isinstance(c, dict) for c in cites):
        print(f"❌ 验收3 引用:citations 形状异常:{type(cites).__name__}")
        return False
    numbers = [c.get("n") for c in cites]
    ok = (
        out.get("sufficient") is True
        and bool(cites)
        and numbers == list(range(1, len(cites) + 1))
        and all(c.get("chunk_id") and (c.get("section_path") or "").strip()
                for c in cites)
    )
    first = cites[0].get("section_path") if cites else None
    print(f"{'✅' if ok else '❌'} 验收3 引用映射: sufficient={out.get('sufficient')}, "
          f"citations={len(cites)} 条, n={numbers[:3]}"
          f"{'...' if len(numbers) > 3 else ''}, 首条 section={first!r}")
    return ok


async def _pool_rows(question: str) -> list[LowConfidenceQuestion]:
    async with base.async_session() as session:
        result = await session.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.raw_question == question)
        )
        return list(result.scalars())


def _tool_run(payload: dict) -> ToolRun:
    """把 query_faq 的返回包成 agent 眼里的 ToolRun,走同一条落池代码。"""
    content = json.dumps(payload, ensure_ascii=False)
    return ToolRun(
        tool_call_id="verify-acceptance-selfcheck",
        name="query_faq",
        ok=True,
        tool_message=ToolMessage(
            content=content,
            tool_call_id="verify-acceptance-selfcheck",
            name="query_faq",
        ),
        signal=payload.get("signal"),
    )


async def _ensure_selfcheck_pooled(query: str, out: dict) -> tuple[bool, str]:
    """self_check 结果落池并回查(前后快照,允许池中已有历史同题行)。

    query_faq 本身不写池,这里复用 agent 的 helper;判定只要求:
    行数不减少,且出现新行或已存在 self_check 行。
    """
    before = await _pool_rows(query)
    if not any(row.source == "self_check" for row in before):
        await _record_low_confidence(None, query, [_tool_run(out)])
    after = await _pool_rows(query)
    before_ids = {row.id for row in before}
    new_rows = [row for row in after if row.id not in before_ids]
    ok = (len(after) >= len(before)
          and (bool(new_rows) or any(row.source == "self_check" for row in after)))
    if new_rows:
        note = f"池中 {len(before)}→{len(after)} 条(新增 {len(new_rows)})"
    elif after:
        note = f"池中 {len(before)}→{len(after)} 条(已有同题行,未重复插入)"
    else:
        note = f"池中 {len(before)}→{len(after)} 条(未落池)"
    return ok, note


async def check_refusal_and_pool() -> bool:
    """4:库外问题拒答;机械闸不落池,self_check 才落池(两套口径)。

    用调用前后快照判定「本次没有新增」:池里可能已有用户浏览器验证时留下的
    self_check 历史行,不能把历史行误当成这次调用写进去的。
    """
    try:
        btc_before = len(await _pool_rows(BTC_QUERY))
        btc = await query_faq.ainvoke({"keyword": BTC_QUERY})
        if not isinstance(btc, dict):
            print(f"❌ 验收4 拒答+落池:query_faq 返回形状异常:"
                  f"{type(btc).__name__}")
            return False
        # 用 agent 同一条落池 helper 跑一遍:机械闸结果必须被 source 守卫拒绝入池。
        await _record_low_confidence(None, BTC_QUERY, [_tool_run(btc)])
        btc_after = len(await _pool_rows(BTC_QUERY))
        ok_btc = (btc.get("sufficient") is False
                  and btc.get("source") == "retrieval_low_conf")
        ok_not_pooled = btc_after == btc_before

        plus = await query_faq.ainvoke({"keyword": PLUS_QUERY})
        if not isinstance(plus, dict):
            print(f"❌ 验收4 拒答+落池:query_faq 返回形状异常:"
                  f"{type(plus).__name__}")
            return False
        plus_source = plus.get("source")
        if plus_source == "self_check":
            ok_plus, snapshot = await _ensure_selfcheck_pooled(PLUS_QUERY, plus)
            plus_note = f"plus self_check {snapshot}"
        elif plus.get("sufficient") is False:
            # 机械闸先拦:按 README 的 fallback 换一个能过分数闸的问法,
            # 实测 self_check 路径与落池,而不是只看历史记录。
            fallback = await query_faq.ainvoke({"keyword": FALLBACK_QUERY})
            if not isinstance(fallback, dict):
                print(f"❌ 验收4 拒答+落池:fallback 返回形状异常:"
                      f"{type(fallback).__name__}")
                return False
            fallback_source = fallback.get("source")
            if fallback_source == "self_check":
                ok_plus, snapshot = await _ensure_selfcheck_pooled(
                    FALLBACK_QUERY, fallback)
                plus_note = (f"plus 被机械闸先拦(source={plus_source});"
                             f"fallback「{FALLBACK_QUERY}」self_check {snapshot}")
            else:
                ok_plus = False
                plus_note = (f"plus 被机械闸先拦(source={plus_source});"
                             f"fallback「{FALLBACK_QUERY}」"
                             f"source={fallback_source},未触发 self_check")
        else:
            ok_plus = False
            plus_note = "问题被判可答,未触发任何拒答闸"

        ok = ok_btc and ok_not_pooled and ok_plus
        print(f"{'✅' if ok else '❌'} 验收4 拒答+落池: "
              f"bitcoin source={btc.get('source')}"
              f"(池中 {btc_before}→{btc_after} 条,应不新增);"
              f"{plus_note}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"❌ 验收4 拒答+落池:{type(e).__name__}: {e}")
        return False


def _preflight_milvus() -> bool:
    try:
        client = milvus_client.get_client()
        milvus_client.ensure_collection(client)
        count = milvus_client.count(client)
    except Exception as e:  # noqa: BLE001
        print(f"❌ 前置检查 Milvus 不可用:{type(e).__name__}: {e}")
        return False
    if count == 0:
        print("❌ 知识库为空:先 make kb-revectorize && make kb-vectorize")
        return False
    print(f"Milvus OK:{count} 块")
    return True


async def _preflight_mysql() -> bool:
    try:
        await _pool_rows("__verify_acceptance_preflight__")
    except Exception as e:  # noqa: BLE001
        print(f"❌ 前置检查 MySQL 不可用:{type(e).__name__}: {e}")
        return False
    return True


async def main() -> int:
    if not _preflight_milvus() or not await _preflight_mysql():
        return 1
    results = [
        check_eval_set(),
        check_report(),
        await check_bm25(),
        await check_citations(),
        await check_refusal_and_pool(),
    ]
    print(f"\n{sum(results)}/{len(results)} 项通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
