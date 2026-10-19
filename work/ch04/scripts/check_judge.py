"""用标注样例重放裁判,量真实 DeepSeek 裁判与人工标注的一致率。
运行:make judge-check(需 .env 里的聊天上游)。

除 6 条标注样例外,还会把 faith_cases 里已处置的个案原样重放做校准:
- 已解决:人工确认确有问题,期望 faithful=false;
- 无需解决:人工判定不构成编造,期望 faithful=true。
faith_cases 为空(0 行)时跳过校准并正常退出;读库失败是基础设施故障,
stderr 报异常类型并以退出码 2 失败,不能伪装成通过;citations 为 NULL
的老个案无法重放,单独计数跳过,不计 FAIL。
"""
import asyncio
import json
import pathlib
import sys

from app.core.judge import judge_answer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests/data/judge_samples.jsonl"
EVAL_SET = ROOT / "tests/data/eval_ch04.jsonl"


async def _check_samples() -> tuple[int, int]:
    rows = [json.loads(ln) for ln in SAMPLES.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    ok = 0
    for s in rows:
        verdict = await judge_answer(
            s["query"], s["evidence"], s["answer"], s["points"])
        hit = verdict["faithful"] == s["expect_faithful"]
        ok += hit
        print(f"{'PASS' if hit else 'FAIL'} {s['note']}: "
              f"expected={s['expect_faithful']} got={verdict['faithful']} "
              f"reason={verdict['reason']}")
    print(f"\n{ok}/{len(rows)}")
    return ok, len(rows)


def _points_by_eval_id() -> dict[str, list[str]]:
    if not EVAL_SET.exists():
        return {}
    points: dict[str, list[str]] = {}
    for ln in EVAL_SET.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            s = json.loads(ln)
            points[s["id"]] = s.get("expect_points") or []
    return points


class _CalibrationUnavailable(RuntimeError):
    """读不到 faith_cases:基础设施故障,不能用「跳过」掩盖。"""


async def _check_faith_cases() -> tuple[int, int]:
    """台账里已处置个案原样重放;空表跳过,读库失败上抛(退出码 2)。"""
    from app.core import citations
    from app.db import repository

    try:
        rows = await repository.list_faith_cases()
    except Exception as exc:  # noqa: BLE001 - 转成明确的校准失败,由 main 退出 2
        print(f"faith_cases 读取失败 ({type(exc).__name__}),无法校准。",
              file=sys.stderr)
        raise _CalibrationUnavailable from exc

    if not rows:
        print("faith_cases 为空(0 行),跳过人工校准。")
        return 0, 0

    disposed = [r for r in rows if r.status in ("已解决", "无需解决")]
    if not disposed:
        print("faith_cases 无已处置个案(已解决/无需解决),跳过人工校准。")
        return 0, 0

    # DDL 允许老数据 citations 为 NULL;空证据下裁判必然判不忠实,是假阴性,不能计 FAIL。
    replayable = [r for r in disposed if r.citations]
    skipped = len(disposed) - len(replayable)
    if skipped:
        print(f"不可校准 {skipped} 条(citations 为空),已跳过。")
    if not replayable:
        return 0, 0

    points_by_id = _points_by_eval_id()
    ok = 0
    for r in replayable:
        # 原样重放:证据用落台账时的 Top-K 快照,答案用当时被判编造的答案。
        evidence = citations.build_evidence(r.citations)
        expect = r.status == "无需解决"
        verdict = await judge_answer(
            r.query, evidence, r.answer, points_by_id.get(r.eval_id, []))
        hit = verdict["faithful"] == expect
        ok += hit
        print(f"{'PASS' if hit else 'FAIL'} faith_cases[{r.eval_id}] {r.status}: "
              f"expected={expect} got={verdict['faithful']} "
              f"reason={verdict['reason']}")
    print(f"\nfaith_cases: {ok}/{len(replayable)}")
    return ok, len(replayable)


async def main() -> int:
    ok, total = await _check_samples()
    try:
        cal_ok, cal_total = await _check_faith_cases()
    except _CalibrationUnavailable:
        return 2
    return 0 if ok == total and cal_ok == cal_total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
