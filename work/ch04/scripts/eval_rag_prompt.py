"""对 RAG 生成 Prompt 跑标注样例(真实上游,需要 .env)。

默认 pytest 不调真实 LLM;本脚本是 Prompt 质量的独立验收入口。

判分:
- 引用:expect_citation 中每个编号都必须以 [n] 形式出现,且编号必须落在
  1 <= n <= len(evidence);答案里出现的任何 [n] 同样必须落在证据范围内
  (防模型凭空造角标);禁止 [1,2] 逗号写法。
- 内容:expect_any 至少命中一个;expect_all 每个都必须出现。
- 负面知识:forbid_patterns 是正则,命中任意一条即判 MISS(如具体到账/送达时效、
  赔偿金额);但命中点紧邻否定语境(如「无法保证 3 天内送达」)不算违规——
  否定的正是那条承诺,不能反判成 MISS。
- 拒答:refusal 样例(「没有查到」口径且无 expect_citation)必须拒答;若同时给出
  实质数值(如「3 天到账」「50 元」),判 MISS。
通过线:>= 5/6。
"""
import asyncio
import json
import pathlib
import re
import sys

from app.core.llm import get_chat_model
from app.core.prompts import RAG_ANSWER_PROMPT

SAMPLES = pathlib.Path(__file__).parent.parent / "tests/data/rag_prompt_samples.jsonl"
EXPECTED_TOTAL = 6
GEN_TIMEOUT = 60.0


def _answer_text(msg) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _evidence(items: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(items))


# 否定词 + 命中点之间允许的最大间隔(同一分句内、紧邻否定才算)。
# 不收裸「未」:「未来 3 天内送达」里的「未」不是否定,不能豁免违规承诺。
_NEGATION_WORDS = ("无法", "不能", "不会", "不可", "未能", "尚未", "未承诺",
                   "未保证", "难以", "没法", "没有", "不")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_COMMA_CITATION_RE = re.compile(r"\[\s*\d+\s*,\s*\d+\s*\]")
# 拒答里不该出现的「实质数值」:带单位的数量(价格/时效/规格等);
# 中文数字同样算(「三天」「五十元」),否则用汉字写答案就能绕过。
_SUBSTANTIVE_NUM_RE = re.compile(
    r"(?:\d+(?:\.\d+)?|[零一二三四五六七八九十百千万两]+)\s*"
    r"(?:元|块|天|日|小时|分钟|周|个?工作日|个月|月|年|%|℃|kg|g|ml|L|次|折|h|H)"
)
_REFUSAL_MARKERS = ("没有查到", "未查到", "无法回答", "不能回答", "暂无相关")
# 「不超过 3 天」「不少于 50 元」是对数值的限定,不是否定承诺;仍应算实质承诺。
_BOUND_PREFIXES = ("不超过", "不少于", "不低于", "不高于", "不多于")


def _negated(answer: str, start: int, window: int = 8) -> bool:
    """命中点前的紧邻分句里出现否定词 → 视为否定语境。

    「无法保证 3 天内送达」中的「3 天内」「保证…送达」都是在否定承诺;
    但「无法保证具体金额,比如 50 元」里的 50 元离否定词超过窗口,仍算实质数值。
    数值限定词(不超过/不少于…)不算否定。
    """
    prefix = answer[max(0, start - window):start]
    prefix = re.split(r"[。;；!！?？,，\n]", prefix)[-1]
    if any(bound in prefix for bound in _BOUND_PREFIXES):
        return False
    return any(word in prefix for word in _NEGATION_WORDS)


def _is_refusal(sample: dict) -> bool:
    """拒答样例:显式标记,或「无引用 + 拒答口径」的既有样例。"""
    if sample.get("refusal") or sample.get("id") == "refusal":
        return True
    expected = sample.get("expect_any") or []
    return not (sample.get("expect_citation") or []) and any(
        marker in word for word in expected for marker in _REFUSAL_MARKERS
    )


def _check(sample: dict, answer: str) -> tuple[bool, list[str]]:
    misses: list[str] = []
    evidence_len = len(sample.get("evidence") or [])
    for idx in sample.get("expect_citation") or []:
        if not 1 <= idx <= evidence_len:
            misses.append(f"期望引用[{idx}]越界(证据只有 {evidence_len} 条)")
        elif f"[{idx}]" not in answer:
            misses.append(f"缺引用[{idx}]")
    if _COMMA_CITATION_RE.search(answer):
        misses.append("引用用了逗号写法")
    # 答案里出现的每个 [n] 都必须能在证据列表里找到对应条目。
    bad = sorted({int(n) for n in _CITATION_RE.findall(answer)
                  if not 1 <= int(n) <= evidence_len})
    if bad:
        misses.append("引用了不存在的证据编号:" + ", ".join(f"[{n}]" for n in bad))
    if sample.get("expect_all"):
        for word in sample["expect_all"]:
            if word not in answer:
                misses.append(f"缺「{word}」")
    if sample.get("expect_any") and not any(
        word in answer for word in sample["expect_any"]
    ):
        misses.append("未命中期望表述:" + "/".join(sample["expect_any"]))
    for pat in sample.get("forbid_patterns") or []:
        for m in re.finditer(pat, answer):
            if _negated(answer, m.start()):
                continue
            misses.append(f"违规承诺「{m.group(0)}」")
            break
    if _is_refusal(sample):
        nums = [m.group(0) for m in _SUBSTANTIVE_NUM_RE.finditer(answer)
                if not _negated(answer, m.start())]
        if nums:
            misses.append("拒答样例给出了实质数值:" + "/".join(nums[:3]))
    return not misses, misses


async def main() -> int:
    samples = [
        json.loads(ln)
        for ln in SAMPLES.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    tot = len(samples)
    if tot != EXPECTED_TOTAL:
        print(f"样本数不符:期望 {EXPECTED_TOTAL},实际 {tot}", file=sys.stderr)
        return 2

    model = get_chat_model(temperature=0)  # 消除重放抖动
    chain = RAG_ANSWER_PROMPT | model
    ok = 0
    for s in samples:
        payload = {"query": s["query"], "evidence": _evidence(s["evidence"])}
        try:
            msg = await asyncio.wait_for(chain.ainvoke(payload), timeout=GEN_TIMEOUT)
            answer = _answer_text(msg).strip()
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整轮
            print("MISS", s["id"], s["query"], f"-> 调用失败({type(exc).__name__})")
            continue
        passed, misses = _check(s, answer)
        ok += passed
        print(
            "OK  " if passed else "MISS",
            s["id"],
            s["query"],
            "->",
            answer.replace("\n", " ")[:180],
        )
        if misses:
            print("      ", "; ".join(misses))
    print(f"Prompt 样例 {ok}/{tot}")
    return 1 if ok < 5 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
