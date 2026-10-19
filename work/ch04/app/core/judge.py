import asyncio
import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import ANSWER_JUDGE_PROMPT

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    covered: list[int] = Field(default_factory=list,
                               description="答案覆盖的要点编号,从 1 开始")
    faithful: bool = Field(description="答案事实是否都能被证据支撑")
    reason: str = Field(default="", description="一句话判断依据")


def _chain(model=None):
    # temperature=0:评估裁判必须可复现,同一输入重放不抖动。
    m = model or get_chat_model(temperature=0)
    return ANSWER_JUDGE_PROMPT | m.with_structured_output(
        JudgeVerdict, method="function_calling",
        extra_body={"thinking": {"type": "disabled"}},
    )


async def judge_answer(query: str, evidence: str, answer: str,
                       points: list[str], model=None) -> dict:
    """覆盖度 + 忠实度裁判。covered 去重排序并裁剪到 1..len(points)。

    裁判只服务离线评估:超时/上游错误不降级,记异常类型后上抛,
    由评估脚本按「失败跳过」处理;日志不含 query/evidence/answer。
    """
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
    try:
        r: JudgeVerdict = await asyncio.wait_for(
            _chain(model).ainvoke({
                "query": query, "evidence": evidence, "answer": answer,
                "points": numbered or "(无)",
            }),
            timeout=settings.judge_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - 上抛给评估脚本跳过该题
        # 只记类型名:parser/上游异常文本可能带用户原文或模型输出。
        logger.warning("answer judge failed (%s)", type(exc).__name__)
        raise
    valid = sorted({i for i in r.covered if 1 <= i <= len(points)})
    return {"covered": valid, "faithful": bool(r.faithful),
            "reason": r.reason or ""}
