import asyncio
import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import SELF_CHECK_PROMPT

logger = logging.getLogger(__name__)


class _Check(BaseModel):
    useful: bool = Field(description="证据是否足以回答")
    reason: str = Field(default="", description="判断依据")


def _chain(model=None):
    m = model or get_chat_model()
    return SELF_CHECK_PROMPT | m.with_structured_output(
        _Check, method="function_calling",
        extra_body={"thinking": {"type": "disabled"}},
    )


async def check_sufficient(query: str, evidence_texts: list[str],
                           model=None) -> dict:
    """生成前证据充分性自评。扁平字段 useful/reason(避开嵌套对象数组 502)。

    失败安全降级:上游报错/超时/输出不可解析时按 useful=True 放行——
    自评是辅助闸,不能因评审模型故障把本来可答的问题误拒;
    真正的「无证据」由检索机械闸(rerank_min_score)兜底。
    """
    evidence = "\n".join(
        f"[{i + 1}] {t}" for i, t in enumerate(evidence_texts)
    ) or "(无证据)"
    try:
        r: _Check = await asyncio.wait_for(
            _chain(model).ainvoke({"query": query, "evidence": evidence}),
            timeout=settings.selfcheck_timeout,
        )
        return {"useful": bool(r.useful), "reason": r.reason or ""}
    except Exception as exc:  # noqa: BLE001 - 自评失败不能拖垮 RAG 主链路
        # 只记类型名:parser/上游异常文本可能带用户原文或模型输出。
        logger.warning(
            "evidence self-check failed (%s), fail-open", type(exc).__name__
        )
        return {"useful": True, "reason": "self_check_unavailable"}
