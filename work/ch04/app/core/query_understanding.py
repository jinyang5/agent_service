import asyncio
import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


def _chain(model=None):
    m = model or get_chat_model()
    # ch03 R11:思考模式下 function_calling 强制 tool_choice 会 400;
    # 关掉 thinking 再走 function_calling 是唯一实测可走的扁平结构化输出路径。
    return QUERY_REWRITE_PROMPT | m.with_structured_output(
        _Rewrite, method="function_calling",
        extra_body={"thinking": {"type": "disabled"}},
    )


async def understand(query: str, model=None) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str]),避开嵌套对象数组 502。

    失败安全降级:上游报错/超时/输出不可解析时回退原句,绝不把异常抛进聊天主链路。
    """
    try:
        r: _Rewrite = await asyncio.wait_for(
            _chain(model).ainvoke({"query": query}),
            timeout=settings.query_understanding_timeout,
        )
        standard = (r.standard or "").strip() or query
        expanded = [w.strip() for w in (r.expanded or []) if w and w.strip()]
        return {"standard": standard, "expanded": expanded}
    except Exception as exc:  # noqa: BLE001 - 检索改写失败不能拖垮聊天主链路
        # 只记类型名:parser 异常的 str 可能带模型输出/用户原文,不落日志。
        logger.warning(
            "query rewrite failed (%s), fallback to raw query", type(exc).__name__
        )
        return {"standard": query, "expanded": []}
