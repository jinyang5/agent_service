# app/tools/infra.py
import asyncio
import json
import logging
from dataclasses import dataclass

from langchain_core.messages import ToolMessage

from app.tools import registry

logger = logging.getLogger(__name__)


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage


def _error_run(tc_id: str, name: str, msg: str) -> ToolRun:
    return ToolRun(
        tool_call_id=tc_id,
        name=name,
        ok=False,
        tool_message=ToolMessage(
            content=f"工具执行失败:{msg}", tool_call_id=tc_id, name=name, status="error"
        ),
    )


async def execute_tool_call(
    tool_call: dict, conversation_id: int, timeout: float = 5.0, max_retries: int = 1
) -> ToolRun:
    name = tool_call["name"]
    tc_id = tool_call["id"]
    args = dict(tool_call.get("args") or {})

    tool = registry.get_tool(name)
    if tool is None:
        return _error_run(tc_id, name, f"未知工具 {name}")

    # Fallback 注入(InjectedToolArg 生效时 INJECT_CONVERSATION 可为空,这段不改变行为)
    if name in registry.INJECT_CONVERSATION:
        args["conversation_id"] = conversation_id

    retries = 0 if name in registry.NO_RETRY else max_retries
    attempt = 0
    while True:
        try:
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
            content = json.dumps(result, ensure_ascii=False, default=str)
            return ToolRun(
                tool_call_id=tc_id,
                name=name,
                ok=True,
                tool_message=ToolMessage(content=content, tool_call_id=tc_id, name=name),
            )
        except Exception as e:  # noqa: BLE001 - 统一兜底转错误回灌
            attempt += 1
            if attempt > retries:
                logger.exception("工具执行失败 name=%s", name)
                return _error_run(tc_id, name, type(e).__name__)
            await asyncio.sleep(0.2 * attempt)
