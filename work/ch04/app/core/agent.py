# app/core/agent.py
import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import settings
from app.core import dsml
from app.core.llm import get_chat_model
from app.core.memory import trim_history
from app.core.prompts import AGENT_SYSTEM, CONVERGE_NUDGE, EVIDENCE_REFUSAL
from app.db import repository
from app.db.models import Message
from app.tools.infra import ToolRun, execute_tool_call
from app.tools.registry import get_all_tools


class ConversationNotFound(Exception):
    pass


@dataclass
class AgentResult:
    conversation_id: int
    answer: str
    tool_calls: list[dict]
    tool_runs: list[ToolRun]
    citations: list[dict] = field(default_factory=list)


def _faq_payload(run: ToolRun) -> dict | None:
    if getattr(run, "name", None) != "query_faq" or not getattr(run, "ok", False):
        return None
    try:
        data = json.loads(run.tool_message.content)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# evidence 是逐行 "[n] q: a",只重写行首编号,不误改问答正文里的方括号数字
_EVIDENCE_N_RE = re.compile(r"(?m)^\[(\d+)\]")


def _faq_signal(runs: list[ToolRun]) -> ToolRun | None:
    """本轮存在任一可用 query_faq 结果时正常收敛,一个都没有才拒答。

    可用 = 执行成功、无 evidence_insufficient 信号、且确实带证据。执行失败(ok=False)
    不算可用,但也不单独触发拒答——与不足混合且无可用结果时才拒。
    其它工具的同名 signal 不拦截;没有 query_faq 调用时返回 None。
    """
    faq_runs = [r for r in runs if getattr(r, "name", None) == "query_faq"]
    if not faq_runs:
        return None
    usable = any(
        getattr(r, "ok", False)
        and getattr(r, "signal", None) is None
        and getattr(r, "citations", None)
        for r in faq_runs
    )
    if usable:
        return None
    insufficient = next(
        (r for r in faq_runs
         if getattr(r, "signal", None) == "evidence_insufficient"),
        None,
    )
    return insufficient or faq_runs[0]


def _faq_citations(runs: list[ToolRun]) -> list[dict]:
    """拼接所有 query_faq run 的 citations(已在 _prepare_turn 全局编号)。"""
    out: list[dict] = []
    for r in runs:
        if getattr(r, "name", None) == "query_faq":
            out.extend(getattr(r, "citations", None) or [])
    return out


def _renumber_faq_citations(runs: list[ToolRun]) -> None:
    """多个 query_faq 调用共享同一份证据编号空间。

    按 run 顺序,第 2 个及以后的 run 的 citations[].n 与 evidence 里的 [k] 都加上
    前面 query_faq run 的证据条数偏移。必须在工具消息落库/回灌模型之前调用。
    """
    offset = 0
    for r in runs:
        if getattr(r, "name", None) != "query_faq" or not getattr(r, "ok", False):
            continue
        citations = getattr(r, "citations", None) or []
        if offset:
            for c in citations:
                if isinstance(c, dict) and isinstance(c.get("n"), int):
                    c["n"] += offset
            payload = _faq_payload(r)
            if payload is not None and isinstance(payload.get("evidence"), str):
                payload["evidence"] = _EVIDENCE_N_RE.sub(
                    lambda m: f"[{int(m.group(1)) + offset}]",
                    payload["evidence"],
                )
                r.tool_message.content = json.dumps(
                    payload, ensure_ascii=False, default=str)
        offset += len(citations)


async def _record_low_confidence(conversation_id: int, message: str,
                                 runs: list[ToolRun]) -> None:
    """拒答时落池:取本轮第一个 source=self_check 的 query_faq 结果;机械闸不落池。"""
    for run in runs:
        data = _faq_payload(run) or {}
        if data.get("source") == "self_check":
            await repository.insert_low_confidence(
                conversation_id, message, "self_check", data.get("reason") or None)
            return


def _text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    # content 为分块 list 时拼接文本片段
    return "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))


def _build_history(rows: list[Message]) -> list[BaseMessage]:
    """跨轮上下文:只回放 user 与「content 非空且非工具调用」的 assistant。
    带 tool_calls 的 assistant(常夹带 preamble)不跨轮回放,避免污染续接。
    SystemMessage 在 trim 之后前置:trim_messages 默认 include_system=False 会把它删掉。"""
    history: list[BaseMessage] = []
    for m in rows:
        if m.role == "user":
            history.append(HumanMessage(m.content or ""))
        elif m.role == "assistant" and m.content and not m.tool_calls:
            history.append(AIMessage(m.content))
    trimmed = trim_history(history, max_tokens=settings.token_budget)
    return [SystemMessage(AGENT_SYSTEM), *trimmed]


async def _prepare_turn(user_id, message, conversation_id, model):
    """共享编排前半:会话身份 → 落 user → 组装历史 → turn1 bind_tools 定工具 →
    落 assistant → 有工具则并行执行落 tool。返回 (cid, messages, ai, runs)。"""
    if conversation_id is None:
        conversation_id = await repository.create_conversation(user_id)
    elif await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)

    await repository.append_message(conversation_id, "user", content=message)
    rows = await repository.list_messages(conversation_id)
    messages = _build_history(rows)

    ai: AIMessage = await model.bind_tools(get_all_tools()).ainvoke(messages)
    # 落库先剥标记:泄漏的 DSML 绝不能进历史 —— 上一轮的机器标记会教坏下一轮。
    await repository.append_message(
        conversation_id, "assistant",
        content=dsml.strip_dsml(_text(ai)) or None, tool_calls=ai.tool_calls or None,
    )

    runs: list[ToolRun] = []
    if ai.tool_calls:
        runs = list(await asyncio.gather(
            *(execute_tool_call(tc, conversation_id) for tc in ai.tool_calls)
        ))
        # 多次 query_faq 的证据编号要先统一,再落库/回灌模型,否则第二轮 [1] 会指错源
        _renumber_faq_citations(runs)
        for r in runs:
            await repository.append_message(
                conversation_id, "tool",
                content=r.tool_message.content, tool_call_id=r.tool_call_id,
            )
    return conversation_id, messages, ai, runs


async def run_agent_turn(user_id, message, conversation_id, model=None) -> AgentResult:
    """非流式出口:一次性返回工具轨迹 + 最终回答(供 /api/agent、eval、单测)。"""
    model = model or get_chat_model()
    conversation_id, messages, ai, runs = await _prepare_turn(
        user_id, message, conversation_id, model)
    if not ai.tool_calls:
        return AgentResult(conversation_id, dsml.clean_answer(_text(ai)), [], [])
    signal = _faq_signal(runs)
    if signal is not None:
        await _record_low_confidence(conversation_id, message, runs)
        answer = EVIDENCE_REFUSAL
        await repository.append_message(conversation_id, "assistant", content=answer)
        return AgentResult(conversation_id, answer, ai.tool_calls, runs)
    final: AIMessage = await model.bind_tools(get_all_tools()).ainvoke(
        [*messages, ai, *(r.tool_message for r in runs), HumanMessage(CONVERGE_NUDGE)])
    # 收敛:绑定工具让模型能走原生 tool-call 通道(不绑时模型会把原生标记当纯文本吐 DSML 乱码),
    # 单轮保证靠「收敛永不执行工具」——_prepare_turn 之外没有工具执行代码路径,而非靠不 bind。
    # 收敛仍偶发泄漏(实测约 5.9%),clean_answer 负责剥标记 / 剥完没话就给兜底话术;
    # 顺带兜住「收敛又返回 tool_calls → 正文为空」的潜在空回答。
    answer = dsml.clean_answer(_text(final))
    await repository.append_message(conversation_id, "assistant", content=answer)
    return AgentResult(conversation_id, answer, ai.tool_calls, runs,
                       citations=_faq_citations(runs))


async def stream_agent_turn(user_id, message, conversation_id, model=None) -> AsyncIterator[dict]:
    """流式出口:产出事件 dict 供 /api/chat 转 SSE。收敛用 astream 逐 token,并 bind 工具 + 指令。"""
    model = model or get_chat_model(streaming=True)
    conversation_id, messages, ai, runs = await _prepare_turn(
        user_id, message, conversation_id, model)

    if not ai.tool_calls:
        # 无工具:turn1 文本即答案(assistant 已落库),整块吐出;同样先剥标记
        yield {"type": "delta", "text": dsml.clean_answer(_text(ai))}
        yield {"type": "done", "conversation_id": conversation_id}
        return

    for tc in ai.tool_calls:
        yield {"type": "tool", "name": tc.get("name") or ""}

    signal = _faq_signal(runs)
    if signal is not None:
        await _record_low_confidence(conversation_id, message, runs)
        yield {"type": "delta", "text": EVIDENCE_REFUSAL}
        await repository.append_message(
            conversation_id, "assistant", content=EVIDENCE_REFUSAL)
        yield {"type": "done", "conversation_id": conversation_id}
        return

    citations = _faq_citations(runs)
    if citations:
        yield {"type": "citations", "items": citations}

    chunks: list[str] = []
    gate = dsml.StreamGate()
    # 收敛:同 run_agent_turn,绑定工具 + 追问指令;单轮保证靠收敛不执行工具,而非不 bind。
    # 流式不能等收完再剥(用户已经看见了),故过一道滚动留尾的闸门:见标记即停吐(见 app/core/dsml.py)。
    async for chunk in model.bind_tools(get_all_tools()).astream(
        [*messages, ai, *(r.tool_message for r in runs), HumanMessage(CONVERGE_NUDGE)]):
        text = chunk.content if isinstance(chunk.content, str) else _text(chunk)
        if not text:
            continue
        for piece in gate.feed(text):
            chunks.append(piece)
            yield {"type": "delta", "text": piece}

    tail, leaked = gate.finish()
    for piece in tail:
        chunks.append(piece)
        yield {"type": "delta", "text": piece}
    if leaked or not "".join(chunks).strip():
        # 泄漏(已吐的干净前缀保留)或收敛又调工具导致什么都没吐 → 补一句人话
        chunks.append(dsml.DSML_FALLBACK)
        yield {"type": "delta", "text": dsml.DSML_FALLBACK}

    await repository.append_message(conversation_id, "assistant", content="".join(chunks))
    yield {"type": "done", "conversation_id": conversation_id}
