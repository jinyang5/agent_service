import logging
from datetime import datetime

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db import repository
from app.kb import dedup, dualwrite
from app.kb.documents import Chunk

logger = logging.getLogger(__name__)


class QaPair(BaseModel):
    question: str
    answer: str


class QaExtraction(BaseModel):
    """结构化输出 schema:用两个并列标量数组一一对应。
    嵌套对象数组(list[对象])在部分 OpenAI 兼容上游会 502,标量 list[str] 稳,故并列数组规避。"""

    questions: list[str] = Field(default_factory=list,
                                 description="问法列表,与 answers 一一对应;无可复用问答则为空")
    answers: list[str] = Field(default_factory=list,
                               description="答案列表,与 questions 一一对应")


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    model = model or get_chat_model()
    # 上游 deepseek-v4-flash 是思考模型,结构化输出只有这一条实测可走的路(2026-10-15 实测):
    #   ① 默认 json_schema response_format → 400「This response_format type is unavailable now」;
    #   ② 思考模式下 function_calling 强制 tool_choice → 400「Thinking mode does not support this tool_choice」;
    #   ③ 关掉思考(extra_body thinking.type=disabled)再走 function_calling → 通,返回校验过的 QaExtraction。
    # 抽取这种固定 schema 的任务不需要思考链,关掉还更快更省。上游不认 thinking 字段时按惯例静默忽略。
    chain = MINING_PROMPT | model.with_structured_output(
        QaExtraction, method="function_calling",
        extra_body={"thinking": {"type": "disabled"}},
    )
    result: QaExtraction = await chain.ainvoke({"conversations": "\n---\n".join(conversation_texts)})
    qs, ans = result.questions, result.answers
    n = min(len(qs), len(ans))
    if len(qs) != len(ans):
        logger.warning("并列数组长度不一致 questions=%d answers=%d,按较短对齐", len(qs), len(ans))
    return [QaPair(question=qs[i], answer=ans[i]) for i in range(n)]


async def _load_conversation_texts() -> list[tuple[str, str]]:
    """返回 [(source_ref, 对话文本)];对话文本由该会话的 user/assistant 消息拼成。"""
    convs = await repository.list_conversations_with_messages()
    out: list[tuple[str, str]] = []
    for conv_id, msgs in convs:
        lines = [f"{m.role}: {m.content}" for m in msgs if m.content]
        if lines:
            out.append((f"conv:{conv_id}", "\n".join(lines)))
    return out


async def mine(batch_size: int = 20, model=None) -> dict:
    sources = await _load_conversation_texts()
    batch_no = datetime.now().strftime("mine-%Y%m%d%H%M%S")
    # 1) 分批抽取 → 写 staging(extracted)
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        for p in pairs:
            await repository.insert_staging(batch_no, batch[0][0], p.question, p.answer)
    # 2) 整体去重(本轮 extracted 内 + 对已有 knowledge)
    staged = [s for s in await repository.list_staging_by_status("extracted") if s.batch_no == batch_no]
    existing = await repository.list_all_questions()
    kept, discarded = dedup.dedupe(staged, existing)
    await repository.set_staging_status([s.id for s in kept], "kept")
    await repository.set_staging_status([s.id for s in discarded], "discarded")
    # 3) kept 写 knowledge_chunks(pending, content_type=mined, questions=真实问法)
    chunks = [Chunk(category="历史对话", questions=s.question, answer=s.answer,
                    section_path="mined", content_type="mined") for s in kept]
    if chunks:
        await dualwrite.write_pending(chunks)
    return {"sources": len(sources), "extracted": len(staged),
            "kept": len(kept), "discarded": len(discarded)}
