import json
import logging
from collections.abc import AsyncIterator, Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from sqlalchemy.exc import SQLAlchemyError

from app.core import agent
from app.core.llm import get_chat_model
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def get_model() -> BaseChatModel:
    return get_chat_model(streaming=True)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> Iterator[str]:
    yield "event: error\n"
    yield _sse({"message": message})


@router.post("/api/chat")
async def chat(req: ChatRequest, model: BaseChatModel = Depends(get_model)):
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in agent.stream_agent_turn(
                req.user_id, req.message, req.conversation_id, model=model
            ):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except agent.ConversationNotFound:
            for f in _sse_error("会话不存在"):
                yield f
            return
        except SQLAlchemyError:
            logger.exception("数据库错误 user_id=%s", req.user_id)
            for f in _sse_error("数据库暂时不可用,请稍后重试"):
                yield f
            return
        except Exception:
            logger.exception("聊天工具编排失败 user_id=%s", req.user_id)
            for f in _sse_error("上游模型暂时不可用,请稍后重试"):
                yield f
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
