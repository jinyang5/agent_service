import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.core import agent
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    try:
        result = await agent.run_agent_turn(req.user_id, req.message, req.conversation_id)
    except agent.ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在")
    except SQLAlchemyError:
        logger.exception("数据库错误 user_id=%s", req.user_id)
        raise HTTPException(status_code=503, detail="数据库暂时不可用,请稍后重试")
    except Exception:
        logger.exception("agent 编排失败 user_id=%s", req.user_id)
        raise HTTPException(status_code=502, detail="上游模型暂时不可用,请稍后重试")

    return AgentResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        tool_calls=[ToolCallView(id=tc["id"], name=tc["name"], args=tc["args"])
                    for tc in result.tool_calls],
        tool_results=[ToolResultView(tool_call_id=r.tool_call_id, name=r.name, ok=r.ok,
                                     content=r.tool_message.content)
                      for r in result.tool_runs],
    )
