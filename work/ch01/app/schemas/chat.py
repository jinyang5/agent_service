from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 ID,同一会话多轮复用")
    message: str = Field(min_length=1, description="用户本轮消息")
