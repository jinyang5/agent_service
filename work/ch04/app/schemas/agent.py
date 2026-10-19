from pydantic import BaseModel, Field, field_validator


class AgentRequest(BaseModel):
    user_id: str = Field(min_length=1, description="用户标识")
    message: str = Field(min_length=1, description="用户本轮消息")
    conversation_id: int | None = Field(default=None, description="续接会话,首轮留空")

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        # Review Focus 4:min_length=1 挡不住纯空白
        if not v.strip():
            raise ValueError("message 不能为空白")
        return v


class ToolCallView(BaseModel):
    id: str
    name: str
    args: dict


class ToolResultView(BaseModel):
    tool_call_id: str
    name: str
    ok: bool
    content: str


class AgentResponse(BaseModel):
    conversation_id: int
    answer: str
    tool_calls: list[ToolCallView]
    tool_results: list[ToolResultView]
