from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, description="用户标识,用于建/归属会话")
    message: str = Field(min_length=1, description="用户本轮消息")
    conversation_id: int | None = Field(default=None, description="续接会话;为空则新建,由 done 帧回传")

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, v: str) -> str:
        # Review Focus 4:min_length=1 挡不住纯空白
        if not v.strip():
            raise ValueError("message 不能为空白")
        return v
