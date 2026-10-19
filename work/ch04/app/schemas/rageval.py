"""RAG 评估页请求体。"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

FaithCaseStatus = Literal["未解决", "已解决", "无需解决"]


class FaithCaseStatusRequest(BaseModel):
    status: FaithCaseStatus = Field(description="台账处置状态")
    resolution: str | None = Field(
        default=None, max_length=300,
        description="处置说明;已解决 / 无需解决时必填,未解决(重开)时忽略",
    )

    @field_validator("resolution")
    @classmethod
    def _strip_resolution(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v
