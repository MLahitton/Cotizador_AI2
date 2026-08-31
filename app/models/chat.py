from typing import Any, Literal

from pydantic import BaseModel, Field

ChatScope = Literal["REQUIREMENT", "ITEM"]
ChatRole = Literal["user", "assistant"]


class ChatConversationMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1, max_length=4000)


class ChatRespondRequest(BaseModel):
    scope: ChatScope
    userMessage: str = Field(min_length=1, max_length=4000)
    conversation: list[ChatConversationMessage] = Field(default_factory=list, max_length=40)
    context: dict[str, Any]


class ChatRespondResponse(BaseModel):
    message: str = Field(min_length=1)
