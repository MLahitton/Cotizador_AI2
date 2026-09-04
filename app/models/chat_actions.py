from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.chat import ChatConversationMessage, ChatScope

ChatActionType = Literal[
    "CHANGE_SYSTEM",
    "CHANGE_GLASS",
    "CHANGE_FINISH",
    "CHANGE_QUANTITY",
    "CHANGE_DIMENSIONS",
    "EXCLUDE_ITEM",
    "INCLUDE_ITEM",
    "CHANGE_COMMERCIAL_LINE",
    "UNKNOWN",
]


class ChatActionInterpretRequest(BaseModel):
    scope: ChatScope
    userMessage: str = Field(min_length=1, max_length=4000)
    conversation: list[ChatConversationMessage] = Field(default_factory=list, max_length=40)
    context: dict[str, Any] = Field(default_factory=dict)


class ChatActionIntent(BaseModel):
    isAction: bool
    actionType: ChatActionType
    scope: ChatScope
    targetReference: str | None = None
    requestedValue: str | None = None
    requestedQuantity: int | None = Field(default=None, gt=0)
    requestedWidthMm: int | None = Field(default=None, gt=0)
    requestedHeightMm: int | None = Field(default=None, gt=0)
    confidence: float = Field(ge=0.0, le=1.0)
    requiresClarification: bool
    clarificationReason: str | None = None
    rawUserMessage: str
