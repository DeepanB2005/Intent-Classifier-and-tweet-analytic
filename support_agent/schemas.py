
from typing import Literal
from pydantic import BaseModel, Field


class SupportAgentOutput(BaseModel):
    intent: str = Field(
        description="Intent predicted by the dedicated intent classifier."
    )

    action: str = Field(
        description="Recommended support action."
    )

    response: str = Field(
        description="Customer-facing support response."
    )

    decision: Literal["AUTO_HANDLE", "ESCALATE"] = Field(
        description="Whether the issue should be handled automatically or escalated."
    )

    reason: str = Field(
        description="Reason for the handling decision."
    )
