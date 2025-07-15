# SPDX-License-Identifier: Apache-2.0

"""
Type definitions that are shared within the Granite 3 family of models
"""

# Third Party
import pydantic
import pydantic_core

# First Party
from granite_common.base.types import (
    AssistantMessage,
    ChatCompletion,
    Document,
    SystemMessage,
    UserMessage,
)


class Hallucination(pydantic.BaseModel):
    """Hallucination data as returned by the model output parser"""

    hallucination_id: str
    risk: str
    reasoning: str | None = None
    response_text: str
    response_begin: int
    response_end: int


class Citation(pydantic.BaseModel):
    """Citation data as returned by the model output parser"""

    citation_id: str
    doc_id: str
    context_text: str
    context_begin: int
    context_end: int
    response_text: str
    response_begin: int
    response_end: int


class ControlsRecord(
    pydantic.BaseModel,
):
    """
    Granite 3.x controls record
    """

    citations: bool | None = None
    hallucinations: bool | None = None
    length: str | None = None  # Length output control variable
    originality: str | None = None

    @pydantic.field_validator("length", mode="after")
    @classmethod
    def _validate_length(cls, value: str | None) -> str | None:
        if value is None or value == "short" or value == "long":
            return value
        raise pydantic_core.PydanticCustomError(
            "length field validator",
            'length ({length}) must be "short" or "long" or None',
            {"length": value},
        )

    @pydantic.field_validator("originality", mode="after")
    @classmethod
    def _validate_originality(cls, value: str | None) -> str | None:
        if value is None or value == "extractive" or value == "abstractive":
            return value
        raise pydantic_core.PydanticCustomError(
            "originality field validator",
            'originality ({originality}) must be "extractive" or "abstractive" or None',
            {"originality": value},
        )


class Granite3ChatCompletion(ChatCompletion):
    """
    Class that represents the inputs that are common to models of the IBM Granite 3.x
    family.
    """

    documents: list[Document] | None = None
    controls: ControlsRecord | None = None
    thinking: bool = False

    @pydantic.field_validator("messages")
    @classmethod
    def _validate_inputs_messages(cls, messages: list) -> list:
        # Make a copy so the validation code below can mutate the messages list but pass
        # through the original value. The caller also might have a pointer to the list.
        original_messages = messages
        messages = messages.copy()

        # There is no supervised fine tuning data for the case of zero messages.
        # Models are not guaranteed to produce a valid response if there are zero
        # messages.
        if len(messages) == 0:
            raise ValueError(
                "No messages. Model behavior for this case is not defined."
            )

        # The first message, and only the first message, may be the system message.
        first_message_is_system_message = isinstance(messages[0], SystemMessage)
        if first_message_is_system_message:
            messages = messages[1:]
            # If there is a system message, there must be at least one more user or
            # assistant message.
            if len(messages) == 0:
                raise ValueError(
                    "Input contains only a system message. Model behavior for this "
                    "case is not defined."
                )

        # The first message that is not a system message must be
        # either a user or assistant message.
        if not isinstance(messages[0], UserMessage | AssistantMessage):
            if first_message_is_system_message:
                raise ValueError(
                    f"First message after system message must be a user or "
                    f"assistant message. Found type {type(messages[0])}"
                )
            raise ValueError(
                f"First message must be a system, user, or assistant "
                f"Found type {type(messages[0])}"
            )

        # Undocumented constraint: All other messages form a conversation that
        # alternates strictly between user and assistant, possibly with tool calls
        # after an assistant turn and before the next user turn.
        # TODO: Validate this invariant.

        # Pydantic will use the value that this validator returns as the value of the
        # messages field. Undo any changes that we made during validation and return
        # the original value.
        return original_messages


class Granite3AssistantMessage(AssistantMessage):
    """
    An assistant message augmented with additional fields that are specific to the
    Granite 3 family of models.
    """

    reasoning_content: str | None = None
    citations: list[Citation] | None = None
    documents: list[Document] | None = None
    hallucinations: list[Hallucination] | None = None
    stop_reason: str | None = None

    raw_content: str | None = pydantic.Field(
        default=None,
        description=(
            "Raw response content without any parsing, for debugging and "
            "re-serialization."
        ),
    )
