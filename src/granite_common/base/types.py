# SPDX-License-Identifier: Apache-2.0

"""
Common shared types
"""

# Standard
from typing import Literal, Self, TypeAlias

# Third Party
from pydantic import Field
from typing_extensions import Any
import pydantic


class NoDefaultsMixin:
    """
    Mixin so that we don't need to copy and paste the code to avoid filling JSON values
    with a full catalog of the default values of rarely-used fields.
    """

    @pydantic.model_serializer(mode="wrap")
    def _workaround_for_design_flaw_in_pydantic(self, nxt):
        """
        Workaround for a design flaw in Pydantic that forces users to accept
        unnecessary garbage in their serialized JSON data or to override
        poorly-documented serialization hooks repeatedly.  Automates overriding said
        poorly-documented serialization hooks for a single dataclass.

        See https://github.com/pydantic/pydantic/issues/4554 for the relevant dismissive
        comment from the devs. This comment suggests overriding :func:`dict()`, but that
        method was disabled a year later. Now you need to add a custom serializer method
        with a ``@model_serializer`` decorator.

        See the docs at
        https://docs.pydantic.dev/latest/api/functional_serializers/
        for some dubious information on how this API works.
        See comments below for important gotchas that aren't in the documentation.
        """
        # Start with the value that self.model_dump() would return without this mixin.
        # Otherwise serialization of sub-records will be inconsistent.
        serialized_value = nxt(self)

        # Figure out which fields are set. Pydantic does not make this easy.
        # Start with fields that are set in __init__() or in the JSON parser.
        fields_to_retain_set = self.model_fields_set

        # Add in fields that were set during validation and extra fields added by
        # setattr().  These fields all go to self.model.extra
        if self.model_extra is not None:  # model_extra is sometimes None. Not sure why.
            # model_extra is a dictionary. There is no self.model_extra_fields_set.
            fields_to_retain_set |= set(list(self.model_extra))

        # Use a subclass hook for the additional fields that fall through the cracks.
        fields_to_retain_set |= set(self._keep_these_fields())

        # Avoid changing Pydantic's field order or downstream code that computes a
        # diff over JSON strings will break.
        fields_to_retain = [k for k in serialized_value if k in fields_to_retain_set]

        # Fields that weren't in the original serialized value should be in a consistent
        # order to ensure consistent serialized output.
        # Use alphabetical order for now and hope for the best.
        fields_to_retain.extend(sorted(fields_to_retain_set - self.model_fields_set))

        result = {}
        for f in fields_to_retain:
            if f in serialized_value:
                result[f] = serialized_value[f]
            else:
                # Sometimes Pydantic adds fields to self.model_fields_set without adding
                # them to the output of self.model_dump()
                result[f] = getattr(self, f)
        for f in self._reserialize_these_fields():
            # Sometimes Pydantic's serializer fails to serialize sub-objects correctly
            field_value = getattr(self, f)
            if field_value is not None:
                result[f] = field_value.model_dump()
        return result

    def _keep_these_fields(self) -> tuple[str]:
        """
        Dataclasses that include this mixin can override this method to add specific
        default values to serialized JSON.

        This is necessary for round-tripping to JSON when there are fields that
        determine which dataclass to use for deserialization.
        """
        return ()

    def _reserialize_these_fields(self) -> tuple[str]:
        """
        Dataclasses that include this mixin can override this method to trigger
        replacing the serialized values of fields with the results of calling
        :func:`model_dump()` on this fields.

        This is necessary because Pydantic's serializer sometimes produces incorrect
        outputs for child objects for reasons unknown when called on the parent object.
        """
        return ()


class _ChatMessageBase(pydantic.BaseModel, NoDefaultsMixin):
    """Base class for all message types.

    Due to the vagaries of Pydantic's JSON parser, we use this class only for common
    functionality, and NOT for defining a common dataclass base type. Use the
    :class:`ChatMessage` type alias to annotate a field or argument as accepting all
    subclasses of this one."""

    content: str
    """Every message has raw string content, even if it also contains parsed structured
    content such as a JSON record."""

    def _keep_these_fields(self):
        return ("role",)


class UserMessage(_ChatMessageBase):
    """User message for an IBM Granite model chat completion request."""

    role: Literal["user"] = "user"


class ToolCall(pydantic.BaseModel, NoDefaultsMixin):
    """Format of an entry in the ``tool_calls`` list of an assistant message"""

    id: str | None = None
    name: str

    # This field should adhere to the argument schema from the  associated
    # FunctionDefinition in the generation request that produced it.
    arguments: dict[str, Any] | None


class AssistantMessage(_ChatMessageBase):
    """
    Lowest-common-denominator assistant message for an IBM Granite model chat
    completion request.
    """

    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall] | None = None


class ToolResultMessage(_ChatMessageBase):
    """
    Message containing the result of a tool call in an IBM Granite model chat completion
    request.
    """

    role: Literal["tool"] = "tool"
    tool_call_id: str


class SystemMessage(_ChatMessageBase):
    """System message for an IBM Granite model chat completion request."""

    role: Literal["system"] = "system"


ChatMessage: TypeAlias = (
    UserMessage | AssistantMessage | ToolResultMessage | SystemMessage
)
"""Type alias for all message types. We use this Union instead of the actual base class
:class:`_ChatMessageBase` so that Pydantic can parse the message list from JSON."""


class ToolDefinition(pydantic.BaseModel, NoDefaultsMixin):
    """
    An entry in the ``tools`` list in an IBM Granite model chat completion request.
    """

    name: str
    description: str | None = None

    # This field holds a JSON schema for a record, but the `jsonschema` package doesn't
    # define an object type for such a schema, instead using a dictionary.
    parameters: dict[str, Any] | None = None


class Document(pydantic.BaseModel, NoDefaultsMixin):
    """RAG documents, which in practice are usually snippets drawn from larger
    documents."""

    text: str
    title: str | None = None
    doc_id: str | int | None = None


class ChatTemplateKwargs(pydantic.BaseModel):
    """
    Values that can appear in the ``chat_template_kwargs`` portion of a valid chat
    completion request for a Granite model.
    """

    model_config = pydantic.ConfigDict(
        # Pass through arbitrary additional keyword arguments for handling by
        # model-specific I/O processors.
        arbitrary_types_allowed=True,
        extra="allow",
    )


class ChatCompletion(pydantic.BaseModel, NoDefaultsMixin):
    """
    Subset of the schema of a chat completion request in vLLM's OpenAI-compatible
    inference API that is exercised by Granite models.
    """

    messages: list[ChatMessage]
    model: str | None = None
    tools: list[ToolDefinition] | None = None
    documents: list[Document] | None = Field(
        default=None,
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            '"title" and "text" keys.'
        ),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    chat_template_kwargs: ChatTemplateKwargs | None = pydantic.Field(
        default=None,
        description=(
            "Additional kwargs to pass to the template renderer. "
            "Will be accessible by the chat template. "
            "Restricted to fields that at least one Granite model "
            "supports."
        ),
    )

    model_config = pydantic.ConfigDict(
        # If an input to this library is an actual vLLM chat completion request, then
        # the request will likely contain additional fields. Ignore these fields for
        # the purposes of `granite-common`.
        extra="ignore",
    )


class GraniteChatCompletion(ChatCompletion):
    """
    Lowest-common-denominator inputs to a chat completion request for an IBM Granite
    model.
    """

    @pydantic.model_validator(mode="after")
    def _validate_documents_at_top_level(self) -> Self:
        """Documents for a Granite model chat completion request should be passed in the
        ``documents`` argument at the top level of the request.

        Detect cases where the documents are hanging off of ``chat_template_kwargs``
        and sanitize appropriately.
        """
        if self.chat_template_kwargs and hasattr(
            self.chat_template_kwargs, "documents"
        ):
            if self.documents is not None:
                raise ValueError(
                    "Conflicting values of documents found in top-level "
                    "'documents' parameter and inside "
                    "'chat_template_kwargs'"
                )
            if not isinstance(self.chat_template_kwargs.documents, list):
                raise ValueError(
                    "'documents' parameter inside 'chat_template_kwargs' is not a list"
                )

            # Round-trip through dict so that documents field disappears from JSON
            # representation.
            args = self.chat_template_kwargs.model_dump()
            self.documents = [Document.model_validate(d) for d in args["documents"]]
            del args["documents"]
            self.chat_template_kwargs = ChatTemplateKwargs.model_validate(args)

        return self
