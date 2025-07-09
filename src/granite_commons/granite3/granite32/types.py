# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Dataclasses that are specific to the Granite 3.2 family of models.
"""

# Third Party
import pydantic
import pydantic_core

# First Party
from granite_commons.base.types import Document
from granite_commons.granite3.types import Granite3ChatCompletion


class ControlsRecord(
    pydantic.BaseModel,
):
    """
    Granite 3.2 controls record
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


class Granite3Point2ChatCompletion(Granite3ChatCompletion):
    """
    Class that represents the inputs to a Granite 3.2 model generation call.
    """

    controls: ControlsRecord | None = None
    thinking: bool = False

    @pydantic.field_validator("documents")
    @classmethod
    def _validate_documents(cls, documents: list[Document] | None) -> list | None:
        """
        Granite 3.2 documents should not have document IDs.
        """
        if documents is not None:
            for i, d in enumerate(documents):
                if not isinstance(d, Document):
                    raise TypeError(
                        f"Expected Document at position {i} but found "
                        f"{d} of type {type(d)}"
                    )
                if d.doc_id is not None:
                    raise ValueError(
                        f"Document at position {i} contains a `doc_id` "
                        f"field. This field is not allowed for Granite "
                        f"3.2."
                    )
        return documents
