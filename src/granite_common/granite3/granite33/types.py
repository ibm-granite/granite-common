# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Dataclasses that are specific to the Granite 3.3 family of models.
"""

# Third Party
import pydantic

# First Party
from granite_common.base.types import Document
from granite_common.granite3.types import Granite3ChatCompletion


class Granite33ChatCompletion(Granite3ChatCompletion):
    """
    Class that represents the inputs to a Granite 3.3 model generation call.
    """

    @pydantic.field_validator("documents")
    @classmethod
    def _validate_documents(cls, documents: list[Document] | None) -> list | None:
        """
        Granite 3.3 documents must have document IDs.
        """
        if documents is not None:
            for i, d in enumerate(documents):
                if not isinstance(d, Document):
                    raise TypeError(
                        f"Expected Document at position {i} but found "
                        f"{d} of type {type(d)}"
                    )
                if d.doc_id is None:
                    raise ValueError(
                        f"Document at position {i} lacks a `doc_id` "
                        f"field. This field is required for Granite "
                        f"3.3."
                    )
        return documents
