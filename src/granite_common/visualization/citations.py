# SPDX-License-Identifier: Apache-2.0

"""
Widget for visualizing citations.
"""

# Standard
import pathlib

# Third Party
import anywidget
import traitlets

# First Party
from granite_common.base.types import (
    ChatCompletion,
    ChatCompletionResponse,
)


class CitationsWidget:
    def show(self, inputs: ChatCompletion, outputs: ChatCompletionResponse):
        documents = [doc.model_dump(mode="json") for doc in inputs._documents()]
        response = inputs.messages[-1].content
        citations = outputs.choices[0].message.content

        return CitationsWidgetInstance(
            response=response, documents=documents, citations=citations
        )


class CitationsWidgetInstance(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "index.js"
    _css = pathlib.Path(__file__).parent / "index.css"

    response = traitlets.Unicode({}).tag(sync=True)
    documents = traitlets.List([]).tag(sync=True)
    citations = traitlets.List([]).tag(sync=True)
