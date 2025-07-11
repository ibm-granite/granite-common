# SPDX-License-Identifier: Apache-2.0

__doc__ = f"""
{__package__} is a Python library that provides enhanced prompt creation and output
parsing for IBM Granite models
"""

# Local
# This file explicitly imports all the symbols that we export at the top level of this
# package's namespace.
from .base.types import (
    AssistantMessage,
    ChatCompletion,
    UserMessage,
)
from .granite3.granite32 import (
    Granite3Point2ChatCompletion,
    Granite3Point2InputProcessor,
    Granite3Point2OutputProcessor,
)

__all__ = (
    obj.__name__
    for obj in (
        AssistantMessage,
        ChatCompletion,
        UserMessage,
        Granite3Point2InputProcessor,
        Granite3Point2OutputProcessor,
        Granite3Point2ChatCompletion,
    )
)
