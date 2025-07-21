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
    Granite32ChatCompletion,
    Granite32InputProcessor,
    Granite32OutputProcessor,
)
from .granite3.granite33 import (
    Granite33ChatCompletion,
    Granite33InputProcessor,
    Granite33OutputProcessor,
)

# The contents of __all__ must be strings
__all__ = (
    obj.__name__
    for obj in (
        AssistantMessage,
        ChatCompletion,
        UserMessage,
        Granite32InputProcessor,
        Granite32OutputProcessor,
        Granite32ChatCompletion,
        Granite33ChatCompletion,
        Granite33InputProcessor,
        Granite33OutputProcessor,
    )
)
