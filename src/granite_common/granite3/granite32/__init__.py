# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Input and output processing for the Granite 3.2 family of models.
"""

# Local
from .input import Granite3Point2InputProcessor
from .output import Granite3Point2OutputProcessor
from .types import ControlsRecord, Granite3Point2ChatCompletion

__all__ = (
    "ControlsRecord",
    "Granite3Point2ChatCompletion",
    "Granite3Point2InputProcessor",
    "Granite3Point2OutputProcessor",
)
