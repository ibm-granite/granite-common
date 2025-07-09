# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Classes and functions that implement common aspects of input and output string 
processing for all Granite models.
"""

# Standard
import abc

# Third Party
import pydantic

# Local
from .types import AssistantMessage, ChatCompletion


class InputProcessor(abc.ABC):
    """
    Interface for generic input processors. An input processor exposes an
    API to transform a chat completion request into a string prompt.
    """

    @abc.abstractmethod
    def transform(
        self, chat_completion: ChatCompletion, add_generation_prompt: bool = True
    ) -> str:
        """
        Convert the structured representation of the inputs to a completion request into
        the string representation of the tokens that should be sent to the model to
        implement said request.

        :param chat_completion: Structured representation of the inputs
        :param add_generation_prompt: If ``True``, the returned prompt string will
            contain a prefix of the next assistant response for use as a prompt to a
            generation request. Otherwise, the prompt will only contain the messages and
            documents in ``input``.

        :returns: String that can be passed to the model's tokenizer to create a prompt
            for generation.
        """


class OutputProcessorParams(pydantic.BaseModel):
    """Base class for parameters that control output processing. See individual output
    processors for model-specific parameters."""


class OutputProcessor(abc.ABC):
    """
    Base class for generic output processors. An output processor exposes an
    API to transform model output into a structured representation of the
    information.

    This interface is very generic; see individual classes for more specific arguments
    """

    @abc.abstractmethod
    def transform(
        self, model_output: str, params: OutputProcessorParams | None = None
    ) -> AssistantMessage:
        """
        Convert the model output generated into a structured representation of the
        information.

        :param model_output: String output of the a generation request, potentially
            incomplete if it was a streaming request
        :param inputs: Optional parameters to control how to parse ``model_output``

        :returns: The parsed output so far, as an instance of :class:`AssistantMessage`
            possibly with model-specific extension fields.
        """
