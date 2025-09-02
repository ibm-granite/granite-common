# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Classes and functions that implement common aspects of input processing for all
LoRA adapters in IBM's `rag-agent-lib` library of intrinsics.
"""


# Standard
import pathlib

# First Party
from granite_common import UserMessage
from granite_common.base.io import ChatCompletionRewriter
from granite_common.base.types import ChatCompletion, VLLMExtraBody

# Local
from .constants import TOP_LOGPROBS
from .util import make_config_dict


def _needs_logprobs(transformations: list | None) -> bool:
    """
    Subroutine to check whether input processing for a model needs to specify logprobs
    in the chat completion arguments.

    :param transformations: Contents of the field by the same name in the YAML file
    :type transformations: list
    :return: ``True`` if this intrinsic produces a field for which logprobs need to be
        enabled for downstream result decoding to succeed.
    :rtype: bool
    """
    if transformations is None:
        return False
    return any(t["type"] == "likelihood" for t in transformations)


class RagAgentLibRewriter(ChatCompletionRewriter):
    """General-purpose chat completion rewriter for use with the models in the
    RAG Agent Library. Reads parameters of the model's input and output formats
    from a YAML configuration file and edits the input chat completion appropriately.
    """

    config: dict
    """Parsed YAML configuration file for the target intrinsic."""

    response_format: dict
    """JSON Schema of expected response format"""

    parameters: dict
    """Additional parameters (key-value pairs) that this rewriter adds to all chat 
    completion requests."""

    extra_body_parameters: dict
    """Extended vLLM-specific parameters that go under the ``extra_body`` element of 
    the parameters field. These parameters need to be merged with any ``extra_body``
    content that is present in incoming requests."""

    instruction: str | None
    """Optional instruction template. If present, a new user message will be added with
    the indicated instruction."""

    def __init__(
        self,
        /,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
        model_name: str | None = None,
    ):
        """
        :param config_file: (optional) Location of YAML configuration file
        :param config_dict: (optional) Parsed contents of YAML configuration file
        :param model_name: (optional) model name to put into chat completion requests,
         or ``None`` to use default model name from the configuration
        """
        self.config = make_config_dict(config_file, config_dict)

        if self.config["parameters"] is not None and not isinstance(
            self.config["parameters"], dict
        ):
            raise TypeError(
                f"'parameters' field must be null (~ in YAML) or contain a mapping "
                f"from chat completion parameter name to value. Current value "
                f"{self.config['parameters']}"
            )

        # Split out parameters that go in extra_body
        self.parameters = self.config["parameters"]
        self.extra_body_parameters = {}
        if "extra_body" in self.parameters:
            self.extra_body_parameters.update(self.parameters["extra_body"])
            del self.parameters["extra_body"]

        # Check if we're supposed to override model name
        if model_name is not None:
            self.parameters["model"] = model_name
        elif self.config["model"]:
            self.parameters["model"] = self.config["model"]

        # Compute additional parameters we need to add to every request
        if _needs_logprobs(self.config["transformations"]):
            self.parameters["logprobs"] = True
            self.parameters["top_logprobs"] = TOP_LOGPROBS
        self.instruction = self.config["instruction"]

        self.extra_body_parameters["guided_json"] = self.config["response_format"]

    def _transform(
        self, chat_completion: ChatCompletion, /, **kwargs
    ) -> ChatCompletion:
        edits = {}
        if self.instruction is not None:
            # Generate and append new user message of instructions
            messages = chat_completion.messages.copy()  # Do not modify input!
            format_args = kwargs.copy()
            if len(messages) > 0:
                format_args["last_message"] = messages[-1].content
            instruction_str = self.instruction.format(**format_args)
            messages.append(UserMessage(content=instruction_str))
            edits["messages"] = messages
        edits.update(self.parameters)

        # TODO: Merge extra params
        extra_body = (
            chat_completion.extra_body.model_dump()
            if chat_completion.extra_body
            else {}
        )
        extra_body.update(self.extra_body_parameters)
        if len(extra_body) > 0:
            edits["extra_body"] = VLLMExtraBody.model_validate(extra_body)

        return chat_completion.model_copy(update=edits)
