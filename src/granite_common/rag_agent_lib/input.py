# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Classes and functions that implement common aspects of input processing for all
LoRA adapters in IBM's `rag-agent-lib` library of intrinsics.
"""


# Standard
import json
import pathlib

# First Party
from granite_common import UserMessage
from granite_common.base.io import ChatCompletion, ChatCompletionRewriter

# Local
from .util import make_config_dict


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

        # Response format is JSON schema
        self.response_format = json.loads(self.config["response_format"])

        if self.config["parameters"] is not None and not isinstance(
            self.config["parameters"], dict
        ):
            raise TypeError(
                f"'parameters' field must be null (~ in YAML) or contain a mapping "
                f"from chat completion parameter name to value. Current value "
                f"{self.config['parameters']}"
            )
        self.parameters = self.config["parameters"]

        if model_name is not None:
            self.parameters["model"] = model_name
        elif self.config["model"]:
            self.parameters["model"] = self.config["model"]

        self.instruction = self.config["instruction"]

    def __call__(self, chat_completion: ChatCompletion, /, **kwargs) -> ChatCompletion:
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

        return chat_completion.model_copy(update=edits)
