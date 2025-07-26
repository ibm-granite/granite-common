# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import pathlib

# Third Party
import yaml

__doc__ = """
Classes and functions that implement common aspects of input processing for all
LoRA adapters in IBM's `rag-agent-lib` library of intrinsics.
"""

# First Party
from granite_common.base.io import ChatCompletion, ChatCompletionRewriter

YAML_REQUIRED_FIELDS = [
    "model",
    "response_format",
    "instruction",
    "parameters",
    "sentence_boundaries",
]


class RagAgentLibRewriter(ChatCompletionRewriter):
    """General-purpose chat completion rewriter for use with the models in the
    RAG Agent Library. Reads parameters of the model's input and output formats
    from a YAML configuration file and edits the input chat completion appropriately.
    """

    response_format: dict
    """JSON Schema of expected response format"""

    def __init__(
        self,
        /,
        config_file: str | pathlib.Path | None = None,
        config_dict: dict | None = None,
    ):
        """
        :param config_file: (optional) Location of YAML configuration file
        :param config_dict: (optional) Parsed contents of YAML configuration file
        """
        if (config_file is None and config_dict is None) or (
            config_file is not None and config_dict is not None
        ):
            raise ValueError("Exactly one of config_file and config_dict must be set.")

        if config_file:
            with open(config_file, encoding="utf8") as file:
                config_dict = yaml.safe_load(file)
        for field in YAML_REQUIRED_FIELDS:
            if field not in config_dict:
                raise ValueError(f"Configuration is missing required field '{field}'")
        self.config = config_dict

        # Response format is JSON schema
        self.response_format = json.loads(self.config["response_format"])

        if config_dict["parameters"] is not None and not isinstance(
            config_dict["parameters"], dict
        ):
            raise TypeError(
                f"'parameters' field must be null (~ in YAML) or contain a mapping "
                f"from chat completion parameter name to value. Current value "
                f"{config_dict['parameters']}"
            )
        self.parameters = config_dict["parameters"]

    def transform(self, chat_completion: ChatCompletion) -> ChatCompletion:
        edits = {}

        # TODO: Add instruction

        edits.update(self.parameters)

        return chat_completion.model_copy(update=edits)
