# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Classes and functions that implement common aspects of output processing for all
LoRA adapters in IBM's `rag-agent-lib` library of intrinsics.
"""

# Standard
import enum
import json
import pathlib

# First Party
from granite_common.base.io import ChatCompletionRewriter

# Local
from .util import make_config_dict


class _MappingType(enum.Enum):
    PASSTHRU = 1
    """Pass through the value from the model output using the type of the raw output
    schema"""


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
        self.config = make_config_dict(config_file, config_dict)

        # Response format is JSON schema
        self.response_format = json.loads(self.config["response_format"])

        # Output schema can be None, indicating same as response format; or another
        # schema, indicating automated data transformation based on field names
        if self.config["output_schema"] is None:
            self.output_schema = None
        else:
            self.output_schema = json.loads(self.config["output_schema"])

        # TODO: Finish initialization

    def _compute_schema_mapping(self) -> dict:
        """Generates and returns the schema mapping for this object's response and
        output schemas.

        :returns: Mapping from fields of ``self.output_schema`` to corresponding fields
            of ``self.response_format``
        """
