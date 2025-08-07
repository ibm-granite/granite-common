# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Classes and functions that implement common aspects of output processing for all
LoRA adapters in IBM's `rag-agent-lib` library of intrinsics.
"""

# Standard
from typing import Any
import abc
import copy
import enum
import json
import math
import pathlib

# First Party
from granite_common.base.io import ChatCompletionRewriter
from granite_common.base.types import ChatCompletionLogProbs

# Local
from . import json_util
from .util import make_config_dict


class _MappingType(enum.Enum):
    PASSTHRU = 1
    """Pass through the value from the model output using the type of the raw output
    schema"""


class TransformationRule(abc.ABC):
    """Base class for transformation rules to apply to JSON outputs of intrinsics."""

    def __init__(
        self, input_path_expr: list[str | int | None], output_name: str | None
    ):
        """
        :param input_path_expr: Path expression that matches all instances of the field
            that this rule transforms. Elements can be strings for object fields,
            ints for list indices, or ``None`` for wildcard matches.
        :param output_name: Name to rename matching fields to, or ``None`` to perform
            no renaming.
        """
        self.input_path_expr = input_path_expr
        self.output_name = output_name

    def _is_input_path(self, path: tuple) -> bool:
        """
        :param path: JSON path as returned by :func:`scalar_paths()`

        :returns: True if this rule should be applied to the indicated path.
        """
        if len(path) != len(self.input_path_expr):
            return False
        for expr_elem, path_elem in zip(self.input_path_expr, path, strict=True):
            # None means "wildcard"
            if expr_elem is not None and path_elem != expr_elem:
                return False
        return True

    def apply(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
    ) -> Any:
        """
        :param parsed_json: Output of running model results through
            :func:`json.loads()`, plus applying zero or more transformation rules.
        :param reparsed_json: Output of running the same model results through
            :func:`json_util.reparse_json_with_offsets()`.
        :param logprobs: Optional logprobs result associated with the original model
            output string.
        :returns: Transformed copy of ``parsed_json`` after applying this rule.
        """
        paths = json_util.scalar_paths(parsed_json)
        paths_reparsed = json_util.scalar_paths(reparsed_json)
        if paths != paths_reparsed:
            # Sanity check; reparsing shouldn't change structure
            raise ValueError(
                f"Mismatched paths between parsed and reparsed JSON "
                f"output of model. "
                f"{paths=}; {paths_reparsed=}"
            )

        begin_to_token = json_util.make_begin_to_token_table(logprobs)

        # Find paths that match the path expression
        paths = [p for p in paths if self._is_input_path(p)]

        # If we get here, we need to modify a JSON object or array
        # Don't modify input
        result = copy.deepcopy(parsed_json)
        for path in paths:
            original_reparsed_value = json_util.fetch_path(reparsed_json, path)
            transformed_value = self._transform(
                original_reparsed_value, logprobs, begin_to_token
            )
            result = json_util.replace_path(result, path, transformed_value)
        return result

    @abc.abstractmethod
    def _transform(
        self,
        value: Any,
        logprobs: ChatCompletionLogProbs | None,
        begin_to_token: dict | None,
    ) -> Any:
        """
        Subclasses should override this method to transform a single scalar value
        from a larger JSON expression.

        :param value: Original value pulled out of the JSON expression, with position
            information attached to any embedded scalars.
        :param logprobs: Optional logprobs result associated with the original model
            output string.
        :param begin_to_token: If ``logprobs`` is present, a precomputed table of
            begin offset to token index for the original model output.

        :returns: New value for the indicated element of the JSON expression.
        """


class DecodeBinaryToProb(TransformationRule):
    def __init__(
        self,
        input_path_expr: list[str | int | None],
        output_name: str | None,
        /,
        positive_label: str | int | bool,
    ):
        super().__init__(input_path_expr, output_name)
        self.positive_label = positive_label

    def _transform(
        self,
        value: Any,
        logprobs: ChatCompletionLogProbs | None,
        begin_to_token: dict | None,
    ) -> Any:
        if not isinstance(value, json_util.JsonLiteralWithPosition):
            raise TypeError(
                f"Expected literal with position, but received '{value}' "
                f"of type {type(value)}"
            )
        json_literal = value
        value_str_offset = json_literal.begin
        if isinstance(json_literal.value, str):
            # Skip double quote at beginning of string literal
            value_str_offset += 1

        if value_str_offset not in begin_to_token:
            raise ValueError(
                f"Value '{json_literal}' starts at position "
                f"{value_str_offset}, "
                f"but there is no token at that position."
            )

        first_token_ix = begin_to_token[json_literal.begin]

        # Assume that probability of first token == probabity of entire literal
        prob = math.exp(logprobs.content[first_token_ix].logprob)

        # Also assume that 1 - probability is the chance of the alternative value
        if json_literal.value == self.positive_label:
            return prob
        return 1 - prob


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
