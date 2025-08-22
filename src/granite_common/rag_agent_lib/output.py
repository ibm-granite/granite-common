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
from granite_common.base.io import ChatCompletionResultProcessor
from granite_common.base.types import (
    ChatCompletion,
    ChatCompletionLogProbs,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)

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


class LikelihoodRule(TransformationRule):
    """
    Transformation rule that decodes one or more yes/no answers to a likelihood ranging
    from 0 to 1.
    """

    def __init__(
        self,
        input_path_expr: list[str | int | None],
        output_name: str | None,
        /,
        positive_label: str | int | bool,
    ):
        """
        :param positive_label: Value that, if present at the indicated path, indicates
            a "yes" answer.
        :type positive_label: str | int | bool
        """
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

        first_token_ix = begin_to_token[value_str_offset]

        # Assume that probability of first token == probabity of entire literal
        prob = math.exp(logprobs.content[first_token_ix].logprob)

        # Also assume that 1 - probability is the chance of the alternative value
        if json_literal.value == self.positive_label:
            return prob
        return 1 - prob


NAME_TO_RULE = {"likelihood": LikelihoodRule}


class RagAgentLibResultProcessor(ChatCompletionResultProcessor):
    """General-purpose chat completion result processor for use with the models in the
    RAG Agent Library. Reads parameters of the model's input and output formats
    from a YAML configuration file and edits the input chat completion appropriately.
    """

    config: dict
    """Parsed YAML configuration file for the target intrinsic."""

    rules: list[TransformationRule]
    """Transformation rules that this object applies, in the order they are applied."""

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

        # Set up transformation rules for the target model's JSON output
        self.rules = []
        if self.config["transformations"]:
            for transform_spec in self.config["transformations"]:
                if transform_spec["type"] not in NAME_TO_RULE:
                    raise ValueError(
                        f"Unknown transformation rule '{transform_spec['type']}'"
                    )
                rule_cls = NAME_TO_RULE[transform_spec["type"]]
                input_path = transform_spec["input_path"]
                output_name = transform_spec["output_name"]
                rule_kwargs = {
                    k: v
                    for k, v in transform_spec.items()
                    if k not in ("type", "input_path", "output_name")
                }
                self.rules.append(rule_cls(input_path, output_name, **rule_kwargs))

    # pylint: disable=unused-argument
    def transform(
        self,
        chat_completion_response: ChatCompletionResponse,
        chat_completion: ChatCompletion | None = None,
    ) -> ChatCompletionResponse:
        transformed_choices = [
            self._transform_choice(c) for c in chat_completion_response.choices
        ]
        return chat_completion_response.model_copy(
            update={"choices": transformed_choices}
        )

    def _transform_choice(
        self,
        choice: ChatCompletionResponseChoice,
    ) -> ChatCompletionResponseChoice:
        # Parse JSON output twice: Once to verify valid JSON and once to compute offsets
        # Note that we don't currently check schema, as that would require an additional
        # library dependency.
        parsed_json = json.loads(choice.message.content)
        reparsed_json = json_util.reparse_json_with_offsets(choice.message.content)
        for rule in self.rules:
            parsed_json = rule.apply(parsed_json, reparsed_json, choice.logprobs)
        updated_message = choice.message.model_copy(
            update={"content": json.dumps(parsed_json)}
        )

        result = choice.model_copy(update={"message": updated_message})

        # Drop logprobs, since they should only be used by this function, and the tokens
        # referenced will no longer match the processed JSON value anyhow.
        # We may need to make this dropping configurable in the future.
        # Ok to modify in place because updated_message is a deep copy.
        if result.logprobs is not None:
            # Don't set the logprobs to None, unset it. There is a distinction in
            # Pydantic between these two states.
            del result.logprobs

        return result
