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

# Third Party
import pydantic

# First Party
from granite_common.base.io import ChatCompletionResultProcessor
from granite_common.base.types import (
    ChatCompletion,
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)

# Local
from . import json_util
from .constants import MESSAGE_SENTENCE_TAG
from .input import sentence_delimiter
from .util import make_config_dict


class _MappingType(enum.Enum):
    PASSTHRU = 1
    """Pass through the value from the model output using the type of the raw output
    schema"""


class TransformationRule(abc.ABC):
    """Base class for transformation rules to apply to JSON outputs of intrinsics."""

    YAML_NAME = None
    """Subclasses should set this to the name of the rule in YAML config files."""

    def __init__(self, input_path_expr: list[str | int | None]):
        """
        :param input_path_expr: Path expression that matches all instances of the field
            that this rule transforms. Elements can be strings for object fields,
            ints for list indices, or ``None`` for wildcard matches.
        """
        self.input_path_expr = input_path_expr

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

    def _matching_paths(self, parsed_json: Any) -> list[tuple]:
        """
        :param parsed_json: Output of running model results through
            :func:`json.loads()`, plus applying zero or more transformation rules.
        :returns: List of paths within ``parsed_json`` that match this rule's input
            path spec.
        """
        return [
            p for p in json_util.scalar_paths(parsed_json) if self._is_input_path(p)
        ]

    def rule_name(self) -> str:
        if self.YAML_NAME is None:
            raise ValueError(f"Attempted to fetch missing rule name for {type(self)}")
        return self.YAML_NAME

    # pylint: disable=unused-argument
    def _prepare(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
        chat_completion: ChatCompletion | None,
    ) -> dict:
        """
        Subclasses may override this method to prepare data structures that should be
        computed once per output.

        :returns: Dict that will be passed to all calls to :func:`self._transform()`
        """
        return {}

    def apply(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
        chat_completion: ChatCompletion | None,
    ) -> Any:
        """
        Main entry point.

        :param parsed_json: Output of running model results through
            :func:`json.loads()`, plus applying zero or more transformation rules.
        :param reparsed_json: Output of running the same model results through
            :func:`json_util.reparse_json_with_offsets()`.
        :param logprobs: Optional logprobs result associated with the original model
            output string, or ``None`` of no logprobs were present.
        :param chat_completion: The chat completion request that produced this output.
        :returns: Transformed copy of ``parsed_json`` after applying this rule.
        """
        paths = [
            p for p in json_util.scalar_paths(parsed_json) if self._is_input_path(p)
        ]
        prepare_output = self._prepare(
            parsed_json, reparsed_json, logprobs, chat_completion
        )

        # If we get here, we need to modify a JSON object or array
        # Don't modify input
        result = copy.deepcopy(parsed_json)
        for path in paths:
            result = self._apply_at_path(result, path, prepare_output)
        return result

    @abc.abstractmethod
    def _apply_at_path(self, result: Any, path: tuple, prepare_output: dict) -> Any:
        """
        Subclasses should modify this

        :param result: Parsed JSON representation of the transformed output at the
            current stage of transformation.  A copy of the original.
        :param path: Current path to transform locally
        :param prepare_output: Dictionary of global data that this object's
            :func:`self._prepare()` method has set aside

        :returns: A modified version of ``result``, which may be modified in place or
            a fresh copy.
        """
        raise NotImplementedError()


class InPlaceTransformation(TransformationRule):
    """
    Base class for ``TransformationRule``s that replace values in place in the source
    JSON. The values replaced can be a scalar, object, or list.
    """

    def _apply_at_path(self, result: Any, path: tuple, prepare_output: dict) -> Any:
        """
        Subclasses should modify this

        :param result: Parsed JSON representation of the transformed output at the
            current stage of transformation.  A copy of the original.
        :param path: Current path to transform locally
        :param prepare_output: Dictionary of global data that this object's
            :func:`self._prepare()` method has set aside

        :returns: A modified version of ``result``, which may be modified in place or
            a fresh copy.
        """
        original_value = json_util.fetch_path(result, path)
        transformed_value = self._transform(original_value, path, prepare_output)
        result = json_util.replace_path(result, path, transformed_value)
        return result

    @abc.abstractmethod
    def _transform(self, value: Any, path: tuple, prepare_output: dict) -> Any:
        """
        Subclasses should override this method to transform a single scalar value
        from a larger JSON expression.

        :param value: Original value pulled out of the JSON expression, with position
            information attached to any embedded scalars.
        :param path: Location in the input JSON where the value was found
        :param prepare_output: Results from calling :func:`self._prepare()`

        :returns: New value for the indicated element of the JSON expression.
        """
        raise NotImplementedError()


class AddFieldsTransformation(TransformationRule):
    """
    Base class for ``TransformationRule``s that add one or more values adjacent to
    an existing value in the source JSON.
    """

    def _apply_at_path(self, result: Any, path: tuple, prepare_output: dict) -> Any:
        """
        Subclasses should modify this

        :param result: Parsed JSON representation of the transformed output at the
            current stage of transformation.  A copy of the original.
        :param path: Current path to transform locally
        :param prepare_output: Dictionary of global data that this object's
            :func:`self._prepare()` method has set aside

        :returns: A modified version of ``result``, which may be modified in place or
            a fresh copy.
        """
        if len(path) == 0:
            raise ValueError(
                "Expected path to field of JSON object, but received zero-length path."
            )

        parent_path = path[:-1]
        parent_object = json_util.fetch_path(result, parent_path)
        if not isinstance(parent_object, dict):
            raise TypeError(
                f"Expected JSON object at path {parent_path} but found value of type "
                f"{type(parent_object)}"
            )

        original_value = parent_object[path[-1]]
        new_values = self._transform(original_value, path, prepare_output)

        # Make a copy, just in case.
        new_parent = parent_object.copy() | new_values
        result = json_util.replace_path(result, parent_path, new_parent)
        return result

    @abc.abstractmethod
    def _transform(self, value: Any, path: tuple, prepare_output: dict) -> dict:
        """
        Subclasses should override this method to transform a single scalar value
        from a larger JSON expression.

        :param value: Original value pulled out of the JSON expression, with position
            information attached to any embedded scalars.
        :param path: Location in the input JSON where the value was found
        :param prepare_output: Results from calling :func:`self._prepare()`

        :returns: Mapping from name of added field to value.
        """
        raise NotImplementedError()


##################################################
# Rule implementation classes start here


class TokenToFloat(InPlaceTransformation):
    """
    Transformation rule that decodes token logprobs to a floating point number.

    The floating point number replaces the original categorical value in the JSON.
    """

    YAML_NAME = "likelihood"

    def __init__(
        self,
        input_path_expr: list[str | int | None],
        /,
        categories_to_values: dict[str | int | bool, float] | None = None,
    ):
        """
        :param categories_to_values: Mapping from categorical labels to floating-point
            values.
        :type categories_to_values: dict[str | int | bool, float]
        """
        super().__init__(input_path_expr)
        self.categories_to_values = categories_to_values

    def _prepare(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
        chat_completion: ChatCompletion | None,
    ) -> dict:
        if logprobs is not None and not isinstance(logprobs, ChatCompletionLogProbs):
            raise TypeError(
                f"Expected ChatCompletionLogProbs, but received {type(logprobs)}"
            )
        if logprobs is None:
            raise TypeError("This rule requires logprobs.  Received None for logprobs.")
        begin_to_token = json_util.make_begin_to_token_table(logprobs)

        return {
            "begin_to_token": begin_to_token,
            "reparsed_json": reparsed_json,
            "logprobs": logprobs,
        }

    def _transform(self, value: Any, path: tuple, prepare_output: dict) -> Any:
        # Retrieve values that are computed during self._prepare()
        begin_to_token = prepare_output["begin_to_token"]
        logprobs = prepare_output["logprobs"]
        reparsed_json = prepare_output["reparsed_json"]

        json_literal = json_util.fetch_path(reparsed_json, path)
        if not isinstance(json_literal, json_util.JsonLiteralWithPosition):
            raise TypeError(
                f"Expected literal with position, but received '{value}' "
                f"of type {type(value)}"
            )
        if value != json_literal.value:
            # Sanity check: Can't apply this rule on a path for which the tokens for
            # the original logprobs are no longer present.
            raise ValueError(
                f"At path {path}, reparsed value '{json_literal}' differs from "
                f"current value '{value}'. This rule cannot decode logprobs under this "
                f"circumstance."
            )

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
        values = []
        weights = []

        # Decode top token.
        # Assume that probability of first token == probability of entire literal
        if json_literal.value in self.categories_to_values:
            values.append(self.categories_to_values[json_literal.value])
            weights.append(math.exp(logprobs.content[first_token_ix].logprob))

        # Decode remaining tokens.
        # Here we assume that the first category that shares a prefix with the token is
        # what the completion would have been had that token been the top-1.
        top_logprob: ChatCompletionLogProb
        for top_logprob in logprobs.content[first_token_ix].top_logprobs:
            if top_logprob.token == logprobs.content[first_token_ix].token:
                # Some inference engines will output the top-1 token both in logprobs
                # and in top_logprobs some of the time. Don't double-count when that
                # happens.
                continue
            for category, value_for_category in self.categories_to_values.items():
                if str(category).startswith(top_logprob.token):
                    # Use the first prefix match
                    values.append(value_for_category)
                    weights.append(math.exp(top_logprob.logprob))
                    break

        # Make the weights sum to 1 and return weighted sum, aka expected value
        if len(values) == 0:
            # No match --> default to 0
            return 0.0
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        return sum(w * v for w, v in zip(weights, values, strict=True))


class DecodeSentences(AddFieldsTransformation):
    """
    Transformation rule that decodes references to sentences by number into begin, end,
    text tuples.
    """

    YAML_NAME = "decode_sentences"

    def __init__(
        self,
        input_path_expr: list[str | int | None],
        /,
        source: str,
        output_names: dict,
    ):
        """
        :param source: Name of the location to look for sentences; can be "last_turn",
            "second_to_last_turn", or "documents".
        :param output_names: Names of new result fields to add
        """
        super().__init__(input_path_expr)

        allowed_sources = ("last_turn", "second_to_last_turn", "documents")
        if source not in allowed_sources:
            raise ValueError(
                f"'source' argument must be one of {allowed_sources}. "
                f"Received '{source}'"
            )
        self.source = source

        if not isinstance(output_names, dict):
            raise TypeError(
                f"Expected mapping for output_names, but received {output_names}"
            )
        for k in output_names:
            if k not in ("begin", "end", "text"):
                raise ValueError(f"Unexpected key '{k}' in output_names")
        self.begin_name = output_names.get("begin")
        self.end_name = output_names.get("end")
        self.text_name = output_names.get("text")

    def _prepare(
        self,
        parsed_json: Any,
        reparsed_json: Any,
        logprobs: ChatCompletionLogProbs | None,
        chat_completion: ChatCompletion | None,
    ) -> dict:
        if self.source == "documents":
            # TODO: Refactor common code under "last_turn" branch
            # TODO: Decode sentences within documents, keeping in mind that sentence
            #  numbers do not go back to zero at each document beginning
            # TODO: Add "documents" output list
            raise NotImplementedError(
                "Decoding document sentence boundaries not currently implemented."
            )
        elif self.source in ("last_turn", "second_to_last_turn"):
            tag = MESSAGE_SENTENCE_TAG
            message_ix = -1 if self.source == "last_turn" else -2
            target_text = chat_completion.messages[message_ix].content
            sentence_num = 0
            # BEGIN code to refactor --------------------------------------------------

            # Build up offsets into the ORIGINAL text, not text with sentence tags
            begins = []
            ends = []
            texts = []

            # There is always at least one sentence if the text is encoded correctly.
            delimiter = sentence_delimiter(tag, sentence_num)
            delimiter_loc = target_text.find(delimiter, 0)
            if delimiter_loc == -1:
                raise ValueError(
                    f"First sentence delimiter '{delimiter}' not found in "
                    f"target string '{target_text}'"
                )
            begin = 0
            tagged_begin = delimiter_loc + len(delimiter)

            while True:
                # Loop invariant: We are looking for the end of sentence <sentence_num>.
                #   <begin> is positioned at the beginning of <sentence_num>,
                #   immediately after the delimiter for that sentence.

                # Delimiter string occurs at the BEGINNING of every sentence.
                # Check for delimiter of next sentence.
                delimiter = sentence_delimiter(tag, sentence_num + 1)
                delimiter_loc = target_text.find(delimiter, tagged_begin)
                if delimiter_loc == -1:
                    # No more sentence markers
                    begins.append(begin)
                    # Begin + (remaining characters in tagged text)
                    ends.append(begin + (len(target_text) - tagged_begin))
                    texts.append(target_text[tagged_begin:])
                    break
                begins.append(begin)
                begin += delimiter_loc
                ends.append(begin)
                new_tagged_begin = delimiter_loc + len(delimiter)
                texts.append(target_text[tagged_begin:new_tagged_begin])
                tagged_begin = new_tagged_begin
                sentence_num += 1
            # END code to refactor ----------------------------------------------------
        else:
            raise ValueError(f"Unexpected source string '{self.source}'")

        return {"begins": begins, "ends": ends, "texts": texts}

    def _transform(self, value: Any, path: tuple, prepare_output: dict) -> dict:
        # Unpack global values we set aside during the prepare phase
        begins = prepare_output["begins"]
        ends = prepare_output["ends"]
        texts = prepare_output["texts"]

        if not isinstance(value, int):
            raise TypeError(
                f"Expected integer sentence number at path {path}, but "
                f"found non-integer value {value} of type {type(value)}"
            )
        sentence_num = value

        result = {}
        if self.begin_name is not None:
            result[self.begin_name] = begins[sentence_num]
        if self.end_name is not None:
            result[self.end_name] = ends[sentence_num]
        if self.text_name is not None:
            result[self.text_name] = texts[sentence_num]
        return result


ALL_RULES = [TokenToFloat, DecodeSentences]
NAME_TO_RULE = {cls.YAML_NAME: cls for cls in ALL_RULES}

# END of rule implementations
############################################


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
                rule_kwargs = {
                    k: v
                    for k, v in transform_spec.items()
                    if k not in ("type", "input_path")
                }
                self.rules.append(rule_cls(input_path, **rule_kwargs))

    # pylint: disable=unused-argument
    def _transform_impl(
        self,
        chat_completion_response: ChatCompletionResponse | dict | pydantic.BaseModel,
        chat_completion: ChatCompletion | None = None,
    ) -> ChatCompletionResponse:
        transformed_choices = [
            self._transform_choice(c, chat_completion)
            for c in chat_completion_response.choices
        ]
        return chat_completion_response.model_copy(
            update={"choices": transformed_choices}
        )

    def _transform_choice(
        self,
        choice: ChatCompletionResponseChoice,
        chat_completion: ChatCompletion | None,
    ) -> ChatCompletionResponseChoice:
        # Parse JSON output twice: Once to verify valid JSON and once to compute offsets
        # Note that we don't currently check schema, as that would require an additional
        # library dependency.
        parsed_json = json.loads(choice.message.content)
        reparsed_json = json_util.reparse_json_with_offsets(choice.message.content)
        for rule in self.rules:
            parsed_json = rule.apply(
                parsed_json, reparsed_json, choice.logprobs, chat_completion
            )
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
