# SPDX-License-Identifier: Apache-2.0

__doc__ = """
JSON parsing code used by this package.
"""

# Standard
from typing import Any
import abc
import bisect
import copy
import json
import re

# Third Party
import pydantic

##########################
# CONSTANTS

# Regexes for non-string JSON tokens that contain literal values.
# You shouldn't use regexes to parse JSON unless you know what you're doing.
# Fortunately we know what we're doing.
DELIM_REGEX = re.compile(r"[\{\}\[\]\,\:]")
NUMBER_REGEX = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
BOOL_REGEX = re.compile(r"true|false")
NULL_REGEX = re.compile(r"null")


##########################
# CLASSES


class JsonLiteral(pydantic.BaseModel):
    value: str | bool | int | float
    begin: int
    end: int


##########################
# FUNCTIONS


def find_string_offsets(json_data: str) -> list[tuple[int, int, str]]:
    """
    Find the offsets of all strings in the input, assuming that this input
    contains valid JSON.

    :param json_data: String containing valid JSON.

    :returns: Begin and end offsets of all strings in ``json_data``, including
     the double quotes.
    """
    result = []
    position = 0
    while position < len(json_data):
        if json_data[position] == '"':
            begin = position
            decoded_str, end = json.decoder.scanstring(json_data, position + 1)
            result.append((begin, end, decoded_str))
            position = end
        position += 1
    return result


def non_string_offsets(json_str, compiled_regex, string_begins, string_ends):
    """
    Identify all matches of a regex that are not within string literals

    :param json_str: Original string of valid JSON data
    :param compiled_regex: Compiled regex for the target token type
    :param string_begins: table of string begin offsets within json_str
    :param string_ends: table of string end offsets within json_str
    :return: List of (begin, end, matched string) tuples
    :rtype: list
    """
    offsets = []
    for match in compiled_regex.finditer(json_str):
        begin, end = match.span()
        delim_str = match.group()
        str_index = bisect.bisect(string_begins, match.span()[0]) - 1
        is_in_string = (
            str_index > 0
            and begin >= string_begins[str_index]
            and end < string_ends[str_index]
        )
        if not is_in_string:
            offsets.append((begin, end, delim_str))
    return offsets


def tokenize_json(json_str: str):
    """
    Lexer for parsing JSON.

    :param json_str: String representation of valid JSON data.
    :type json_str: str
    :returns: List of tuples of (begin, end, value, type)
    """
    string_offsets = find_string_offsets(json_str)
    string_begins = [s[0] for s in string_offsets]
    string_ends = [s[1] for s in string_offsets]

    delim_offsets = non_string_offsets(
        json_str, DELIM_REGEX, string_begins, string_ends
    )
    number_offsets = non_string_offsets(
        json_str, NUMBER_REGEX, string_begins, string_ends
    )
    bool_offsets = non_string_offsets(json_str, BOOL_REGEX, string_begins, string_ends)
    null_offsets = non_string_offsets(json_str, NULL_REGEX, string_begins, string_ends)

    # n-way merge
    all_offsets = sorted(
        [t + ("delim",) for t in delim_offsets]
        + [t + ("number",) for t in number_offsets]
        + [t + ("bool",) for t in bool_offsets]
        + [t + ("null",) for t in null_offsets]
        + [t + ("string",) for t in string_offsets]
    )
    return all_offsets


def reparse_value(tokens, offset) -> tuple[Any, int]:
    """
    Main entry point for a recursive-descent JSON parser with offset generation.
    Assumes valid JSON.

    :param tokens: Token stream as produced by :func:`tokenize_json()`
    :param offset: Token offset at which to start parsing
    :return: Value parsed at the indicated offset in the token stream and next offset
    """
    begin, end, value, type_ = tokens[offset]
    if type_ == "delim":
        if value == "{":
            return reparse_object(tokens, offset + 1)
        if value == "[":
            return reparse_list(tokens, offset + 1)
        raise ValueError(f"Unexpected token '{value}' found at {begin}")
    if type_ in ("string", "number", "bool", "null"):
        return JsonLiteral(value=value, begin=begin, end=end), offset + 1
    raise ValueError(f"Unexpected type string {type_}")


def reparse_object(tokens, offset) -> tuple[dict, int]:
    """
    Subroutine for handling object parsing inside reparse_value()
    """
    result = {}
    while True:
        begin, _, value, type_ = tokens[offset]
        if type_ == "delim" and value == "}":
            # Closing curly brace
            return result, offset + 1

        # Attempt to consume a key: value pair
        # Key part
        if type_ != "string":
            raise ValueError(f"Expected string at {begin} but found '{value}'")
        next_key = value
        offset += 1
        begin, _, value, type_ = tokens[offset]

        # Colon part
        if type_ != "delim" or value != ":":
            raise ValueError(f"Expected ':' at {begin} but found '{value}'")
        offset += 1

        # Value part
        next_value, offset = reparse_value(tokens, offset)
        result[next_key] = next_value
        begin, _, value, type_ = tokens[offset]

        # Comma or closing curly brace
        if type_ != "delim":
            raise ValueError(f"Expected delimiter at {begin} but found '{value}'")
        if value == ",":
            offset += 1
        elif value != "}":
            raise ValueError(f"Expected comma or '}}' at {begin} but found '{value}'")


def reparse_list(tokens, offset) -> tuple[list, int]:
    """
    Subroutine for handling list parsing inside reparse_value()
    """
    result = []
    while True:
        begin, _, value, type_ = tokens[offset]
        if type_ == "delim" and value == "]":
            # Closing square bracket
            return result, offset + 1

        # Attempt to consume a list element
        next_value, offset = reparse_value(tokens, offset)
        result.append(next_value)
        begin, _, value, type_ = tokens[offset]

        # Optional comma
        if type_ != "delim":
            raise ValueError(f"Expected delimiter at {begin} but found '{value}'")
        if value == ",":
            offset += 1


def reparse_json_with_offsets(json_str: str) -> Any:
    """
    Reparse a JSON string to compute the offsets of all literals

    :param json_str: String known to contain valid JSON data
    :type json_str: str
    :return: Parsed representation of ``json_str``, with literals at the leaf nodes of
      the parse tree replaced with instances of :class:`JsonLiteral` containing position
      information.
    :rtype: Any
    """
    tokens = tokenize_json(json_str)
    return reparse_value(tokens, 0)[0]


def scalar_paths(parsed_json) -> list[tuple]:
    """
    :param parsed_json: JSON data parsed into native Python objects

    :returns: A list of paths to scalar values within ``parsed_json``, where each
        path is expressed as a tuple. The root element of a bare scalar is an empty
        tuple.
    """
    result = []
    if isinstance(parsed_json, dict):
        for key, value in parsed_json.items():
            result.extend([(key,) + t for t in scalar_paths(value)])
    elif isinstance(parsed_json, list):
        for i, value in enumerate(parsed_json):
            result.extend([(i,) + t for t in scalar_paths(value)])
    else:
        # Base case
        result.append(tuple())
    return result


def fetch_path(json_value: Any, path: tuple):
    """
    :param json_value: Parsed JSON value
    :param path: A tuple of names/numbers that indicates a path from root to a leaf
        or internal node of ``json_value``

    :returns: The node at the indicated leaf node
    """
    cur_json_value = json_value
    ix = 0
    while ix < len(path):
        cur_elem = path[ix]
        if not isinstance(cur_elem, str | int):
            raise TypeError(
                f"Found {type(cur_elem)} at element {ix} of path {path} "
                f"Expected string or integer"
            )
        if not isinstance(cur_json_value, dict | list):
            raise TypeError(
                f"Found {type(cur_json_value)} at path {path[:ix]} "
                f"of {json_value}. Was expecting dict or list."
            )
        cur_json_value = cur_json_value[cur_elem]
        ix += 1
    return cur_json_value


def replace_path(json_value: Any, path: tuple, new_value: Any):
    """
    Modify a parsed JSON value in place by setting a particular path.

    :param json_value: Parsed JSON value
    :param path: A tuple of names/numbers that indicates a path from root to leaf of
        ``json_value``
    :param new_value: New value to put in place at the indicated location
    """
    where_to_write = fetch_path(json_value, path[:-1])
    where_to_write[path[-1]] = new_value  # Modify in place


class TransformationRule(abc.ABC):
    """Base class for transformation rules to apply to JSON outputs of intrinsics."""

    def __init__(self, input_name: str):
        """
        :param input_name: Name of the field that this rule transforms, or "<root>" to
         indicate that the transformation should apply at the root of a scalar value.
        """
        self.input_name = input_name

    # def target_paths(self, parsed_json: Any):
    #     """
    #     Method for subclasses to use to find where to apply themselves.

    #     :param parsed_json: Output of running model results through
    #         :func:`json.loads()`, plus applying zero or more transformation rules.
    #     :returns: Paths to

    def apply(self, parsed_json: Any, reparsed_json: Any) -> Any:
        """
        :param parsed_json: Output of running model results through
            :func:`json.loads()`, plus applying zero or more transformation rules.
        :param reparsed_json: Output of running the same model results through
            :func:`json_util.reparse_json_with_offsets()`.
        :returns: Transformed copy of ``parsed_json`` after applying this rule.
        """
        paths = scalar_paths(parsed_json)
        paths_reparsed = scalar_paths(reparsed_json)
        if paths != paths_reparsed:
            # Sanity check; reparsing shouldn't change structure
            raise ValueError(
                f"Mismatched paths between parsed and reparsed JSON "
                f"output of model. "
                f"{paths=}; {paths_reparsed=}"
            )

        if self.input_name == "<root>":
            # Special case: Transformation rule expects a scalar, so there's no dict or
            # list to modify in place.
            if len(paths) != 1 and len(paths[0]) != 0:
                raise ValueError(
                    f"This rule applies to a scalar JSON value but "
                    f"received a parsed value of type {type(parsed_json)}"
                )
            if isinstance(parsed_json, dict | list):
                # Belt and suspenders error checking
                raise TypeError(
                    f"Attempted to transform non-scalar JSON value {parsed_json} "
                    f"with a transformation rule that applies only to scalar values."
                )
            return self._transform(parsed_json)

        # Filter down to name specified in YAML file
        paths = [p for p in paths if p[-1] == self.input_name]

        # Don't modify input
        result = copy.deepcopy(parsed_json)

        for path in paths:
            original_value = fetch_path(result, path)
            transformed_value = self._transform(original_value)
            replace_path(result, path, transformed_value)
        return result

    @abc.abstractmethod
    def _transform(self, value: int | float | str | None) -> int | float | str | None:
        """
        Subclasses should override this method to transform a single scalar value
        from a larger JSON expression.

        :param value: Original value pulled out of the JSON expression.
        """

    @abc.abstractmethod
    def _transform(self, value: int | float | str | None) -> int | float | str | None:
        """
        Subclasses should override this method to transform a single scalar value
        from a larger JSON expression.

        :param value: Original value pulled out of the JSON expression.
        """
