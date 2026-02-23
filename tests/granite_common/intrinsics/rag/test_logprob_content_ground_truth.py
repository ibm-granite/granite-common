# SPDX-License-Identifier: Apache-2.0

"""Tests for the logprobs_workaround in IntrinsicsResultProcessor."""

# Standard
import json
import pathlib

# First Party
from granite_common.base.types import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionResponse,
)
from granite_common.intrinsics.output import (
    IntrinsicsResultProcessor,
    _logprobs_workaround,
)


def _make_logprobs(tokens: list[tuple[str, float]]) -> ChatCompletionLogProbs:
    """Helper to build a ChatCompletionLogProbs from (token, logprob) pairs."""
    content = []
    for tok, lp in tokens:
        content.append(
            ChatCompletionLogProbsContent(
                token=tok,
                logprob=lp,
                top_logprobs=[
                    ChatCompletionLogProb(token=tok, logprob=lp),
                ],
            )
        )
    return ChatCompletionLogProbs(content=content)


# Harmony final channel: <|channel|>final<|message|>...<|end|>
_SINGLE_CHANNEL_PREFIX = [
    ("<|channel|>", 0.0),
    ("final", 0.0),
    ("<|message|>", 0.0),
]
_SINGLE_CHANNEL_SUFFIX = [
    ("<|end|>", 0.0),
]

# With whitespace between tokens (as seen from some gpt-oss outputs).
_SINGLE_CHANNEL_PREFIX_WS = [
    ("<|channel|>", 0.0),
    ("\n", 0.0),
    ("final", 0.0),
    ("\n", 0.0),
    ("<|message|>", 0.0),
    ("\n", 0.0),
]
_SINGLE_CHANNEL_SUFFIX_RETURN = [
    ("\n", 0.0),
    ("<|return|>", 0.0),
]

# Multi-channel sequence (analysis + final) as seen from gpt-oss.
_MULTI_CHANNEL_PREFIX = [
    ("<|channel|>", 0.0),
    ("analysis", 0.0),
    ("<|message|>", 0.0),
    ("some analysis text", 0.0),
    ("<|end|>", 0.0),
    ("<|start|>", 0.0),
    ("assistant", 0.0),
    ("<|channel|>", 0.0),
    ("final", 0.0),
    ("<|message|>", 0.0),
]

_PAYLOAD_TOKENS = [
    ('"', 0.0),
    ("answer", -0.001),
    ("able", 0.0),
    ('"', 0.0),
]


class TestLogprobsWorkaround:
    """Tests for the _logprobs_workaround helper."""

    def test_no_channel_returns_none(self):
        """Logprobs without channel tokens return None."""
        logprobs = _make_logprobs(_PAYLOAD_TOKENS)
        assert _logprobs_workaround(logprobs) is None

    def test_single_channel(self):
        """Final channel payload is extracted."""
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + _PAYLOAD_TOKENS + _SINGLE_CHANNEL_SUFFIX
        )
        result = _logprobs_workaround(logprobs)
        assert result is not None
        content, lp = result
        assert content == '"answerable"'
        tokens = [c.token for c in lp.content]
        assert tokens == ['"', "answer", "able", '"']

    def test_single_channel_with_whitespace(self):
        """Final channel with whitespace between control tokens."""
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX_WS + _PAYLOAD_TOKENS + _SINGLE_CHANNEL_SUFFIX_RETURN
        )
        result = _logprobs_workaround(logprobs)
        assert result is not None
        content, lp = result
        # Leading whitespace token after <|message|> is included
        assert content.strip() == '"answerable"'

    def test_single_channel_return_stop(self):
        """<|return|> is also recognized as end of final channel."""
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + _PAYLOAD_TOKENS + [("<|return|>", 0.0)]
        )
        result = _logprobs_workaround(logprobs)
        assert result is not None
        content, lp = result
        assert content == '"answerable"'

    def test_multi_channel(self):
        """Analysis channel is skipped, final channel payload extracted."""
        logprobs = _make_logprobs(
            _MULTI_CHANNEL_PREFIX + _PAYLOAD_TOKENS + _SINGLE_CHANNEL_SUFFIX
        )
        result = _logprobs_workaround(logprobs)
        assert result is not None
        content, lp = result
        assert content == '"answerable"'
        tokens = [c.token for c in lp.content]
        assert tokens == ['"', "answer", "able", '"']

    def test_json_object_payload(self):
        """JSON object payload is extracted from final channel."""
        payload = [("{", 0.0), ('"key"', 0.0), (":", 0.0), ('"val"', 0.0), ("}", 0.0)]
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + payload + _SINGLE_CHANNEL_SUFFIX
        )
        result = _logprobs_workaround(logprobs)
        assert result is not None
        content, _lp = result
        assert content == '{"key":"val"}'
        assert json.loads(content) == {"key": "val"}

    def test_json_array_payload(self):
        """JSON array payload is extracted from final channel."""
        payload = [("[", 0.0), ("1", 0.0), (",", 0.0), ("2", 0.0), ("]", 0.0)]
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + payload + _SINGLE_CHANNEL_SUFFIX
        )
        result = _logprobs_workaround(logprobs)
        assert result is not None
        content, _lp = result
        assert content == "[1,2]"
        assert json.loads(content) == [1, 2]

    def test_none_content(self):
        """Logprobs with None content returns None."""
        logprobs = ChatCompletionLogProbs(content=None)
        assert _logprobs_workaround(logprobs) is None


_TEST_DATA_DIR = pathlib.Path(__file__).parent / "testdata"


def _wrap_logprobs_with_channel(original_logprobs, prefix_tokens, suffix_tokens=None):
    """Helper to wrap original logprobs with channel token entries."""
    prefix = [
        ChatCompletionLogProbsContent(token=tok, logprob=lp, top_logprobs=[])
        for tok, lp in prefix_tokens
    ]
    parts = prefix + list(original_logprobs.content)
    if suffix_tokens:
        suffix = [
            ChatCompletionLogProbsContent(token=tok, logprob=lp, top_logprobs=[])
            for tok, lp in suffix_tokens
        ]
        parts = parts + suffix
    return ChatCompletionLogProbs(content=parts)


class TestResultProcessorWithLogprobsWorkaround:
    """End-to-end test of IntrinsicsResultProcessor with logprobs_workaround."""

    def _make_config(self):
        return {
            "model": None,
            "logprobs_workaround": True,
            "response_format": {
                "type": "string",
                "enum": ["answerable", "unanswerable"],
            },
            "transformations": [
                {
                    "type": "likelihood",
                    "categories_to_values": {
                        "answerable": 1.0,
                        "unanswerable": 0.0,
                    },
                    "input_path": [],
                },
                {
                    "type": "nest",
                    "input_path": [],
                    "field_name": "answerability_likelihood",
                },
            ],
        }

    def _load_answerable_output(self):
        model_output_file = (
            _TEST_DATA_DIR
            / "test_canned_output"
            / "model_output"
            / "answerability_answerable.json"
        )
        with open(model_output_file, encoding="utf-8") as f:
            return ChatCompletionResponse.model_validate_json(f.read())

    def test_single_channel_in_logprobs(self):
        """Single final channel wrapper in logprobs."""
        model_output = self._load_answerable_output()
        original_logprobs = model_output.choices[0].logprobs
        wrapped_logprobs = _wrap_logprobs_with_channel(
            original_logprobs, _SINGLE_CHANNEL_PREFIX, _SINGLE_CHANNEL_SUFFIX
        )
        wrapped_choice = model_output.choices[0].model_copy(
            update={"logprobs": wrapped_logprobs}
        )
        wrapped_output = model_output.model_copy(update={"choices": [wrapped_choice]})

        processor = IntrinsicsResultProcessor(config_dict=self._make_config())
        result = processor.transform(wrapped_output)
        result_json = json.loads(result.choices[0].message.content)
        assert "answerability_likelihood" in result_json
        assert isinstance(result_json["answerability_likelihood"], float)
        assert result_json["answerability_likelihood"] > 0.9

    def test_multi_channel_in_logprobs(self):
        """Analysis + final channel in logprobs, final channel extracted."""
        model_output = self._load_answerable_output()
        original_logprobs = model_output.choices[0].logprobs
        wrapped_logprobs = _wrap_logprobs_with_channel(
            original_logprobs,
            _MULTI_CHANNEL_PREFIX,
            _SINGLE_CHANNEL_SUFFIX,
        )
        wrapped_choice = model_output.choices[0].model_copy(
            update={"logprobs": wrapped_logprobs}
        )
        wrapped_output = model_output.model_copy(update={"choices": [wrapped_choice]})

        processor = IntrinsicsResultProcessor(config_dict=self._make_config())
        result = processor.transform(wrapped_output)
        result_json = json.loads(result.choices[0].message.content)
        assert "answerability_likelihood" in result_json
        assert isinstance(result_json["answerability_likelihood"], float)
        assert result_json["answerability_likelihood"] > 0.9

    def test_no_channel_falls_back_to_content(self):
        """Without channel tokens in logprobs, falls back to message.content."""
        model_output = self._load_answerable_output()
        processor = IntrinsicsResultProcessor(config_dict=self._make_config())
        result = processor.transform(model_output)
        result_json = json.loads(result.choices[0].message.content)
        assert "answerability_likelihood" in result_json
        assert isinstance(result_json["answerability_likelihood"], float)
        assert result_json["answerability_likelihood"] > 0.9

    def test_no_logprobs_falls_back_to_content(self):
        """Without logprobs, falls back to message.content."""
        raw = {
            "id": "test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": '"answerable"',
                        "role": "assistant",
                    },
                }
            ],
            "created": 0,
            "model": "test",
        }
        model_output = ChatCompletionResponse.model_validate(raw)
        config = self._make_config()
        # Remove likelihood rule since no logprobs
        config["transformations"] = [
            t for t in config["transformations"] if t["type"] != "likelihood"
        ]
        processor = IntrinsicsResultProcessor(config_dict=config)
        result = processor.transform(model_output)
        result_json = json.loads(result.choices[0].message.content)
        assert "answerability_likelihood" in result_json

    def test_channel_in_content_resolved_by_logprobs(self):
        """Channel tokens in content are resolved when logprobs provide ground truth."""
        model_output = self._load_answerable_output()
        original_logprobs = model_output.choices[0].logprobs

        wrapped_logprobs = _wrap_logprobs_with_channel(
            original_logprobs, _SINGLE_CHANNEL_PREFIX, _SINGLE_CHANNEL_SUFFIX
        )
        wrapped_choice = model_output.choices[0].model_copy(
            update={
                "logprobs": wrapped_logprobs,
                "message": model_output.choices[0].message.model_copy(
                    update={
                        "content": (
                            '<|channel|>final<|message|>"answerable"<|end|>'
                        )
                    }
                ),
            }
        )
        wrapped_output = model_output.model_copy(update={"choices": [wrapped_choice]})

        processor = IntrinsicsResultProcessor(config_dict=self._make_config())
        result = processor.transform(wrapped_output)
        result_json = json.loads(result.choices[0].message.content)
        assert "answerability_likelihood" in result_json
        assert isinstance(result_json["answerability_likelihood"], float)
        assert result_json["answerability_likelihood"] > 0.9

    def test_disabled_uses_content_directly(self):
        """Without logprobs_workaround, channel tokens in logprobs are ignored."""
        model_output = self._load_answerable_output()
        config = self._make_config()
        config["logprobs_workaround"] = False
        processor = IntrinsicsResultProcessor(config_dict=config)
        # This should work because message.content is clean JSON
        result = processor.transform(model_output)
        result_json = json.loads(result.choices[0].message.content)
        assert "answerability_likelihood" in result_json
