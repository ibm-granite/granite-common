# SPDX-License-Identifier: Apache-2.0

"""Tests for deriving content from logprob token texts in IntrinsicsResultProcessor."""

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
    _content_from_logprobs,
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


# Single channel sequence wrapping a payload.
_SINGLE_CHANNEL_PREFIX = [
    ("<|channel|>", 0.0),
    ("\n", 0.0),
    ("final", 0.0),
    ("\n", 0.0),
    ("<|message|>", 0.0),
    ("\n", 0.0),
]
_SINGLE_CHANNEL_SUFFIX = [
    ("\n", 0.0),
    ("<|return|>", 0.0),
]

# Multi-channel sequence (analysis + role markers + final) as seen from
# gpt-oss hallucination_detection.
_MULTI_CHANNEL_PREFIX = [
    ("<|channel|>", 0.0),
    ("analysis", 0.0),
    ("<|message|>", 0.0),
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


class TestContentFromLogprobs:
    """Tests for the _content_from_logprobs helper."""

    def test_payload_only(self):
        """Logprobs with only payload tokens return content unchanged."""
        logprobs = _make_logprobs(_PAYLOAD_TOKENS)
        result = _content_from_logprobs(logprobs)
        assert result is not None
        content, lp = result
        assert content == '"answerable"'
        tokens = [c.token for c in lp.content]
        assert tokens == ['"', "answer", "able", '"']

    def test_single_channel_wrapper(self):
        """Single channel wrapper is stripped, payload extracted."""
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + _PAYLOAD_TOKENS + _SINGLE_CHANNEL_SUFFIX
        )
        result = _content_from_logprobs(logprobs)
        assert result is not None
        content, lp = result
        assert content == '"answerable"'
        tokens = [c.token for c in lp.content]
        assert tokens == ['"', "answer", "able", '"']

    def test_multi_channel_wrapper(self):
        """Multiple channel sequences are stripped, payload extracted."""
        logprobs = _make_logprobs(_MULTI_CHANNEL_PREFIX + _PAYLOAD_TOKENS)
        result = _content_from_logprobs(logprobs)
        assert result is not None
        content, lp = result
        assert content == '"answerable"'
        tokens = [c.token for c in lp.content]
        assert tokens == ['"', "answer", "able", '"']

    def test_json_object_payload(self):
        """JSON object payload is extracted from channel tokens."""
        payload = [("{", 0.0), ('"key"', 0.0), (":", 0.0), ('"val"', 0.0), ("}", 0.0)]
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + payload + _SINGLE_CHANNEL_SUFFIX
        )
        result = _content_from_logprobs(logprobs)
        assert result is not None
        content, _lp = result
        assert content == '{"key":"val"}'
        assert json.loads(content) == {"key": "val"}

    def test_json_array_payload(self):
        """JSON array payload is extracted from channel tokens."""
        payload = [("[", 0.0), ("1", 0.0), (",", 0.0), ("2", 0.0), ("]", 0.0)]
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + payload + _SINGLE_CHANNEL_SUFFIX
        )
        result = _content_from_logprobs(logprobs)
        assert result is not None
        content, _lp = result
        assert content == "[1,2]"
        assert json.loads(content) == [1, 2]

    def test_none_content(self):
        """Logprobs with None content returns None."""
        logprobs = ChatCompletionLogProbs(content=None)
        assert _content_from_logprobs(logprobs) is None


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


class TestResultProcessorWithLogprobGroundTruth:
    """End-to-end test of IntrinsicsResultProcessor using logprob ground truth."""

    def _make_config(self):
        return {
            "model": None,
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
        """Single channel wrapper in logprobs, clean content."""
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
        """Multiple channel sequences in logprobs, clean content."""
        model_output = self._load_answerable_output()
        original_logprobs = model_output.choices[0].logprobs
        wrapped_logprobs = _wrap_logprobs_with_channel(
            original_logprobs, _MULTI_CHANNEL_PREFIX
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

    def test_clean_logprobs_no_channel(self):
        """Clean logprobs without channel tokens still work correctly."""
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

        # Wrap logprobs and also put channel tokens in content
        wrapped_logprobs = _wrap_logprobs_with_channel(
            original_logprobs, _SINGLE_CHANNEL_PREFIX, _SINGLE_CHANNEL_SUFFIX
        )
        wrapped_choice = model_output.choices[0].model_copy(
            update={
                "logprobs": wrapped_logprobs,
                "message": model_output.choices[0].message.model_copy(
                    update={
                        "content": (
                            '<|channel|>\nfinal\n<|message|>\n"answerable"\n<|return|>'
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
