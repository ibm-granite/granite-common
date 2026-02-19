# SPDX-License-Identifier: Apache-2.0

"""Tests for channel token stripping in IntrinsicsResultProcessor."""

# Standard
import json
import pathlib

# Third Party
import pytest

# First Party
from granite_common.base.types import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionResponse,
)
from granite_common.intrinsics.output import (
    IntrinsicsResultProcessor,
    _strip_channel_tokens,
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


class TestStripChannelTokens:
    """Tests for the _strip_channel_tokens helper."""

    def test_no_channel_tokens(self):
        """Content and logprobs without channel tokens are unchanged."""
        content = '"answerable"'
        logprobs = _make_logprobs(_PAYLOAD_TOKENS)
        stripped, lp = _strip_channel_tokens(content, logprobs)
        assert stripped == '"answerable"'
        assert lp is logprobs  # Same object, not copied

    def test_clean_content_single_channel_logprobs(self):
        """Content is clean, logprobs have a single channel wrapper."""
        content = '"answerable"'
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + _PAYLOAD_TOKENS + _SINGLE_CHANNEL_SUFFIX
        )
        stripped, lp = _strip_channel_tokens(content, logprobs)
        assert stripped == '"answerable"'
        payload_tokens = [c.token for c in lp.content]
        assert payload_tokens == ['"', "answer", "able", '"']

    def test_clean_content_multi_channel_logprobs(self):
        """Content is clean, logprobs have multiple channel sequences."""
        content = '"answerable"'
        logprobs = _make_logprobs(_MULTI_CHANNEL_PREFIX + _PAYLOAD_TOKENS)
        stripped, lp = _strip_channel_tokens(content, logprobs)
        assert stripped == '"answerable"'
        payload_tokens = [c.token for c in lp.content]
        assert payload_tokens == ['"', "answer", "able", '"']

    def test_channel_tokens_in_both(self):
        """Channel tokens in both content and logprobs are stripped."""
        content = '<|channel|>\nfinal\n<|message|>\n"answerable"\n<|return|>'
        logprobs = _make_logprobs(
            _SINGLE_CHANNEL_PREFIX + _PAYLOAD_TOKENS + _SINGLE_CHANNEL_SUFFIX
        )
        stripped, lp = _strip_channel_tokens(content, logprobs)
        assert stripped == '"answerable"'
        payload_tokens = [c.token for c in lp.content]
        assert payload_tokens == ['"', "answer", "able", '"']

    def test_strip_content_without_logprobs(self):
        """Channel tokens in content are stripped when logprobs is None."""
        content = '<|channel|>\nfinal\n<|message|>\n"answerable"\n<|return|>'
        stripped, lp = _strip_channel_tokens(content, None)
        assert stripped == '"answerable"'
        assert lp is None

    def test_clean_content_no_logprobs(self):
        """Clean content with no logprobs passes through."""
        content = '"unanswerable"'
        stripped, lp = _strip_channel_tokens(content, None)
        assert stripped == '"unanswerable"'
        assert lp is None


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


class TestResultProcessorWithChannelTokens:
    """End-to-end test of IntrinsicsResultProcessor with channel tokens."""

    def _make_config(self, strip=True):
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
            "strip_channel_tokens": strip,
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

        processor = IntrinsicsResultProcessor(config_dict=self._make_config(strip=True))
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

        processor = IntrinsicsResultProcessor(config_dict=self._make_config(strip=True))
        result = processor.transform(wrapped_output)
        result_json = json.loads(result.choices[0].message.content)
        assert "answerability_likelihood" in result_json
        assert isinstance(result_json["answerability_likelihood"], float)
        assert result_json["answerability_likelihood"] > 0.9

    def test_disabled_with_channel_content_fails(self):
        """Without strip_channel_tokens, channel content causes failure."""
        config = self._make_config(strip=False)
        del config["strip_channel_tokens"]
        processor = IntrinsicsResultProcessor(config_dict=config)

        raw = {
            "id": "test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": (
                            '<|channel|>\nfinal\n<|message|>\n"answerable"\n<|return|>'
                        ),
                        "role": "assistant",
                    },
                }
            ],
            "created": 0,
            "model": "test",
        }
        model_output = ChatCompletionResponse.model_validate(raw)
        with pytest.raises(json.JSONDecodeError):
            processor.transform(model_output)
