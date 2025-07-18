# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Tests of code under ``granite_common.granite3.granite32``
"""

# Standard
import json

# Third Party
import pydantic
import pytest
import torch
import transformers

# First Party
from granite_common import (
    AssistantMessage,
    ChatCompletion,
    Granite32ChatCompletion,
    Granite32InputProcessor,
    Granite32OutputProcessor,
    UserMessage,
)
from granite_common.granite3.granite32 import constants
from granite_common.granite3.types import (
    Citation,
    ControlsRecord,
    Document,
    Granite3AssistantMessage,
    Granite3ChatCompletion,
    Hallucination,
)

# All the different chat completion requests that are tested in this file, serialized as
# JSON strings. Represented as a dictionary instead of a list so that pytest output will
# show the short key instead of the long value when referencing a single run of a test
INPUT_JSON_STRS = {
    "simple": """
{
    "messages":
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "Say 'boo'!"}
    ]
}
""",
    "thinking_tag": """
{
    "messages":
    [
        {"role": "user", "content": "What is 1 + 1? Answer with just a number please."}
    ],
    "thinking": true
}
""",
    "hallucinations": """
{
    "messages":
    [
        {"role": "user", "content": "Who invented the flub flibber?"}
    ],
    "documents":
    [
        {"text": "Joe Smith invented the wheel."}
    ],
    "hallucinations": true
}
""",
    "custom_system_prompt": """
{
    "messages":
    [
        {"role": "system", "content": "Answer all questions like a three year old. \
Use as few words as possible. Be extremely concise."},
        {"role": "user", "content": "Hi, I would like some advice on the best tax \
strategy for managing dividend income."}
    ]
}
""",
    "tools": """
{
    "messages":
    [
        {"role": "user", "content": "Where is my money? I'm Joe User and I'm 27 years \
old."}
    ],
    "tools":[
        {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            }
        },
        {
            "name": "find_money",
            "description": "Locate a person's money.",
            "parameters": {
                "type": "object",
                "name": {
                    "type": "string",
                    "description": "Full legal name of the person"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "How old the person is"
                }
            }
        }
    ]
    
}
""",
    "documents": """
{
    "messages":
    [
        {"role": "user", "content": "What's another word for thesaurus?"}
    ],
    "documents":
    [
        {"text": "It's a small world, but I wouldn't want to have to paint it."},
        {"text": "Whenever I think of the past, it brings back so many memories."}
    ]
}
""",
}

msg = UserMessage(content="Hello")
no_thinking_input = ChatCompletion(messages=[msg])
thinking_input = ChatCompletion(messages=[msg], thinking=True)

thought = "Think think"
response = "respond respond"
pre_thought = "something before"
no_cot_output = f"{thought} {response}"
no_thinking_output = f"{thought} {constants.COT_END} {response}"
no_response_output = f"{constants.COT_START}\n\n{response}"
cot_output = f"{constants.COT_START}\n\n{thought}\n{constants.COT_END}\n\n{response}"
cot_alt_output = (
    f"{constants.COT_START_ALTERNATIVES[-1]}\n\n{thought}\n"
    f"{constants.COT_END_ALTERNATIVES[-1]}\n\n{response}"
)
cot_mixed_output = (
    f"{constants.COT_START}\n\n{thought}\n"
    f"{constants.COT_END_ALTERNATIVES[-1]}\n\n{response}"
)
cot_pre_output = (
    f"{pre_thought} {constants.COT_START} {thought} "
    f"{constants.COT_END_ALTERNATIVES[-1]} {response}"
)

no_constituent_output = "Mad about dog!"
citation_example = '<co>1</co> Document 0: "Dog info"'
citation_output = (
    f"{no_constituent_output}<co>1</co>\n\n{constants.CITATION_START}"
    f"\n\n{citation_example}\n\n"
)
hallucination_example = "1. Risk low: Mad about dog"
citation_hallucination_output = (
    f"{citation_output}{constants.HALLUCINATION_START}\n\n{hallucination_example}\n\n"
)
expected_citation = Citation(
    citation_id="1",
    doc_id="0",
    context_text="Dog info",
    context_begin=0,
    context_end=8,
    response_text="Mad about dog!",
    response_begin=0,
    response_end=14,
)
expected_document = Document(doc_id="0", text="Dog info")
doc_input = Granite3ChatCompletion(messages=[msg], documents=[{"text": "Dog info"}])
expected_hallucination = Hallucination(
    hallucination_id="1",
    risk="low",
    response_text="Mad about dog",
    response_begin=0,
    response_end=13,
)


@pytest.fixture(name="input_json_str", scope="module", params=INPUT_JSON_STRS)
def _input_json_str(request: pytest.FixtureRequest) -> str:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different chat completion requests."""
    return INPUT_JSON_STRS[request.param]


@pytest.fixture(name="tokenizer", scope="module")
def _tokenizer() -> transformers.PreTrainedTokenizerBase:
    """Pytest fixture with a shared handle on the tokenizer for the target model."""
    model_path = constants.MODEL_HF_PATH_2B
    try:
        ret = transformers.AutoTokenizer.from_pretrained(
            model_path, local_files_only=False
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.skip(f"No tokenizer for {model_path}: {e}")
    return ret


@pytest.fixture(name="model", scope="module")
def _model() -> transformers.AutoModelForCausalLM:
    """Pytest fixture with a loaded copy of one of the target models for the tests
    in this file."""

    # Prevent thrashing when running tests in parallel
    torch.set_num_threads(2)

    model_path = constants.MODEL_HF_PATH_2B
    try:
        ret = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=False,
            torch_dtype="bfloat16",  # You'll get float32 if you don't set this.
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.skip(f"No model for {model_path}: {e}")
    return ret


@pytest.mark.parametrize(
    ["length", "originality", "error"],
    [
        (None, None, None),
        ("short", None, None),
        (None, "abstractive", None),
        ("long", "extractive", None),
        ("BAD_VAL", "abstractive", "input_value='BAD_VAL'"),
        ("long", "BAD_VAL", "input_value='BAD_VAL'"),
        ("BAD_VAL", "Another Bad Value", "input_value='BAD_VAL'"),
        ("ShOrT", None, "input_value='ShOrT'"),
        (None, "aBsTrAcTiVe", "input_value='aBsTrAcTiVe'"),
        (1, None, "input_type=int"),
        (None, 2, "input_type=int"),
    ],
)
def test_controls_field_validators(length, originality, error):
    if error:
        with pytest.raises(pydantic.ValidationError, match=error):
            ControlsRecord(length=length, originality=originality)
    else:
        ControlsRecord(length=length, originality=originality)


def test_read_inputs(input_json_str):
    """
    Verify that the dataclasses for the Granite 3.2 I/O processor can parse Granite
    3.2 JSON
    """
    input_json = json.loads(input_json_str)
    input_obj = ChatCompletion.model_validate(input_json)
    input_obj_2 = ChatCompletion.model_validate_json(input_json_str)

    assert input_obj == input_obj_2

    # Parse additional Model-specific fields
    granite_input_obj = Granite32ChatCompletion.model_validate(input_obj.model_dump())

    # Verify that we can convert back to JSON without crashing
    granite_input_obj.model_dump_json()
    input_obj.model_dump_json()


def test_same_input_string(
    tokenizer: transformers.PreTrainedTokenizerBase, input_json_str: str
):
    """
    Verify that the I/O processor produces the exact same input string as the Jinja
    template that ships with the model.
    """

    # First apply the Jinja template
    input_json = json.loads(input_json_str)
    input_kwargs = input_json.copy()
    del input_kwargs["messages"]
    transformers_str = tokenizer.apply_chat_template(
        input_json["messages"],
        **input_kwargs,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Then compare against the input processor
    inputs = Granite32ChatCompletion.model_validate_json(input_json_str)
    io_proc_str = Granite32InputProcessor().transform(inputs)

    print(f"{io_proc_str=}")
    print(f"{transformers_str=}")

    assert io_proc_str == transformers_str


def test_basic_inputs_to_string():
    """
    Basic test against canned output in case the developer doesn't have a way to load
    an actual Granite 3.2 tokenizer for output comparisons.

    Chat input:

    chat = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]

    Expected similar (dates will vary) chat template request generated:

    <|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
    Today's Date: February 17, 2025.
    You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Hello, how are you?<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>I'm doing great. How can I help you today?\
<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>I'd like to show off how chat templating works!\
<|end_of_text|>
    """
    chatRequest = Granite32InputProcessor().transform(
        chat_completion=ChatCompletion(
            messages=[
                UserMessage(content="Hello, how are you?"),
                AssistantMessage(content="I'm doing great. How can I help you today?"),
                UserMessage(content="I'd like to show off how chat templating works!"),
            ]
        ),
        add_generation_prompt=False,
    )

    chatReqStart = "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date:"
    assert chatRequest.startswith(chatReqStart)

    chatReqModelMsg = "You are Granite, developed by IBM. You are a helpful AI \
assistant.<|end_of_text|>"
    assert chatReqModelMsg in chatRequest

    chatReqBody = """\
<|start_of_role|>user<|end_of_role|>Hello, how are you?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>I'm doing great. How can I help you today?\
<|end_of_text|>
<|start_of_role|>user<|end_of_role|>I'd like to show off how chat templating works!\
<|end_of_text|>"""
    assert chatReqBody in chatRequest

    assert chatRequest.endswith("")


def test_run_model(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.PreTrainedTokenizerBase,
    input_json_str: str,
):
    """
    Run inference end-to-end with each of the test inputs in this file.
    """
    chat_completion = Granite32ChatCompletion.model_validate_json(input_json_str)
    prompt = Granite32InputProcessor().transform(chat_completion)
    model_input = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = transformers.GenerationConfig(
        max_length=1024, num_beams=1, do_sample=False
    )
    model_output_tensor = model.generate(
        **model_input, generation_config=generation_config
    )
    assert model_output_tensor.shape[0] == 1
    model_output = tokenizer.decode(
        model_output_tensor[0, model_input["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    next_message = Granite32OutputProcessor().transform(model_output, chat_completion)

    print(f"{next_message=}")

    assert isinstance(next_message, Granite3AssistantMessage)
    assert (
        next_message.content or next_message.tool_calls
    )  # Make sure we don't get empty result

    # TODO: Verify outputs in greater detail


@pytest.mark.parametrize(
    ["chat_completion", "model_output", "exp_thought", "exp_resp"],
    [
        # No thinking flag
        (no_thinking_input, no_thinking_output, None, no_thinking_output),
        (no_thinking_input, cot_output, None, cot_output),
        # Thinking flag
        (thinking_input, no_cot_output, None, no_cot_output),
        (thinking_input, no_thinking_output, None, no_thinking_output),
        (thinking_input, no_response_output, None, no_response_output),
        (thinking_input, cot_output, thought, response),
        (thinking_input, cot_alt_output, thought, response),
        (thinking_input, cot_mixed_output, thought, response),
        (thinking_input, cot_pre_output, thought, f"{pre_thought} {response}"),
    ],
)
def test_cot_parsing(chat_completion, model_output, exp_thought, exp_resp):
    """Test the parsing logic for CoT reasoning output"""
    result = Granite32OutputProcessor().transform(model_output, chat_completion)
    assert result.reasoning_content == exp_thought
    assert result.content == exp_resp
    assert result.raw_content is None or result.raw_content == model_output


@pytest.mark.parametrize(
    [
        "chat_completion",
        "model_output",
        "exp_document",
        "exp_citation",
        "exp_hallucination",
        "exp_resp",
    ],
    [
        # No constituents
        (
            no_thinking_input,
            no_constituent_output,
            None,
            None,
            None,
            no_constituent_output,
        ),
        # Citation
        (
            doc_input,
            citation_output,
            [expected_document],
            [expected_citation],
            None,
            no_constituent_output,
        ),
        # Citation and hallucination
        (
            doc_input,
            citation_hallucination_output,
            [expected_document],
            [expected_citation],
            [expected_hallucination],
            no_constituent_output,
        ),
    ],
)
def test_citation_hallucination_parsing(
    chat_completion,
    model_output,
    exp_document,
    exp_citation,
    exp_hallucination,
    exp_resp,
):
    """Test the parsing logic for Rag and hallucinations output"""
    result = Granite32OutputProcessor().transform(model_output, chat_completion)
    assert result.content == exp_resp
    assert result.citations == exp_citation
    assert result.documents == exp_document
    assert result.hallucinations == exp_hallucination
