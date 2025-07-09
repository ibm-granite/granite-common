# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Tests of code under ``granite_commons.granite3.granite32``
"""

# Standard
import json

# Third Party
import pydantic
import pytest
import transformers

# First Party
from granite_commons.base.types import ChatCompletion
from granite_commons.granite3.granite32 import constants, io, types

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
        {"role": "user", "content": "I'd like to show off how chat templating works!"}
    ]
}
""",
    "thinking_tag": """
{
    "messages":
    [
        {"role": "user", "content": "How much wood would a wood chuck chuck?"}
    ],
    "thinking": true
}
""",
    "custom_system_prompt": """
{
    "messages":
    [
        {"role": "system", "content": "Answer all questions like a three year old."},
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
}


@pytest.fixture(scope="session", params=INPUT_JSON_STRS)
def input_json_str(request: pytest.FixtureRequest) -> str:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different chat completion requests."""
    return INPUT_JSON_STRS[request.param]


@pytest.fixture(scope="session")
def tokenizer() -> transformers.PreTrainedTokenizerBase:
    """Pytext fixture with a shared handle on the tokenizer for the target model."""
    model_path = constants.MODEL_HF_PATH_2B
    try:
        ret = transformers.AutoTokenizer.from_pretrained(
            model_path, local_files_only=False
        )
    except Exception as e:
        pytest.skip(f"No tokenizer for {model_path}: {e}")
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
            types.ControlsRecord(length=length, originality=originality)
    else:
        types.ControlsRecord(length=length, originality=originality)


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
    granite_input_obj = types.Granite3Point2ChatCompletion.model_validate(
        input_obj.model_dump()
    )

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
    inputs = types.Granite3Point2ChatCompletion.model_validate_json(input_json_str)
    io_proc_str = io.Granite3Point2InputProcessor().transform(inputs)

    print(f"{io_proc_str=}")
    print(f"{transformers_str=}")

    assert io_proc_str == transformers_str
