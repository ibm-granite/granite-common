# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Tests of code under ``granite_common.rag_agent_lib``
"""

# Standard
import json
import os
import pathlib

# Third Party
import pytest
import yaml

# First Party
from granite_common import ChatCompletion, RagAgentLibRewriter
from granite_common.base.types import ChatCompletionResponse
from granite_common.rag_agent_lib import json_util
from granite_common.rag_agent_lib.output import RagAgentLibResultProcessor


def _read_file(name):
    with open(name, encoding="utf-8") as f:
        return f.read()


_TEST_DATA_DIR = pathlib.Path(os.path.dirname(__file__)) / "testdata"


_INPUT_JSON_DIR = _TEST_DATA_DIR / "input_json"
_INPUT_YAML_DIR = _TEST_DATA_DIR / "input_yaml"


# Combinations of YAML and JSON files that go together.
_YAML_JSON_COMBOS = {
    # Short name => YAML file, JSON file, model file
    "answerability_simple": (
        _INPUT_YAML_DIR / "answerability.yaml",
        _INPUT_JSON_DIR / "simple.json",
        "ibm-granite/intrinsics-lib/answerability/lora/granite-3.3-2b-instruct",
    ),
    "answerability_extra_params": (
        _INPUT_YAML_DIR / "answerability.yaml",
        _INPUT_JSON_DIR / "extra_params.json",
        None,
    ),
    "answerability_answerable": (
        _INPUT_YAML_DIR / "answerability.yaml",
        _INPUT_JSON_DIR / "answerable.json",
        "ibm-granite/intrinsics-lib/answerability/lora/granite-3.3-2b-instruct",
    ),
    "instruction": (
        _INPUT_YAML_DIR / "instruction.yaml",
        _INPUT_JSON_DIR / "instruction.json",
        None,  # Fake config, no model
    ),
    "hallucination": (
        _INPUT_YAML_DIR / "hallucination.yaml",
        _INPUT_JSON_DIR / "hallucination.json",
        None,  # TODO: Add model once we have a checkpoint
    ),
}

_YAML_JSON_COMBOS_WITH_MODEL = {
    k: v for k, v in _YAML_JSON_COMBOS.items() if v[2] is not None
}


@pytest.fixture(name="yaml_json_combo", scope="module", params=_YAML_JSON_COMBOS)
def _yaml_json_combo(request: pytest.FixtureRequest) -> tuple[str, str, str]:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different combinations of IO configuration and chat completion request.

    Uses the files in ``testdata/input_json`` and ``testdata/input_yaml``.

    Returns tuple of short name, YAML file, JSON file, and model directory
    """
    return (request.param,) + _YAML_JSON_COMBOS[request.param]


@pytest.fixture(
    name="yaml_json_combo_with_model",
    scope="module",
    params=_YAML_JSON_COMBOS_WITH_MODEL,
)
def _yaml_json_combo_with_model(request: pytest.FixtureRequest) -> tuple[str, str, str]:
    """Version of :func:`_yaml_json_combo()` fixture with only the inputs that have
    models
    """
    return (request.param,) + _YAML_JSON_COMBOS[request.param]


def test_no_orphan_files():
    """Check whether there are input files that aren't used by any test."""
    used_json_files = set(t[1].name for t in _YAML_JSON_COMBOS.values())
    all_json_files = os.listdir(_INPUT_JSON_DIR)
    used_yaml_files = set(t[0].name for t in _YAML_JSON_COMBOS.values())
    all_yaml_files = os.listdir(_INPUT_YAML_DIR)

    for f in all_json_files:
        if f not in used_json_files:
            raise ValueError(
                f"JSON File '{f}' not used. Files are {all_json_files}; "
                f"Used files are {list(used_json_files)}"
            )
    for f in all_yaml_files:
        if f not in used_yaml_files:
            raise ValueError(
                f"YAML File '{f}' not used. Files are {all_yaml_files}; "
                f"Used files are {list(used_yaml_files)}"
            )


def test_read_yaml():
    """Sanity check to verify that reading a model's YAML file from disk works."""
    # Read from local disk
    with open(_INPUT_YAML_DIR / "answerability.yaml", encoding="utf8") as file:
        data = yaml.safe_load(file)
    assert data["model"] == "answerability"

    RagAgentLibRewriter(config_file=_INPUT_YAML_DIR / "answerability.yaml")

    # TODO: Test reading from Hugging Face Hub once a suitable YAML file is uploaded


_CANNED_INPUT_EXPECTED_DIR = _TEST_DATA_DIR / "test_canned_input"


def test_canned_input(yaml_json_combo):
    """
    Verify that a given combination of chat completion and rewriting config produces
    the expected output
    """
    short_name, yaml_file, json_file, _ = yaml_json_combo

    # Temporary: Use a YAML file from local disk
    rewriter = RagAgentLibRewriter(config_file=yaml_file)

    json_data = _read_file(json_file)
    before = ChatCompletion.model_validate_json(json_data)
    after = rewriter.transform(before, test_kwarg="George")
    after_json = after.model_dump_json(indent=2)

    expected_file = _CANNED_INPUT_EXPECTED_DIR / f"{short_name}.json"
    with open(expected_file, encoding="utf-8") as f:
        expected_json = f.read()

    print(f"{after_json=}")
    assert after_json == expected_json


# Combinations of YAML and canned output files that go together.
# Canned output is in test_canned_output/model_output/<short name>.json
_YAML_OUTPUT_COMBOS = {
    # Short name => YAML file
    "answerability_answerable": _INPUT_YAML_DIR / "answerability.yaml",
    "answerability_unanswerable": _INPUT_YAML_DIR / "answerability.yaml",
}


_CANNED_OUTPUT_MODEL_INPUT_DIR = _TEST_DATA_DIR / "test_canned_output/model_input"
_CANNED_OUTPUT_MODEL_OUTPUT_DIR = _TEST_DATA_DIR / "test_canned_output/model_output"
_CANNED_OUTPUT_EXPECTED_DIR = _TEST_DATA_DIR / "test_canned_output/expected_result"


@pytest.fixture(name="yaml_output_combo", scope="module", params=_YAML_OUTPUT_COMBOS)
def _yaml_output_combo(request: pytest.FixtureRequest) -> tuple[str, str]:
    """Pytest fixture that iterates over the various inputs to
    :func:`test_canned_output()`

    :returns: Tuple of:
     * short name of test case
     * location of YAML file
     * location of model input file
     * location of model raw output file
     * location of expected file
    """
    return (
        request.param,
        _YAML_OUTPUT_COMBOS[request.param],
        _CANNED_OUTPUT_MODEL_INPUT_DIR / f"{request.param}.json",
        _CANNED_OUTPUT_MODEL_OUTPUT_DIR / f"{request.param}.json",
        _CANNED_OUTPUT_EXPECTED_DIR / f"{request.param}.json",
    )


def test_canned_output(yaml_output_combo):
    """
    Verify that the output processing for each model works on previous model outputs
    read from disk. Model outputs are stored in OpenAI format.

    :param yaml_output_combo: Fixture containing pairs of short name, IO YAML file
    """
    _, yaml_file, input_file, output_file, expected_file = yaml_output_combo

    processor = RagAgentLibResultProcessor(config_file=yaml_file)
    with open(input_file, encoding="utf-8") as f:
        model_input = ChatCompletion.model_validate_json(f.read())
    with open(output_file, encoding="utf-8") as f:
        model_output = ChatCompletionResponse.model_validate_json(f.read())

    transformed = processor.transform(model_output, model_input)

    transformed_str = transformed.model_dump_json(indent=4)

    with open(expected_file, encoding="utf-8") as f:
        expected = ChatCompletionResponse.model_validate_json(f.read())
    expected_str = expected.model_dump_json(indent=4)

    assert transformed_str == expected_str


_REPARSE_JSON_DIR = _TEST_DATA_DIR / "test_reparse_json"
_REPARSE_JSON_FILES = [
    name for name in os.listdir(_REPARSE_JSON_DIR) if name.endswith(".json")
]


@pytest.fixture(name="reparse_json_file", scope="module", params=_REPARSE_JSON_FILES)
def _reparse_json_file(request: pytest.FixtureRequest) -> tuple[str, str, str]:
    """Pytest fixture that returns each file in _REPARSE_JSON_DIR in turn"""
    return request.param


def test_reparse_json(reparse_json_file):
    """Ensure that we can reparse JSON data to find position information for
    literals."""
    json_file = _REPARSE_JSON_DIR / reparse_json_file
    json_str = _read_file(json_file)

    parsed_json = json.loads(json_str)
    reparsed_json = json_util.reparse_json_with_offsets(json_str)

    assert json_util.scalar_paths(parsed_json) == json_util.scalar_paths(reparsed_json)


# def test_run_transformers(yaml_json_combo_with_model):
#     """
#     Run the target model on transformers.
#     """
#     short_name, yaml_file, json_file, model_path = yaml_json_combo_with_model

#     #model = peft.PeftModel.from_pretrained(model_path)
#     model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
