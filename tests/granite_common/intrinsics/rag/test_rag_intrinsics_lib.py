# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Tests of code under ``granite_common.rag_agent_lib``
"""

# Standard
import copy
import json
import os
import pathlib

# Third Party
import openai
import pydantic
import pytest
import requests
import yaml

# First Party
from granite_common import ChatCompletion, IntrinsicsRewriter
from granite_common.base.types import ChatCompletionResponse
from granite_common.intrinsics import json_util, util
from granite_common.intrinsics.output import IntrinsicsResultProcessor
import granite_common.util


def _read_file(name):
    with open(name, encoding="utf-8") as f:
        return f.read()


_TEST_DATA_DIR = pathlib.Path(os.path.dirname(__file__)) / "testdata"

# Location from which our tests download adapters and YAML files
_RAG_INTRINSICS_REPO_NAME = "ibm-granite/granite-lib-rag-r1.0"


_INPUT_JSON_DIR = _TEST_DATA_DIR / "input_json"
_INPUT_YAML_DIR = _TEST_DATA_DIR / "input_yaml"
_INPUT_ARGS_DIR = _TEST_DATA_DIR / "input_args"


class YamlJsonCombo(pydantic.BaseModel):
    """Dataclass that drives configuration for most tests in this file."""

    short_name: str
    """Short name for the test scenario, for printing to the logs."""
    yaml_file: pathlib.Path | None = None
    """Location of local YAML file, or ``None`` to download from remote repo."""
    inputs_file: pathlib.Path
    """Location of local JSON input file."""
    arguments_file: pathlib.Path | None = None
    """Location of local JSON file of arguments, or ``None`` if no arguments."""
    task: str | None = None
    """Name of target task, used for loading adapters. ``None`` means no adapter and
    no inference tests."""
    is_alora: bool = False
    """``True`` to use the activated LoRA variant of the model for inference tests."""
    repo_id: str = _RAG_INTRINSICS_REPO_NAME
    """Repo on Hugging Face Hub from which the adapter for this intrinsic should be
    loaded."""
    base_model_id: str = "ibm-granite/granite-4.0-micro"
    """Base model on which the target adapter was trained. Should be small enough to
    run on the CI server."""


_YAML_JSON_COMBOS_LIST = [
    # Short name => YAML file, JSON file, model file, arguments file, is aLoRA
    YamlJsonCombo(
        short_name="answerability_simple",
        inputs_file=_INPUT_JSON_DIR / "simple.json",
        task="answerability",
    ),
    YamlJsonCombo(
        short_name="answerability_extra_params",
        yaml_file=_INPUT_YAML_DIR / "answerability.yaml",
        inputs_file=_INPUT_JSON_DIR / "extra_params.json",
        task=None,  # Fake config, no inference
    ),
    YamlJsonCombo(
        short_name="answerability_answerable",
        inputs_file=_INPUT_JSON_DIR / "answerable.json",
        task="answerability",
    ),
    YamlJsonCombo(
        short_name="answerability_answerable_alora",
        inputs_file=_INPUT_JSON_DIR / "answerable.json",
        task="answerability",
        is_alora=True,
    ),
    YamlJsonCombo(
        short_name="answerability_unanswerable",
        inputs_file=_INPUT_JSON_DIR / "unanswerable.json",
        task="answerability",
    ),
    YamlJsonCombo(
        short_name="instruction",
        yaml_file=_INPUT_YAML_DIR / "instruction.yaml",
        inputs_file=_INPUT_JSON_DIR / "instruction.json",
        arguments_file=_INPUT_ARGS_DIR / "instruction.json",
        task=None,  # Fake config, no model
    ),
    YamlJsonCombo(
        short_name="hallucination_detection",
        inputs_file=_INPUT_JSON_DIR / "hallucination_detection.json",
        task="hallucination_detection",
    ),
    # aLoRA adapter for this intrinsic not currently available
    # YamlJsonCombo(
    #     short_name="hallucination_detection_alora",
    #     inputs_file=_INPUT_JSON_DIR / "hallucination_detection.json",
    #     task="hallucination_detection",
    #     is_alora=True
    # ),
    YamlJsonCombo(
        short_name="query_rewrite",
        inputs_file=_INPUT_JSON_DIR / "query_rewrite.json",
        task="query_rewrite",
    ),
    YamlJsonCombo(
        short_name="requirement_check",
        inputs_file=_INPUT_JSON_DIR / "requirement_check.json",
        arguments_file=_INPUT_ARGS_DIR / "requirement_check.json",
        task="requirement_check",
        # Granite 4.0 adapters not currently available
        repo_id="ibm-granite/rag-intrinsics-lib",
        base_model_id="ibm-granite/granite-3.3-2b-instruct",
    ),
    YamlJsonCombo(
        short_name="requirement_check_alora",
        inputs_file=_INPUT_JSON_DIR / "requirement_check.json",
        arguments_file=_INPUT_ARGS_DIR / "requirement_check.json",
        task="requirement_check",
        is_alora=True,
        # Granite 4.0 adapters not currently available
        repo_id="ibm-granite/rag-intrinsics-lib",
        base_model_id="ibm-granite/granite-3.3-2b-instruct",
    ),
    YamlJsonCombo(
        short_name="uncertainty",
        inputs_file=_INPUT_JSON_DIR / "uncertainty.json",
        task="uncertainty",
        # Granite 4.0 adapters not currently available
        repo_id="ibm-granite/rag-intrinsics-lib",
        base_model_id="ibm-granite/granite-3.3-2b-instruct",
    ),
    YamlJsonCombo(
        short_name="uncertainty_alora",
        inputs_file=_INPUT_JSON_DIR / "uncertainty.json",
        task="uncertainty",
        is_alora=True,
        # Granite 4.0 adapters not currently available
        repo_id="ibm-granite/rag-intrinsics-lib",
        base_model_id="ibm-granite/granite-3.3-2b-instruct",
    ),
    YamlJsonCombo(
        short_name="context_relevance",
        inputs_file=_INPUT_JSON_DIR / "context_relevance.json",
        arguments_file=_INPUT_ARGS_DIR / "context_relevance.json",
        task="context_relevance",
    ),
    YamlJsonCombo(
        short_name="context_relevance_alora",
        inputs_file=_INPUT_JSON_DIR / "context_relevance.json",
        arguments_file=_INPUT_ARGS_DIR / "context_relevance.json",
        task="context_relevance",
        is_alora=True,
    ),
    YamlJsonCombo(
        short_name="answer_relevance_classifier",
        inputs_file=_INPUT_JSON_DIR / "answer_relevance_classifier.json",
        task="answer_relevance_classifier",
    ),
    # aLoRA adapter for this intrinsic not currently available
    # YamlJsonCombo(
    #     short_name="answer_relevance_classifier_alora",
    #     inputs_file=_INPUT_JSON_DIR / "answer_relevance_classifier.json",
    #     task="answer_relevance_classifier",
    #     is_alora=True,
    # ),
    YamlJsonCombo(
        short_name="answer_relevance_rewriter",
        inputs_file=_INPUT_JSON_DIR / "answer_relevance_rewriter.json",
        arguments_file=_INPUT_ARGS_DIR / "answer_relevance_rewriter.json",
        task="answer_relevance_rewriter",
    ),
    # aLoRA adapter for this intrinsic not currently available
    # YamlJsonCombo(
    #     short_name="answer_relevance_rewriter_alora",
    #     inputs_file=_INPUT_JSON_DIR / "answer_relevance_rewriter.json",
    #     arguments_file=_INPUT_ARGS_DIR / "answer_relevance_rewriter.json",
    #     task="answer_relevance_rewriter",
    #     is_alora=True,
    # ),
    YamlJsonCombo(
        short_name="citations",
        inputs_file=_INPUT_JSON_DIR / "citations.json",
        task="citations",
    ),
    # aLoRA adapter for this intrinsic not currently available
    # YamlJsonCombo(
    #     short_name="citations_alora",
    #     inputs_file=_INPUT_JSON_DIR / "citations.json",
    #     task="citations",
    #     is_alora=True,
    # ),
]
_YAML_JSON_COMBOS = {c.short_name: c for c in _YAML_JSON_COMBOS_LIST}


# All combinations of input and model
_YAML_JSON_COMBOS_WITH_MODEL = {
    k: v for k, v in _YAML_JSON_COMBOS.items() if v.task is not None
}

# All combinations of input and model that are not aLoRA models (includes no model)
_YAML_JSON_COMBOS_NO_ALORA = {
    k: v for k, v in _YAML_JSON_COMBOS.items() if not v.is_alora
}


@pytest.fixture(name="yaml_json_combo", scope="module", params=_YAML_JSON_COMBOS)
def _yaml_json_combo(request: pytest.FixtureRequest) -> YamlJsonCombo:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different combinations of IO configuration and chat completion request.

    Uses the files in ``testdata/input_json`` and ``testdata/input_yaml``.

    Returns test configuration.
    """
    return _YAML_JSON_COMBOS[request.param]


@pytest.fixture(
    name="yaml_json_combo_no_alora", scope="module", params=_YAML_JSON_COMBOS_NO_ALORA
)
def _yaml_json_combo_no_alora(
    request: pytest.FixtureRequest,
) -> YamlJsonCombo:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different combinations of IO configuration and chat completion request. Ignores
    model configs that use the aLoRA variant of the model.

    Uses the files in ``testdata/input_json`` and ``testdata/input_yaml``.

    Returns tuple of short name, YAML file, JSON file, model directory, and
    arguments file.
    """
    return _YAML_JSON_COMBOS_NO_ALORA[request.param]


@pytest.fixture(
    name="yaml_json_combo_with_model",
    scope="module",
    params=_YAML_JSON_COMBOS_WITH_MODEL,
)
def _yaml_json_combo_with_model(request: pytest.FixtureRequest) -> YamlJsonCombo:
    """Version of :func:`_yaml_json_combo()` fixture with only the inputs that have
    models. Includes an additional flag for whether the model is LoRA or aLoRA
    """
    return _YAML_JSON_COMBOS_WITH_MODEL[request.param]


def test_no_orphan_files():
    """Check whether there are input files that aren't used by any test."""
    used_json_files = set(t.inputs_file for t in _YAML_JSON_COMBOS.values())
    all_json_files = list(_INPUT_JSON_DIR.iterdir())
    used_yaml_files = set(t.yaml_file for t in _YAML_JSON_COMBOS.values())
    all_yaml_files = list(_INPUT_YAML_DIR.iterdir())

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
    assert data["model"] is None

    original_data = copy.deepcopy(data)

    # Instantiate directly from dictionary
    IntrinsicsRewriter(config_dict=data)

    # Data shouldn't be modified
    assert original_data == data

    # Manually run through the make_config_dict() function, because apparently users
    # will try to do this.
    data2 = util.make_config_dict(_INPUT_YAML_DIR / "answerability.yaml")
    IntrinsicsRewriter(config_dict=data2)

    # Read from local disk
    IntrinsicsRewriter(config_file=_INPUT_YAML_DIR / "answerability.yaml")

    # Read from Hugging Face hub.
    local_path = util.obtain_io_yaml(
        "answerability", "granite-4.0-micro", _RAG_INTRINSICS_REPO_NAME
    )
    IntrinsicsRewriter(config_file=local_path)


_CANNED_INPUT_EXPECTED_DIR = _TEST_DATA_DIR / "test_canned_input"


def test_canned_input(yaml_json_combo_no_alora):
    """
    Verify that a given combination of chat completion and rewriting config produces
    the expected output
    """
    cfg = yaml_json_combo_no_alora
    if cfg.arguments_file:
        with open(cfg.arguments_file, encoding="utf8") as f:
            transform_kwargs = json.load(f)
    else:
        transform_kwargs = {}

    # Temporary: Use a YAML file from local disk
    config_file = cfg.yaml_file
    if not config_file:
        # Load from HF Hub if not local file
        config_file = util.obtain_io_yaml(cfg.task, cfg.base_model_id, cfg.repo_id)
    rewriter = IntrinsicsRewriter(config_file=config_file)

    json_data = _read_file(cfg.inputs_file)
    before = ChatCompletion.model_validate_json(json_data)
    after = rewriter.transform(before, **transform_kwargs)
    after_json = after.model_dump_json(indent=2)

    expected_file = _CANNED_INPUT_EXPECTED_DIR / f"{cfg.short_name}.json"
    with open(expected_file, encoding="utf-8") as f:
        expected_json = f.read()

    print(f"{after_json=}")
    assert after_json == expected_json


@pytest.mark.block_network
def test_openai_compat(yaml_json_combo_no_alora):
    """
    Verify that the dataclasses for intrinsics chat completions can be directly passed
    to the OpenAI Python API without raising parsing errors.
    """
    cfg = yaml_json_combo_no_alora
    if cfg.arguments_file:
        with open(cfg.arguments_file, encoding="utf8") as f:
            transform_kwargs = json.load(f)
    else:
        transform_kwargs = {}

    # Temporary: Use a YAML file from local disk
    rewriter = IntrinsicsRewriter(config_file=cfg.yaml_file)
    json_data = _read_file(cfg.inputs_file)
    before = ChatCompletion.model_validate_json(json_data)
    after = rewriter.transform(before, **transform_kwargs)

    # Create a fake connection to the API so we can use its request validation code.
    # Note that network access is blocked for this test case.
    openai_base_url = "http://localhost:98765/not/a/valid/url"
    openai_api_key = "not_a_valid_api_key"
    client = openai.OpenAI(base_url=openai_base_url, api_key=openai_api_key)

    # OpenAI requires a model name
    before.model = "dummy_model_name"
    after.model = "dummy_model_name"

    # The client should get all the way through validation and fail to connect
    with pytest.raises(openai.APIConnectionError):
        client.chat.completions.create(**(before.model_dump()))
    with pytest.raises(openai.APIConnectionError):
        client.chat.completions.create(**(after.model_dump()))


# Combinations of YAML and canned output files that go together.
# Canned output is in test_canned_output/model_output/<short name>.json
_YAML_OUTPUT_COMBOS = {
    # Short name => YAML file
    "answerability_answerable": _INPUT_YAML_DIR / "answerability.yaml",
    "answerability_unanswerable": _INPUT_YAML_DIR / "answerability.yaml",
    "query_rewrite": _INPUT_YAML_DIR / "query_rewrite.yaml",
    "context_relevance": _INPUT_YAML_DIR / "context_relevance.yaml",
    "hallucination_detection": _INPUT_YAML_DIR / "hallucination_detection.yaml",
    "citations": _INPUT_YAML_DIR / "citations.yaml",
    "requirement_check": _INPUT_YAML_DIR / "requirement_check.yaml",
    "answer_relevance_classifier": _INPUT_YAML_DIR / "answer_relevance_classifier.yaml",
    "answer_relevance_rewriter": _INPUT_YAML_DIR / "answer_relevance_rewriter.yaml",
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

    processor = IntrinsicsResultProcessor(config_file=yaml_file)
    with open(input_file, encoding="utf-8") as f:
        model_input = ChatCompletion.model_validate_json(f.read())
    with open(output_file, encoding="utf-8") as f:
        model_output = ChatCompletionResponse.model_validate_json(f.read())

    transformed = processor.transform(model_output, model_input)

    # Pull this string out of the debugger to update expected file
    transformed_str = transformed.model_dump_json(indent=4)

    with open(expected_file, encoding="utf-8") as f:
        expected = ChatCompletionResponse.model_validate_json(f.read())
    expected_str = expected.model_dump_json(indent=4)

    # Do an approximate comparison of numeric values.
    # Can't use pytest.approx() because of lists and floats encoded as strings
    transformed_json = _round_floats(json.loads(transformed_str))
    expected_json = _round_floats(json.loads(expected_str))

    assert transformed_json == expected_json


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


def _round_floats(json_data, num_digits: int = 2):
    """Round all floating-point numbers in a JSON value to facilitate comparisons.

    :param json_data: Arbitrary JSON data.
    :param num_digits: How many decimal points to round to

    :returns: Copy of the input with all floats rounded
    """
    result = copy.deepcopy(json_data)
    for path in json_util.scalar_paths(result):
        value = json_util.fetch_path(result, path)
        if isinstance(value, float):
            json_util.replace_path(result, path, round(value, num_digits))
        elif isinstance(value, str):
            # Test for floating-point number encoded as a string.
            # In Python this test is supposed to use exceptions as control flow.
            try:
                str_as_float = float(value)
                json_util.replace_path(result, path, round(str_as_float, num_digits))
            except ValueError:
                # flow through
                pass

            # Test for JSON object or array encoded as a string
            if value[0] in ("{", "["):
                try:
                    str_as_json = json.loads(value)
                    rounded_json = _round_floats(str_as_json, num_digits)
                    rounded_json_as_str = json.dumps(rounded_json)
                    json_util.replace_path(result, path, rounded_json_as_str)
                except json.JSONDecodeError:
                    # flow through
                    pass
    return result


def test_run_transformers(yaml_json_combo_with_model):
    """
    Run the target model end-to-end on transformers.
    """
    cfg = yaml_json_combo_with_model
    if cfg.arguments_file:
        with open(cfg.arguments_file, encoding="utf8") as f:
            transform_kwargs = json.load(f)
    else:
        transform_kwargs = {}

    # Load input request
    with open(cfg.inputs_file, encoding="utf-8") as f:
        model_input = ChatCompletion.model_validate_json(f.read())

    # Download files from Hugging Face Hub
    try:
        lora_dir = util.obtain_lora(
            cfg.task, cfg.base_model_id, cfg.repo_id, alora=cfg.is_alora
        )
    except requests.exceptions.HTTPError:
        pytest.xfail("Downloads fail on CI server because repo is private")

    # Load IO config YAML for this model
    io_yaml_path = lora_dir / "io.yaml"
    if not os.path.exists(io_yaml_path):
        # Use local files until proper configs are up on Hugging Face
        io_yaml_path = cfg.yaml_file
    rewriter = IntrinsicsRewriter(config_file=io_yaml_path)
    result_processor = IntrinsicsResultProcessor(config_file=io_yaml_path)

    # Prepare inputs for inference
    transformed_input = rewriter.transform(model_input, **transform_kwargs)

    # Run the model using Hugging Face APIs
    model, tokenizer = granite_common.util.load_transformers_lora(lora_dir)
    generate_input, other_input = (
        granite_common.util.chat_completion_request_to_transformers_inputs(
            transformed_input.model_dump(), tokenizer, model
        )
    )
    responses = granite_common.util.generate_with_transformers(
        tokenizer, model, generate_input, other_input
    )

    # Output processing
    transformed_responses = result_processor.transform(responses, transformed_input)

    # Pull this string out of the debugger to create a fresh expected file.
    transformed_str = transformed_responses.model_dump_json(indent=4)
    print(transformed_str)

    with open(
        _TEST_DATA_DIR / f"test_run_transformers/{cfg.short_name}.json",
        encoding="utf-8",
    ) as f:
        expected = ChatCompletionResponse.model_validate_json(f.read())
    # expected_str = expected.model_dump_json(indent=4)

    # Correct for floating point rounding.
    # Can't use pytest.approx() because of lists
    transformed_json = _round_floats(
        json_util.parse_inline_json(transformed_responses.model_dump()), num_digits=2
    )
    expected_json = _round_floats(
        json_util.parse_inline_json(expected.model_dump()), num_digits=2
    )
    if transformed_json != expected_json:
        # Simple comparison failed.
        # Pull out just the content and attempt a more sophisticated comparison
        assert len(transformed_responses.choices) == len(expected.choices)

        for tc, ec in zip(transformed_responses.choices, expected.choices, strict=True):
            t_json = json.loads(tc.message.content)
            e_json = json.loads(ec.message.content)

            assert t_json == pytest.approx(e_json, abs=0.1)
