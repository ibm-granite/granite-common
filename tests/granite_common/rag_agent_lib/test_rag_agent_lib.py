# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Tests of code under ``granite_common.rag_agent_lib``
"""

# Standard
import os
import pathlib

# Third Party
import pytest
import yaml

# First Party
from granite_common import ChatCompletion, RagAgentLibRewriter


def _read_file(name):
    with open(name, encoding="utf-8") as f:
        return f.read()


_TEST_DATA_DIR = pathlib.Path(os.path.dirname(__file__)) / "testdata"


_INPUT_JSON_DIR = _TEST_DATA_DIR / "input_json"
_INPUT_YAML_DIR = _TEST_DATA_DIR / "input_yaml"

# Combinations of YAML and JSON files that go together.
_YAML_JSON_COMBOS = {
    # Short name => YAML file, JSON file
    "answerability_simple": (
        _INPUT_YAML_DIR / "answerability.yaml",
        _INPUT_JSON_DIR / "simple.json",
    ),
    "answerability_extra_params": (
        _INPUT_YAML_DIR / "answerability.yaml",
        _INPUT_JSON_DIR / "extra_params.json",
    ),
    "instruction": (
        _INPUT_YAML_DIR / "instruction.yaml",
        _INPUT_JSON_DIR / "instruction.json",
    ),
}


@pytest.fixture(name="yaml_json_combo", scope="module", params=_YAML_JSON_COMBOS)
def _yaml_json_combo(request: pytest.FixtureRequest) -> tuple[str, str]:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different combinations of IO configuration and chat completion request.

    Uses the files in ``testdata/input_json`` and ``testdata/input_yaml``.

    Returns tuple of short name, YAML file, and JSON file
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


def test_canned_input(yaml_json_combo):
    """
    Verify that a given combination of chat completion and rewriting config produces
    the expected output
    """
    short_name, yaml_file, json_file = yaml_json_combo

    # Temporary: Use a YAML file from local disk
    rewriter = RagAgentLibRewriter(config_file=yaml_file)

    json_data = _read_file(json_file)
    before = ChatCompletion.model_validate_json(json_data)
    after = rewriter(before, test_kwarg="George")
    after_json = after.model_dump_json(indent=2)

    expected_file = _TEST_DATA_DIR / f"test_canned_input/{short_name}.json"
    with open(expected_file, encoding="utf-8") as f:
        expected_json = f.read()

    print(f"{after_json=}")
    assert after_json == expected_json
