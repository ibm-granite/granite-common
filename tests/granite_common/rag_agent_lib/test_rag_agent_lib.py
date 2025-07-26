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

# File names minus the ".json" suffix
_INPUT_JSON_DIR = _TEST_DATA_DIR / "input_json"
_INPUT_JSON_DATA = {
    p[:-5]: _read_file(_INPUT_JSON_DIR / p)
    for p in os.listdir(_INPUT_JSON_DIR)
    if p.endswith(".json")
}


@pytest.fixture(name="input_json", scope="module", params=_INPUT_JSON_DATA)
def _input_json(request: pytest.FixtureRequest) -> tuple[str, str]:
    """Pytest fixture that allows us to run a given test case repeatedly with multiple
    different chat completion requests.

    Uses the files in ``testdata/input_json``.

    Returns tuple of file name prefix and file contents
    """
    return request.param, _INPUT_JSON_DATA[request.param]


def test_read_yaml():
    """Verify that reading a model's YAML file from disk works."""
    data_dir = _TEST_DATA_DIR / "test_read_yaml"

    # Read from local disk
    with open(data_dir / "answerability.yaml", encoding="utf8") as file:
        data = yaml.safe_load(file)
    assert data["model"] == "answerability"

    RagAgentLibRewriter(config_file=data_dir / "answerability.yaml")

    # TODO: Test reading from Hugging Face Hub once a suitable YAML file is uploaded


def test_canned_input(input_json):
    """
    Verify that a given combination of chat completion and rewriting config produces
    the expected output
    """
    input_name, json_data = input_json

    # Temporary: Use a YAML file from local disk
    yaml_file = _TEST_DATA_DIR / "test_read_yaml/answerability.yaml"
    rewriter = RagAgentLibRewriter(config_file=yaml_file)

    before = ChatCompletion.model_validate_json(json_data)
    after = rewriter.transform(before)
    after_json = after.model_dump_json(indent=2)

    expected_file = (
        _TEST_DATA_DIR / f"test_canned_input/answerability_{input_name}.json"
    )
    with open(expected_file, encoding="utf-8") as f:
        expected_json = f.read()

    print(f"{after_json=}")

    assert after_json == expected_json
