# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Constants relating to of input and output processing for LoRA adapters in IBM's 
`rag-agent-lib` library of intrinsics.
"""

YAML_REQUIRED_FIELDS = [
    "model",
    "response_format",
    "transformations",
    "instruction",
    "parameters",
    "sentence_boundaries",
]
"""Fields that must be present in every intrinsic's YAML configuration file."""

YAML_JSON_FIELDS = [
    "response_format",
]
"""Fields of the YAML file that contain JSON values as strings"""

INTRINSICS_LIB_REPO_NAME = "ibm-granite/intrinsics-lib"
"""Location of the intrinsics library on Huggingface Hub"""

BASE_MODEL_TO_CANONICAL_NAME = {
    "ibm-granite/granite-3.3-8b-instruct": "granite-3.3-8b-instruct",
    "ibm-granite/granite-3.3-2b-instruct": "granite-3.3-2b-instruct",
    "granite-3.3-8b-instruct": "granite-3.3-8b-instruct",
    "granite-3.3-2b-instruct": "granite-3.3-2b-instruct",
}
"""Base model names that we accept for LoRA/aLoRA adapters in intrinsics-lib."""

TOP_LOGPROBS = 10
"""Number of logprobs we request per token when decoding logprobs."""
