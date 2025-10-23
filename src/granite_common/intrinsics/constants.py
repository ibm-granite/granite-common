# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Constants relating to of input and output processing for LoRA adapters in IBM's 
`rag-agent-lib` library of intrinsics.
"""

INTRINSICS_LIB_REPO_ID = "ibm-granite/granite-3.3-8b-rag-agent-lib"
YAML_REQUIRED_FIELDS = [
    "model",
    "response_format",
    "transformations",
]
"""Fields that must be present in every intrinsic's YAML configuration file."""

YAML_OPTIONAL_FIELDS = [
    "docs_as_message",
    "instruction",
    "parameters",
    "sentence_boundaries",
]
"""Fields that may be present in every intrinsic's YAML configuration file. If 
not present, the parsed config dictionary will contain a null value in thier place."""

YAML_JSON_FIELDS = [
    "response_format",
]
"""Fields of the YAML file that contain JSON values as strings"""

RAG_INTRINSICS_LIB_REPO_NAME = "ibm-granite/rag-intrinsics-lib"
"""Location of the RAG intrinsics library on Huggingface Hub"""

BASE_MODEL_TO_CANONICAL_NAME = {
    "ibm-granite/granite-3.3-8b-instruct": "granite-3.3-8b-instruct",
    "ibm-granite/granite-3.3-2b-instruct": "granite-3.3-2b-instruct",
    "granite-3.3-8b-instruct": "granite-3.3-8b-instruct",
    "granite-3.3-2b-instruct": "granite-3.3-2b-instruct",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-20b": "gpt-oss-20b",
}
"""Base model names that we accept for LoRA/aLoRA adapters in intrinsics-lib.
Each model name maps to the name of the directory that contains (a)LoRA adapters for
that model."""

TOP_LOGPROBS = 10
"""Number of logprobs we request per token when decoding logprobs."""
