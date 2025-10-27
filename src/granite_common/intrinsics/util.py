# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Common utility functions for this package.
"""

# Standard
import copy
import json
import os
import pathlib

# Third Party
import yaml

# Local
from .constants import (
    BASE_MODEL_TO_CANONICAL_NAME,
    RAG_INTRINSICS_LIB_REPO_NAME,
    YAML_JSON_FIELDS,
    YAML_OPTIONAL_FIELDS,
    YAML_REQUIRED_FIELDS,
)


def make_config_dict(
    config_file: str | pathlib.Path | None = None, config_dict: dict | None = None
) -> dict | None:
    """Common initialization code for reading YAML config files in factory classes.

    Also parses JSON fields.
    """
    if (config_file is None and config_dict is None) or (
        config_file is not None and config_dict is not None
    ):
        raise ValueError("Exactly one of config_file and config_dict must be set.")

    all_fields = sorted(YAML_REQUIRED_FIELDS + YAML_OPTIONAL_FIELDS)

    if config_dict:
        # Don't modify input
        config_dict = copy.deepcopy(config_dict)
    if config_file:
        with open(config_file, encoding="utf8") as file:
            config_dict = yaml.safe_load(file)

    # Validate top-level field names. No schema checking for YAML, so we need to do this
    # manually.
    for field in YAML_REQUIRED_FIELDS:
        if field not in config_dict:
            raise ValueError(f"Configuration is missing required field '{field}'")
    for name in config_dict:
        if name not in all_fields:
            raise ValueError(
                f"Configuration contains unexpected top-level field "
                f"'{name}'. Known top level fields are: {all_fields}"
            )
    for name in YAML_OPTIONAL_FIELDS:
        # Optional fields should be None if not present, to simplify downstream code.
        if name not in config_dict:
            config_dict[name] = None

    # Parse fields that contain JSON data.
    for name in YAML_JSON_FIELDS:
        if config_dict[name]:
            value = config_dict[name]
            # Users seem to be intent on passing YAML data through this function
            # multiple times, so we assume that values other than a string have already
            # been parsed by a previous call of this function.
            if isinstance(value, str):
                try:
                    config_dict[name] = json.loads(value)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Error parsing JSON in '{name}' field. Raw value was '{value}'"
                    ) from e

    return config_dict


def obtain_lora(
    intrinsic_name: str,
    target_model_name: str,
    /,
    alora: bool = False,
    repo_id=RAG_INTRINSICS_LIB_REPO_NAME,
    cache_dir: str | None = None,
    file_glob: str = "*",
) -> pathlib.Path:
    """
    Downloads a cached copy of a LoRA or aLoRA adapter from the
    [Granite Intrinsics Library](
    https://huggingface.co/ibm-granite/granite-3.3-8b-rag-agent-lib) if one is not
    already in the local cache.

    :param intrinsic_name: Short name of the intrinsic model, such as "certainty".
    :param target_model_name: Name of the base model for the LoRA or aLoRA adapter.
    :param alora: If ``True``, load aLoRA version of intrinsic; otherwise use LoRA
    :param repo_id: Optional name of Hugging Face Hub repository containing a collection
        of LoRA and/or aLoRA adapters for intrinsics.
        Default is to use rag-intrinsics-lib.
    :param cache_dir: Local directory to use as a cache (in Hugging Face Hub format),
        or ``None`` to use the Hugging Face Hub default location.
    :param file_glob: Only files that match this glob will be downloaded to the cache.

    :returns: the full path to the local copy of the specified (a)LoRA adapter.
    This path is suitable for passing to commands that will serve the adapter.
    """
    # Third Party
    import huggingface_hub

    # Normalize target model name.
    # Confusing syntax here brought to you by pylint.
    target_model_name = BASE_MODEL_TO_CANONICAL_NAME.get(
        target_model_name, target_model_name
    )

    lora_str = "alora" if alora else "lora"

    lora_subdir_name = f"{intrinsic_name}/{lora_str}/{target_model_name}"

    # Download just the files for this LoRA if not already present
    local_root_path = huggingface_hub.snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{lora_subdir_name}/{file_glob}",
        cache_dir=cache_dir,
    )
    lora_dir = pathlib.Path(local_root_path) / lora_subdir_name

    # Hugging Face Hub API will happily download nothing. Check whether that happened.
    if not os.path.exists(lora_dir):
        raise ValueError(
            f"Intrinsic '{intrinsic_name}' as "
            f"{'aLoRA' if alora else 'LoRA'} adapter on base model "
            f"'{target_model_name}' not found in "
            f"{repo_id} repository on Hugging Face Hub. "
            f"Searched for path {lora_subdir_name}/{file_glob}"
        )

    return lora_dir


def obtain_io_yaml(
    intrinsic_name: str,
    target_model_name: str,
    /,
    alora: bool = False,
    repo_id=RAG_INTRINSICS_LIB_REPO_NAME,
    cache_dir: str | None = None,
) -> pathlib.Path:
    """
    Downloads a cached copy of an ``io.yaml`` configuration file for an intrinsic in
    the [Granite Intrinsics Library](
    https://huggingface.co/ibm-granite/granite-3.3-8b-rag-agent-lib) if one is not
    already in the local cache.

    :param intrinsic_name: Short name of the intrinsic model, such as "certainty".
    :param target_model_name: Name of the base model for the LoRA or aLoRA adapter.
    :param alora: If ``True``, load aLoRA version of intrinsic; otherwise use LoRA
    :param repo_id: Optional name of Hugging Face Hub repository containing a collection
        of LoRA and/or aLoRA adapters for intrinsics.
        Default is to use rag-intrinsics-lib.
    :param cache_dir: Local directory to use as a cache (in Hugging Face Hub format),
        or ``None`` to use the Hugging Face Hub default location.

    :returns: the full path to the local copy of the specified (a)LoRA adapter.
    This path is suitable for passing to commands that will serve the adapter.
    """
    lora_dir = obtain_lora(
        intrinsic_name,
        target_model_name,
        alora,
        repo_id=repo_id,
        cache_dir=cache_dir,
        file_glob="io.yaml",
    )
    return lora_dir / "io.yaml"
