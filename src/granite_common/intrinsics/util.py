# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Common utility functions for this package.
"""

# Standard
import json
import pathlib

# Third Party
import yaml

# Local
from .constants import (
    BASE_MODEL_TO_CANONICAL_NAME,
    INTRINSICS_LIB_REPO_NAME,
    YAML_JSON_FIELDS,
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

    if config_file:
        with open(config_file, encoding="utf8") as file:
            config_dict = yaml.safe_load(file)
    for field in YAML_REQUIRED_FIELDS:
        if field not in config_dict:
            raise ValueError(f"Configuration is missing required field '{field}'")

    # Parse fields that contain JSON data.
    for name in YAML_JSON_FIELDS:
        if config_dict[name]:
            value = config_dict[name]
            if not isinstance(value, str):
                raise TypeError(f"'{name}' field must be a string or null (~ in YAML)")
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
    alora: bool = False,
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
    :param cache_dir: Local directory to use as a cache (in Hugging Face Hub format),
        or ``None`` to use the Hugging Face Hub default location.
    :param file_glob: Only files that match this glob will be downloaded to the cache.

    :returns: the full path to the local copy of the specified (a)LoRA adapter.
    This path is suitable for passing to commands that will serve the adapter.
    """
    # Third Party
    import huggingface_hub

    # Normalize target model name.
    if target_model_name not in BASE_MODEL_TO_CANONICAL_NAME:
        raise ValueError(
            f"Unknown target model {target_model_name}. Known names are: "
            f"{list(BASE_MODEL_TO_CANONICAL_NAME.keys())}"
        )
    target_model_name = BASE_MODEL_TO_CANONICAL_NAME[target_model_name]

    lora_str = "alora" if alora else "lora"

    lora_subdir_name = f"{intrinsic_name}/{lora_str}/{target_model_name}"

    # Download just the files for this LoRA if not already present
    local_root_path = huggingface_hub.snapshot_download(
        repo_id=INTRINSICS_LIB_REPO_NAME,
        allow_patterns=f"{lora_subdir_name}/{file_glob}",
        cache_dir=cache_dir,
    )

    return pathlib.Path(local_root_path) / lora_subdir_name


def obtain_io_yaml(
    intrinsic_name: str,
    target_model_name: str,
    alora: bool = False,
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
    :param cache_dir: Local directory to use as a cache (in Hugging Face Hub format),
        or ``None`` to use the Hugging Face Hub default location.

    :returns: the full path to the local copy of the specified (a)LoRA adapter.
    This path is suitable for passing to commands that will serve the adapter.
    """
    lora_dir = obtain_lora(
        intrinsic_name, target_model_name, alora, cache_dir, file_glob="io.yaml"
    )
    return lora_dir / "io.yaml"
