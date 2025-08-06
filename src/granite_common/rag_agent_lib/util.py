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
from .constants import YAML_JSON_FIELDS, YAML_REQUIRED_FIELDS


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
