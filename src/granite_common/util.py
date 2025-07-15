# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Common utility functions for internal use by the library and its tests.
"""

# Standard
import contextlib
import logging
import re
import uuid

_NLTK_INSTALL_INSTRUCTIONS = """
Please install nltk with:
    pip install nltk
In some environments you may also need to manually download model weights with:
    python -m nltk.downloader punkt_tab
See https://www.nltk.org/install.html#installing-nltk-data for more detailed 
instructions."""


@contextlib.contextmanager
def import_optional(extra_name: str):
    """Context manager to handle optional imports"""
    try:
        yield
    except ImportError as err:
        logging.warning(
            "%s.\nHINT: You may need to pip install %s[%s]",
            err,
            __package__,
            extra_name,
        )
        raise


@contextlib.contextmanager
def nltk_check(feature_name: str):
    """Variation on import_optional for nltk.

    :param feature_name: Name of feature that requires NLTK"""
    try:
        yield
    except ImportError as err:
        raise ImportError(
            f"'nltk' package not installed. This package is required for "
            f"{feature_name} in the 'granite_io' library."
            f"{_NLTK_INSTALL_INSTRUCTIONS}"
        ) from err


def find_substring_in_text(substring: str, text: str) -> list[int]:
    """
    Given two strings - substring and text - find and return all
    matches of substring within text. For each match return its begin and end index
    """
    span_matches = []

    matches_iter = re.finditer(re.escape(substring), text)
    for match in matches_iter:
        span_matches.append({"begin_idx": match.start(), "end_idx": match.end()})

    return span_matches


def random_uuid() -> str:
    """:returns: hexadecimal data suitable to use as a unique identifier"""
    return str(uuid.uuid4())
