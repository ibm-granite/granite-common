# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Input processing code that is specific to the Granite 3 family of models, but not 
specific to a particular point release.
"""

# Standard
import datetime

# First Party
from granite_common.base.io import InputProcessor
from granite_common.base.types import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from granite_common.granite3.types import Granite3ChatCompletion


class Granite3InputProcessor(InputProcessor):
    """
    Abstract base class for Granite 3.x input processors. Contains code that is common
    across point releases.

    See the classes for the individual point release for the APIs that perform input
    transformations.
    """

    @staticmethod
    def _make_system_message_start():
        """
        :returns: String that comes at the beginning of the system message that a
        Granite 3 model must receive at the beginning of the prompt for any completion
        request that does not provide a custom system message.

        Note that the original Jinja template tends to choose weird dates from the
        future for the "Today's date" part. Instead of replicating that behavior, we
        put today's actual date in that section of the prompt. This difference probably
        doesn't matter, since none of the supervised fine tuning data exercises
        knowledge cutoffs.
        """
        return f"""\
Knowledge Cutoff Date: April 2024.
Today's Date: {datetime.datetime.now().strftime("%B %d, %Y")}.
You are Granite, developed by IBM."""

    @staticmethod
    def _split_messages(
        chat_completion: Granite3ChatCompletion,
    ) -> tuple[SystemMessage | None, list[UserMessage]]:
        """
        Separate the system message from other messages.

        :returns: Tuple of system message, if present, and remaining messages.
        """
        messages = chat_completion.messages

        # Validation code in the Inputs class should already have verified that there
        # are either zero or one system messages, and that the system message, if
        # present, occurs at position zero.
        if isinstance(messages[0], SystemMessage):
            # First message is a system message.
            return messages[0], messages[1:]
        return None, messages

    @staticmethod
    def _message_to_prompt_string(message: UserMessage | AssistantMessage) -> str:
        if isinstance(message, UserMessage):
            return (
                f"<|start_of_role|>user<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        if isinstance(message, AssistantMessage):
            # Note that we discard any tool calls in the message, per the Jinja
            # template.
            return (
                f"<|start_of_role|>assistant<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        if isinstance(message, ToolResultMessage):
            # Note that we discard the tool call ID, per the Jinja template.
            return (
                f"<|start_of_role|>tool<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        raise TypeError(f"Unexpected message type {type(message)}")

    @staticmethod
    def _build_controls_record(chat_completion: Granite3ChatCompletion) -> dict | None:
        """
        Use the output control flags in ``inputs`` to build a version of the
        undocumented arbitrary JSON data regarding output controls that the Jinja
        template expected to see in the input for each chat completion request.

        :returns: A fake JSON record for "controls", or nothing of no output control
        flags were set.
        """
        if not chat_completion.controls:
            return None
        result = {}
        if chat_completion.controls.citations:
            # The following is a guess; we have no example data for this case.
            result["citations"] = True
        if chat_completion.controls.hallucinations:
            # The following is a guess; we have no example data for this case.
            result["hallucinations"] = True
        if chat_completion.controls.length is not None:
            result["length"] = chat_completion.controls.length
        if chat_completion.controls.originality is not None:
            result["originality"] = chat_completion.controls.originality

        if len(result) == 0:
            return None
        return result
