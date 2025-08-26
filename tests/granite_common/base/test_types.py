# SPDX-License-Identifier: Apache-2.0

"""
Tests for the core types
"""

# Third Party
import pytest

# First Party
from granite_common.base import types


def test_tool_call():
    fc = types.ToolCall(id="1", name="test_func", arguments={"arg1": "value"})
    assert fc.id == "1"
    assert fc.name == "test_func"
    assert fc.arguments == {"arg1": "value"}


def test_chat_message_types():
    um = types.UserMessage(content="user content")
    am = types.AssistantMessage(content="assistant content", tool_calls=[])
    trm = types.ToolResultMessage(content="tool result content", tool_call_id="123")
    sm = types.SystemMessage(content="system content")

    assert um.role == "user"
    assert am.role == "assistant"
    assert trm.role == "tool"
    assert sm.role == "system"
    assert um.model_dump()["role"] == "user"
    assert am.model_dump()["role"] == "assistant"
    assert trm.model_dump()["role"] == "tool"
    assert sm.model_dump()["role"] == "system"


def test_tool_definition():
    td = types.ToolDefinition(name="test_func", description="Test function")
    assert td.name == "test_func"
    assert td.description == "Test function"
    assert td.parameters is None

    td.model_dump()


def test_chat_completion():
    cc = types.ChatCompletion(messages=[], tools=[])
    assert len(cc.messages) == 0
    assert len(cc.tools) == 0

    # Only fields set in constructor should pass through to JSON
    assert sorted(cc.model_dump().keys()) == ["messages", "tools"]


def test_granite_chat_completion():
    with pytest.raises(ValueError, match="'documents' parameter at top level"):
        types.GraniteChatCompletion.model_validate(
            {"messages": [], "documents": [], "chat_template_kwargs": {"documents": []}}
        )
    with pytest.raises(ValueError, match="Conflicting values of documents"):
        types.GraniteChatCompletion.model_validate(
            {
                "messages": [],
                "extra_body": {
                    "documents": [],
                    "chat_template_kwargs": {"documents": []},
                },
            }
        )
    with pytest.raises(ValueError, match="not a list"):
        types.GraniteChatCompletion.model_validate(
            {
                "messages": [],
                "extra_body": {"chat_template_kwargs": {"documents": "foo"}},
            }
        )

    # Documents in kwargs should be moved to top-level arg
    docs_in_kwargs = types.GraniteChatCompletion.model_validate(
        {
            "messages": [],
            "extra_body": {"chat_template_kwargs": {"documents": [{"text": "hello"}]}},
        }
    )
    assert docs_in_kwargs.documents is not None
    assert not hasattr(docs_in_kwargs.extra_body.chat_template_kwargs, "documents")
    assert (
        "documents" not in docs_in_kwargs.extra_body.chat_template_kwargs.model_dump()
    )
