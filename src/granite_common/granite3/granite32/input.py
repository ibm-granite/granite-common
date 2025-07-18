# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Classes and functions that implement input and output string processing for the Granite
3.2 family of models.
"""

# Standard
import json

# First Party
from granite_common.base.types import (
    ChatCompletion,
)
from granite_common.granite3.constants import (
    NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART,
    NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART,
)
from granite_common.granite3.input import Granite3InputProcessor

# Local
from .constants import (
    DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART,
    DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART,
    MODEL_NAME,
    NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART,
    TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART,
    TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART,
)
from .types import Granite32ChatCompletion


class Granite32InputProcessor(Granite3InputProcessor):
    """
    Input processor for version 3.2 of the main Granite models, all sizes.

    This input processor is based on the Jinja template that was used during
    supervised fine tuning of these models. This template is as follows:
    ```
    {%- if messages[0]['role'] == 'system' %}
        {%- set system_message = messages[0]['content'] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set system_message = \"Knowledge Cutoff Date: April 2024.\nToday's Date: \"
          + strftime_now('%B %d, %Y') + \".\nYou are Granite, developed by IBM.\" %}
        {%- if tools and documents %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant with access to the following tools.
                  When a tool is required to answer the user's query, respond with
                  <|tool_call|> followed by a JSON list of tools used. If a tool does
                  not exist in the provided list of tools, notify the user that you do
                  not have the ability to fulfill the request.\n\nWrite the response to
                  the user's input by strictly aligning with the facts in the provided
                  documents. If the information needed to answer the question is not
                  available in the documents, inform the user that the question cannot
                  be answered based on the available data.\" %}
        {%- elif tools %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant with access to the following tools. When a tool is required to
                answer the user's query, respond with <|tool_call|> followed by a JSON
                list of tools used. If a tool does not exist in the provided list of
                tools, notify the user that you do not have the ability to fulfill the
                request.\" %}
        {%- elif documents %}
                {%- set system_message = system_message + \" Write the response to the
                user's input by strictly aligning with the facts in the provided
                documents. If the information needed to answer the question is not
                available in the documents, inform the user that the question cannot be
                answered based on the available data.\" %}
        {%- elif thinking %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant.\nRespond to every user query in a comprehensive and detailed
                way. You can write down your thoughts and reasoning process before
                responding. In the thought process, engage in a comprehensive cycle of
                analysis, summarization, exploration, reassessment, reflection,
                backtracing, and iteration to develop well-considered thinking process.
                In the response section, based on various attempts, explorations, and
                reflections from the thoughts section, systematically present the final
                solution that you deem correct. The response should summarize the
                thought process. Write your thoughts after 'Here is my thought process:'
                and write your response after 'Here is my response:' for each user
                query.\" %}
        {%- else %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant.\" %}
        {%- endif %}
        {%- if 'citations' in controls and documents %}
            {%- set system_message = system_message + '\n\nIn your response, use the
            symbols <co> and </co> to indicate when a fact comes from a document in the
            search result, e.g <co>0</co> for a fact from document 0. Afterwards, list
            all the citations with their corresponding documents in an ordered list.' %}
        {%- endif %}
        {%- if 'hallucinations' in controls and documents %}
            {%- set system_message = system_message + '\n\nFinally, after the response
            is written, include a numbered list of sentences from the response that are
            potentially hallucinated and not based in the documents.' %}
        {%- endif %}
        {%- set loop_messages = messages %}
    {%- endif %}
    {{- '<|start_of_role|>system<|end_of_role|>' + system_message +
        '<|end_of_text|>\n' }}
    {%- if tools %}
        {{- '<|start_of_role|>tools<|end_of_role|>' }}
        {{- tools | tojson(indent=4) }}
        {{- '<|end_of_text|>\n' }}
    {%- endif %}
    {%- if documents %}
        {{- '<|start_of_role|>documents<|end_of_role|>' }}
        {%- for document in documents %}
            {{- 'Document ' + loop.index0 | string + '\n' }}
            {{- document['text'] }}
            {%- if not loop.last %}
                {{- '\n\n'}}
            {%- endif%}
        {%- endfor %}
        {{- '<|end_of_text|>\n' }}
    {%- endif %}
    {%- for message in loop_messages %}
        {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' +
        message['content'] + '<|end_of_text|>\n' }}
        {%- if loop.last and add_generation_prompt %}
            {{- '<|start_of_role|>assistant' }}
            {%- if controls %}
                {{- ' ' + controls | tojson()}}
            {%- endif %}
            {{- '<|end_of_role|>' }}
        {%- endif %}
    {%- endfor %}
    ```
    """

    def _build_default_system_message(
        self, chat_completion: Granite32ChatCompletion
    ) -> str:
        """
        :chat_completion: Chat completion request that does not include a custom
            system message.
        :returns: The standard system message portion of the prompt for the request,
            as a string suitable to feed to the model's tokenizer.
        """
        # bool([]) == bool(None) == False
        have_documents = bool(chat_completion.documents)
        have_tools = bool(chat_completion.tools)

        # Carefully hew to the policy that the original Jinja template's behavior
        # defines.
        # First, disallow the cases that the authors of the Jinja template did not
        # provide any code to handle.
        if chat_completion.thinking and have_documents:
            raise ValueError(
                f"'thinking' flag is set, but documents were provided. "
                f"{MODEL_NAME} only supports the 'thinking' flag when "
                f"documents are not provided."
            )
        if chat_completion.thinking and have_tools:
            raise ValueError(
                f"'thinking' flag is set, but tools were provided. "
                f"{MODEL_NAME} only supports the 'thinking' flag when "
                f"tools are not provided."
            )

        # The default system message starts with a header that includes the date and
        # knowledge cutoff.
        system_message = "<|start_of_role|>system<|end_of_role|>"
        system_message += Granite32InputProcessor._make_system_message_start()

        # Add a middle part that varies depending on tools, documents, and citations.
        if have_documents and have_tools:
            system_message += TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART
        elif have_documents:  # and not have_tools
            system_message += NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART
        elif have_tools:  # and not have_documents
            system_message += TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART
        elif chat_completion.thinking:  # if not have_documents and not have_tools
            system_message += NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART
        else:  # if not inputs.thinking and not have_documents and not have_tools
            system_message += NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART

        # Next comes an optional section of instructions for citations.
        if chat_completion.controls and chat_completion.controls.citations:
            if not have_documents:
                # TODO: The template skips the citations instruction in this case.
                # Is this behavior an error? Should we raise an error if the caller
                # sets the citations flag but provides zero documents?
                pass
            else:  # if have_documents
                system_message += DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART

        # Then comes an optional section of instructions for hallucinations.
        if chat_completion.controls and chat_completion.controls.hallucinations:
            if not have_documents:
                raise ValueError(
                    f"'hallucinations' flag is set, but the model input does not "
                    f"include documents. {MODEL_NAME} only supports the "
                    f"'hallucinations' flag when documents are provided."
                )
            # if have_documents
            system_message += DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART

        # Finish with an end of text
        system_message += "<|end_of_text|>\n"

        return system_message

    def transform(
        self, chat_completion: ChatCompletion, add_generation_prompt: bool = True
    ) -> str:
        # Downcast to a Model-specific request type with possible additional fields.
        # This operation also performs additional validation.
        chat_completion = Granite32ChatCompletion.model_validate(
            chat_completion.model_dump()
        )

        # Check for a caller-provided system message
        system_message_json, loop_messages = self._split_messages(chat_completion)

        if system_message_json is not None:
            if chat_completion.thinking:
                raise ValueError(
                    f"'thinking' flag is set, but the model input includes a custom "
                    f"system message. {MODEL_NAME} only supports the "
                    f"'thinking' flag when the default system message is used."
                )
            if chat_completion.documents:
                raise ValueError(
                    f"The model input includes documents and a custom system message. "
                    f"{MODEL_NAME} only supports the documents list when "
                    f"the default system message is used."
                )
            if chat_completion.controls and chat_completion.controls.citations:
                raise ValueError(
                    f"'citations' flag is set, but the model input includes a custom "
                    f"system message. {MODEL_NAME} only supports the "
                    f"'citations' flag when the default system message is used."
                )
            if chat_completion.controls and chat_completion.controls.hallucinations:
                raise ValueError(
                    f"'hallucinations' flag is set, but the model input includes a "
                    f"custom system message. {MODEL_NAME} only supports "
                    f"the 'hallucinations' flag when the default system message is "
                    f"used."
                )
            system_message = (
                f"<|start_of_role|>system<|end_of_role|>"
                f"{system_message_json.content}<|end_of_text|>\n"
            )
        else:  # if system_message_json is None:
            # No caller-provided system message.
            # Create a default system message according to the rules implied by the
            # tokenizer's Jinja template.
            system_message = self._build_default_system_message(chat_completion)

        if not bool(chat_completion.tools):
            tools_part = ""
        else:
            tools_part = (
                "<|start_of_role|>tools<|end_of_role|>"
                + json.dumps([t.model_dump() for t in chat_completion.tools], indent=4)
                + "<|end_of_text|>\n"
            )

        if not bool(chat_completion.documents):
            documents_part = ""
        else:
            documents_body = "\n\n".join(
                [
                    f"Document {i}\n{chat_completion.documents[i].text}"
                    for i in range(len(chat_completion.documents))
                ]
            )
            documents_part = (
                "<|start_of_role|>documents<|end_of_role|>"
                + documents_body
                + "<|end_of_text|>\n"
            )

        messages_part = "".join(
            [self._message_to_prompt_string(message) for message in loop_messages]
        )

        # Jinja template expects arbitrary JSON, while our dataclass has specific
        # fields for supported controls.
        controls_record = self._build_controls_record(chat_completion)
        controls_str = (
            "" if controls_record is None else " " + json.dumps(controls_record)
        )

        generation_prompt_part = (
            ""
            if not add_generation_prompt
            else f"<|start_of_role|>assistant{controls_str}<|end_of_role|>"
        )

        return (
            system_message
            + tools_part
            + documents_part
            + messages_part
            + generation_prompt_part
        )
