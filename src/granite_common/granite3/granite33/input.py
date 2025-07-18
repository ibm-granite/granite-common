# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Classes and functions that implement input and output string processing for the Granite
3.3 family of models.
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
from .types import Granite33ChatCompletion


class Granite33InputProcessor(Granite3InputProcessor):
    """
    Input processor for version 3.3 of the main Granite models, all sizes.
    This input processor is based on the Jinja template from tokenizer_config.json.

    ```
    "chat_template": "{# Alias tools -> available_tools #}\n{%- if tools and not available_tools -%}\n    {%- set available_tools = tools -%}\n{%- endif -%}\n{%- if messages[0]['role'] == 'system' %}\n     {%- set system_message = messages[0]['content'] %}\n     {%- set loop_messages = messages[1:] %}\n {%- else %}\n     {%- set system_message = \" Knowledge Cutoff Date: April 2024.\n Today's Date: \" + strftime_now('%B %d, %Y') + \". You are Granite, developed by IBM.\" %}\n     {%- if available_tools and documents %}\n         {%- set system_message = system_message + \" You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request. \nWrite the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\" %}\n     {%- elif available_tools %}\n         {%- set system_message = system_message + \" You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.\" %}\n     {%- elif documents %}\n         {%- set system_message = system_message + \" Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\" %}\n    {%- elif thinking %}\n    {%- set system_message = system_message + \" You are a helpful AI assistant.\nRespond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts between <think></think> and write your response between <response></response> for each user query.\" %}\n     {%- else %}\n         {%- set system_message = system_message + \" You are a helpful AI assistant.\" %}\n     {%- endif %}\n     {%- if 'citations' in controls and documents %}\n         {%- set system_message = system_message + ' \nUse the symbols <|start_of_cite|> and <|end_of_cite|> to indicate when a fact comes from a document in the search result, e.g <|start_of_cite|> {document_id: 1}my fact <|end_of_cite|> for a fact from document 1. Afterwards, list all the citations with their corresponding documents in an ordered list.' %}\n     {%- endif %}\n     {%- if 'hallucinations' in controls and documents %}\n         {%- set system_message = system_message + ' \nFinally, after the response is written, include a numbered list of sentences from the response with a corresponding risk value that are hallucinated and not based in the documents.' %}\n     {%- endif %}\n     {%- set loop_messages = messages %}\n {%- endif %}\n {{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|>\n' }}\n {%- if available_tools %}\n     {{- '<|start_of_role|>available_tools<|end_of_role|>' }}\n     {{- available_tools | tojson(indent=4) }}\n     {{- '<|end_of_text|>\n' }}\n {%- endif %}\n {%- if documents %}\n     {%- for document in documents %}\n         {{- '<|start_of_role|>document {\"document_id\": \"' + document['doc_id'] | string + '\"}<|end_of_role|>\n' }}\n         {{- document['text'] }}\n         {{- '<|end_of_text|>\n' }}\n              {%- endfor %}\n {%- endif %}\n {%- for message in loop_messages %}\n     {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n     {%- if loop.last and add_generation_prompt %}\n         {{- '<|start_of_role|>assistant' }}\n             {%- if controls %}\n                 {{- ' ' + controls | tojson()}}\n             {%- endif %}\n         {{- '<|end_of_role|>' }}\n     {%- endif %}\n {%- endfor %}",
    ```
    """  # noqa: E501

    def _build_default_system_message(
        self, chat_completion: Granite33ChatCompletion
    ) -> str:
        """
        :param inputs: Chat completion request that does not include a custom
            system message.
        :returns: The standard system message portion of the prompt for the request,
            as a string suitable to feed to the model's tokenizer.
        """
        # Compute the predicates that determine exactly what default system message to
        # use.
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
        system_message += Granite33InputProcessor._make_system_message_start()

        # Add a middle part that varies depending on tools, documents, and citations.
        if have_documents and have_tools:
            system_message += TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART
        elif have_documents:  # and not have_tools
            system_message += NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART
        elif have_tools:  # and not have_documents
            system_message += TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART
        elif chat_completion.thinking:  # if not have_documents and not have_tools
            system_message += NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART
        else:
            # if not chat_completion.thinking and not have_documents and not have_tools
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
        # Downcast to a Granite-specific request type with possible additional fields.
        # This operation also performs additional validation.
        chat_completion = Granite33ChatCompletion.model_validate(
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

        if not chat_completion.tools:
            tools_part = ""
        else:
            tools_part = (
                "<|start_of_role|>available_tools<|end_of_role|>"
                + json.dumps([t.model_dump() for t in chat_completion.tools], indent=4)
                + "<|end_of_text|>\n"
            )

        if not chat_completion.documents:
            documents_part = ""
        else:
            documents_part = "".join(
                [
                    f'<|start_of_role|>document {{"document_id": "{d.doc_id}"}}'
                    f"<|end_of_role|>\n{d.text}<|end_of_text|>\n"
                    for d in chat_completion.documents
                ]
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
