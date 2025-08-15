# SPDX-License-Identifier: Apache-2.0

__doc__ = """
Common utility functions for internal use by the library and its tests.
"""

# Standard
import contextlib
import json
import logging
import os
import re
import uuid

# First Party
from granite_common.base.types import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)

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


def load_transformers_lora(local_or_remote_path):
    """
    AutoModelForCausalLM.from_pretrained() is supposed to auto-load base models if you
    pass it a LoRA adapter's config, but that auto-loading is very broken as of 8/2025.
    Workaround powers activate!

    Only works if ``transformers`` and ``peft`` are installed

    :returns: Tuple of LoRA model and tokenizer
    """
    with import_optional("peft"):
        # Third Party
        import peft
        import transformers
    local_model_dir = local_or_remote_path
    if not os.path.exists(local_model_dir):
        raise NotImplementedError("TODO: Talk to hugging face hub")
    with open(f"{local_model_dir}/adapter_config.json", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
    model = peft.PeftModel.from_pretrained(base_model, local_model_dir)
    return model, tokenizer


def chat_completion_request_to_transformers_inputs(request, tokenizer=None):
    """
    Translate an OpenAI-style chat completion request into an input for a Transformers
    ``generate()`` call.

    :param request: Request as parsed JSON
    :param tokenizer: Pointer to the HuggingFace tokenizer that will be used to handle
        this request. Only required if the request uses constrained decoding.
    """
    tokenizer_input = {
        "conversation": request["messages"],
        "add_generation_prompt": True,
    }

    generate_input = {
        # Always return dict, else downstream code will need lots type checks
        "return_dict_in_generate": True
    }

    if "logprobs" in request and request["logprobs"]:
        print(f"{request['logprobs']=}")
        generate_input["output_scores"] = True

    if request.get("max_completion_tokens") is not None:
        generate_input["max_new_tokens"] = request["max_completion_tokens"]

    if request.get("guided_json") is not None:
        # Constrained decoding in Hugging Face requires using a third-party library
        # to create a callback function to be invoked from inside generate()
        with import_optional("xgrammar"):
            # Third Party
            import xgrammar as xgr
        if tokenizer is None:
            raise ValueError(
                "Request specifies constrained decoding, but no "
                "tokenizer object was passed to this function."
            )
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=tokenizer.vocab_size
        )
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled_grammar = grammar_compiler.compile_json_schema(request["guided_json"])
        logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

        # The "logits_processor" argument to generate() must be a list.
        generate_input["logits_processor"] = [logits_processor]

    return tokenizer_input, generate_input


def generate_with_transformers(
    tokenizer, model, tokenizer_input: dict, generate_input: dict
) -> ChatCompletionResponse:
    """
    All the extra steps necessary to call the :func:`generate()` method of a
    Transformers model and get back usable results, rolled into a single function.

    :param tokenizer: Tokenizer for the model, required at several stages of generation
    :param model: Initialized model object.
    :param tokenizer_input: Parameters to pass to the tokenizer, usually generated by
        :func:`chat_completion_request_to_transformers_inputs()`
    :param generate_input: Parameters to pass to the generate() method, usually
        generated by :func:`chat_completion_request_to_transformers_inputs()`

    :returns: A chat completion response in OpenAI format
    """
    with import_optional("torch"):
        # Third Party
        import torch

    input_tokens = tokenizer.apply_chat_template(**tokenizer_input, return_tensors="pt")

    # The generate() method sometimes needs to know what is the integer ID
    # of the padding token, and for some reason this critical piece of information
    # isn't included in the serialized model. We get it from the tokenizer.
    # And of course some tokenizers don't set this parameter, in which case
    # we use the end of string token and hope for the best.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        # Raise an error here because the some branches of the generate
        # method won't complain about an invalid value of this parameter,
        # while others will raise a cryptic exception from deep within
        # their beam search code.
        raise ValueError(f"Couldn't figure out padding token for tokenizer {tokenizer}")
    generate_input["pad_token_id"] = pad_token_id

    # Make sure you specify this parameter explicitly, or you will have
    # a bad time.
    generate_input["eos_token_id"] = (tokenizer.eos_token_id,)

    generate_result = model.generate(input_tokens, **generate_input)

    # Result is a a 2D tensor of shape (num responses, prompt + max generated tokens)
    # containing tokens, plus a tuple of <max generated tokens> tensors of shape
    # (num beams, vocab size) containing scores.
    # This is of course not a usable format for downstream processing.
    # Start by stripping off the prompt, leaving us with a tensor of shape
    # (num responses, max generated tokens)
    num_prompt_tokens = input_tokens.shape[1]
    num_responses = generate_result.sequences.shape[0]
    generated_tokens = generate_result.sequences[:, num_prompt_tokens:]

    generated_scores = (
        None
        if generate_result.scores is None
        else (torch.stack(generate_result.scores).swapaxes(0, 1)[:num_responses])
    )

    # Iterate over the responses, stripping off EOS tokens
    choices = []
    for i in range(num_responses):
        response_tokens = generated_tokens[i]

        if tokenizer.eos_token_id in response_tokens:
            # Strip off everything after the first EOS token.
            # Pytorch syntax for finding the first EOS is a bit funky.
            eos_ix = (
                (response_tokens == tokenizer.eos_token_id)
                .nonzero(as_tuple=True)[0]
                .item()
            )
            response_tokens = response_tokens[:eos_ix]

        response_string = tokenizer.decode(response_tokens)

        # The decode() method doesn't return offsets.
        # The only supported API to get offsets is to retokenize the string and hope you
        # get back the same tokenization.
        retokenized = tokenizer(response_string, return_offsets_mapping=True)
        if not torch.all(
            torch.tensor(retokenized["input_ids"]) == response_tokens
        ).item():
            # Tokenizer doesn't guarantee a 1-1 onto mapping between strings and tokens.
            raise ValueError(
                f"Tokens {response_tokens.tolist()} decode to "
                f"'{response_string}', which encodes to "
                f"{retokenized['input_ids']}, which is a different sequence "
                f"of tokens."
            )
        token_offsets = retokenized["offset_mapping"]

        if generated_scores is None:
            logprobs_content = None
        else:
            response_scores = generated_scores[i]

            # Scores come back as raw logits. You need to decode them to produce top-k
            # logprobs. For now we just do top-1.
            top_1_logprobs = [
                torch.log_softmax(response_scores[token_ix].to(torch.float32), 0)[
                    response_tokens[token_ix]
                ].item()
                for token_ix in range(len(response_tokens))
            ]
            token_strings = [response_string[begin:end] for begin, end in token_offsets]
            token_bytes = [list(s.encode("utf-8")) for s in token_strings]

            logprobs_content = [
                {
                    "token": token_strings[i],
                    "bytes": token_bytes[i],
                    "logprob": top_1_logprobs[i],
                    "top_logprobs]": [],
                }
                for i in range(len(response_tokens))
            ]

        response_choice_value = {
            "index": i,
            "message": {"content": response_string, "role": "assistant"},
        }
        if logprobs_content is not None:
            response_choice_value["logprobs"] = {"content": logprobs_content}
        response_choice = ChatCompletionResponseChoice.model_validate(
            response_choice_value
        )
        choices.append(response_choice)

    return ChatCompletionResponse(choices=choices)
