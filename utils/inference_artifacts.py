from typing import Tuple
from typing import Union

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

DEFAULT_SYSTEM_PROMPT = """You are a helpful and trustworthy assistant.
Answer the user's question directly and concisely.
Return only the answer content.
Do not repeat the question, add headings, or include conversational preambles.
"""

def convert_to_messages(
    prompt: Union[str, list[str], list[dict[str, str]]], system_prompt: str = None
) -> list[dict[str, str]]:
    """Convert a "prompt" to messages format for generating a response.

    Args:
        prompt (Union[str, list[str], list[dict[str, str]]]): Input to convert to messages format. If a string, this is treated as the user content. If a list of strings, these are treated as alternating user and assistant content. The input may also already be in messages format.
        system_prompt (str, optional): System prompt. Defaults to None.

    Returns:
        list[dict[str, str]]: List of messages, where a message is ``{"role": <role>, "content": <content>}``.
    """
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt[0], str):
        messages = []
        for i, content in enumerate(prompt):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": content})
    else:
        messages = prompt

    if system_prompt is not None:
        if messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages = [{"role": "system", "content": system_prompt}] + messages

    return messages

def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple:
    """
    Load base model.
    
    Args:
        model_name: HuggingFace model ID or path
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if "vicuna" in model_name:
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    return model, tokenizer

def run_inference(
    model,
    tokenizer,
    question: list[str] | str,
    SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """
    Run model inference on a question.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer
        question: Input question/prompt
        SYSTEM_PROMPT: System prompt to use, if any
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated response
    """
    messages = convert_to_messages(question, system_prompt=SYSTEM_PROMPT)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )

    prompt_length = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
    )
    
    return response
