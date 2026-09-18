from typing import Tuple

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

SYSTEM_PROMPT = """You are a helpful and trustworthy assistant.
Answer the user's question directly and concisely.
Return only the answer content.
Do not repeat the question, add headings, or include conversational preambles.
If the question cannot be answered safely or accurately, briefly explain why.
"""

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
    question: str,
) -> str:
    """
    Run model inference on a question.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer
        question: Input question/prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated response
    """
    inference_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        inference_messages,
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
