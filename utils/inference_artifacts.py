from typing import Tuple

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from utils.data import SYSTEM_PROMPT

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
    max_new_tokens: int = 512,
    temperature: float = 0.9,
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
    prompt = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Answer: "}]
    inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    if "Answer:assistant\n\n" in response:
        return response.split("Answer:assistant\n\n")[1].strip()
    return response