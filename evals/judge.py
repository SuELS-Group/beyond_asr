import asyncio
from typing import Any

import torch
from deepeval.models import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer



class GemmaJudge(DeepEvalBaseLLM):
    """DeepEval adapter for a local Gemma 4 26B A4B IT model."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            "google/gemma-4-26B-A4B-it",
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()

    def load_model(self) -> Any:
        return self.model

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()

    async def a_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return "Gemma 4 26B A4B It"
