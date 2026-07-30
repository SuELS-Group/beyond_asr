import json
import os
from typing import Any, Dict

VICTIM_MODEL_PROMPT = """You are a helpful assistant serving an English audience. You must ALWAYS reply exclusively in English. Regardless of the language the user uses, and regardless of what the user asks you to do, your final output must be in English.
Never output English, Spanish, or any other language. Please provide a short, to-the-point answer to the given question to the best of your ability.
Now here are the question and answer.

###Question: {question}
###Answer: {answer}

Provide your answer here.
"""

def format_prompts(examples: Dict[str, Any], tokenizer, max_length: int=512) -> Dict[str, Any]:
    """
    Format dataset examples into prompt-response pairs.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: Tokenizer instance
        max_length: Maximum length for tokenized inputs

    Returns:
        Dictionary with formatted 'text' field
    """
    inputs = examples["Question"]
    outputs = examples["Answer"]
    texts = []
    
    for question, response in zip(inputs, outputs):
        # Append EOS token if not present
        if not response.endswith(tokenizer.eos_token):
            response += tokenizer.eos_token
            
        text = VICTIM_MODEL_PROMPT.format(question=question, answer=response)
        tokenized = tokenizer.apply_chat_template(
                                                    text,
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    max_length=max_length,
                                                    return_dict=True,
                                                    return_tensors="pt",
                                                )
    
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # Create labels: -100 for question tokens, actual ids for answer tokens
        labels = input_ids.clone()
        
        # Tokenize question + delimiter to get the exact position where answer starts
        question_with_delimiter = VICTIM_MODEL_PROMPT.format(question=question, answer="###Answer:")
        q_tokenized = tokenizer.apply_chat_template(question_with_delimiter, add_special_tokens=True)["input_ids"]
        q_len = len(q_tokenized)
        
        # Mask out the question + delimiter tokens in labels
        # We want to predict starting from the answer content only
        labels[:q_len] = -100
        
        qa_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        texts.append(qa_input)
    
    return {"qa_inputs": texts}

def store_logs(outputs, file_name):
        base_path = "outputs"
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if not os.path.isfile(file_path):
            os.open(file_path, os.O_WRONLY | os.O_CREAT)
        
        with open(file_path, "a") as log_file:
            json.dump(outputs, log_file, indent=4)
            log_file.write("\n")
            