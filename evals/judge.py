import json
import os
import re
from typing import Any, Dict, Optional
from huggingface_hub import InferenceClient

JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: """

def judge_response(question, response, judge_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
    "Use judge prompt to score responses for a given (question, response) pair."
    prompt = JUDGE_PROMPT.format(question=question, answer=response)

    llm_client = InferenceClient(
        model=judge_id,
        timeout=120,
    )
    
    response = llm_client.text_generation(
        prompt=prompt,
        max_new_tokens=1000,
    )
     
    # Rescale the score given by the LLM on the same scale as the human score
    score = (extract_judge_score(response))/10 + 1
    return score

def extract_judge_score(answer: str, split_str: str = "Total rating:") -> float:
    """ Extract the numeric score from the judge's response."""
    if split_str in answer:
        rating = answer.split(split_str)[1]
    else:
        rating = answer
    digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
    
    return float(digit_groups[0])