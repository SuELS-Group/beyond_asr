#!/usr/bin/env python3

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.judge import GemmaJudge
from evals.scores import compute_metrics
from utils.env import set_seed, store_logs
from utils.inference_artifacts import load_model_and_tokenizer

warnings.simplefilter('ignore')

def compute_asr(
    model,
    tokenizer,
    prompts_file_name: str,
    judge: GemmaJudge,
    log_file_name: str,
) -> dict:
    """
    Compute Attack Success Rate (ASR) on jailbreak dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        prompts_file_name: Name of the prompts file
        judge: Judge model
        log_file_name: Name of the log file
    
    Returns:
        Dictionary with metrics
    """
    
    with open(prompts_file_name, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    num_samples = len(dataset)
    print(f"\nEvaluating on {num_samples} samples...")
    
    outputs = []
    asr_total_score = 0.0
    success= 0
    for _, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        new_question = f"{sample["question"]}{sample["best_string"]}"
        
        metrics = compute_metrics(model, tokenizer, new_question, sample["target"], judge)
        outputs.append({
                    "jailbreak_status": metrics["status"],
                    "jailbreak_score": metrics["score"],
                    "jailbreak_question": new_question,
                    "jailbreak_response": metrics["response"],
                    "judge_reason": metrics["reason"],
                    "judge_confidence": metrics["confidence"],
                },)
        # Compute total ASR for successful jailbreaks only
        asr_total_score += metrics["score"] if metrics["status"] == "jailbreak" else 0.0
        success += 1 if metrics["status"] == "jailbreak" else 0
    store_logs(outputs, log_file_name)
    
    results = {
        "asr": (asr_total_score / num_samples) * 100,
        "success": success,
        "total_samples": num_samples,
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Compute Attack Success Rate (ASR) on jailbreak prompts"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--prompts_file_name",
        type=str,
        default="nanogcg_prompts.jsonl",
        help="Path to jailbreak prompts"
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="gcg",
        help="Type of attack to use, default: gcg"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed" 
    )
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load judge model
    judge = GemmaJudge()
    
    # Setup log file
    model_name = args.model_name.split("/")[1].replace("-", "").lower()
    log_file_name = f"asr_base_{model_name}_{args.attack_type}_{args.seed}.json"
    print(f"\nLogs path: {log_file_name}")
    
    # Compute ASR
    results = compute_asr(
        model,
        tokenizer,
        args.prompts_file_name,
        judge,
        log_file_name,
    )
    
    # Print results
    print("\n" + "="*50)
    print("ATTACK SUCCESS RATE (ASR) RESULTS")
    print("="*50)
    print(f"Model: {args.model_name}")
    print("-"*50)
    print(f"ASR: {results['asr']:.2f}%")
    print(f"Successful jailbreaks: {results['success']}")
    print(f"Total Samples: {results['total_samples']}")
    print("="*50)

if __name__ == "__main__":
    main()
