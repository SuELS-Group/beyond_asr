#!/usr/bin/env python3

import argparse
import json
import sys
import warnings
from pathlib import Path

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
    dataset: list,
    judge: GemmaJudge,
    log_file_name: str,
) -> dict:
    """
    Compute Attack Success Rate (ASR) on jailbreak dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: List of jailbreak prompts
        judge: Judge model
        log_file_name: Name of the log file
    
    Returns:
        Dictionary with metrics
    """
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
    # Persist log file
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
        default="lmsys/vicuna-7b-v1.5",
        help="Base model name or path"
    )
    parser.add_argument(
        "--prompts_file_name",
        type=str,
        help="Path to jailbreak prompts"
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
    print(f"\nLoading judge model...")
    judge = GemmaJudge()
    
    # Loading prompts dataset
    print(f"\nLoading prompts dataset: {args.prompts_file_name}")
    with open(args.prompts_file_name, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Setup log file
    model_name = args.model_name.split("/")[1].replace("-", "").lower()
    if "gcg" in args.prompts_file_name:
        attack_type = "gcg"  
    elif "pap" in args.prompts_file_name:
        attack_type = "pap"
    else:
        attack_type = "lrt"
    log_file_name = f"asr_base_{model_name}_{attack_type}_{args.seed}.json"
    print(f"\nASR Evaluation Log file: {log_file_name}")
    
    # Compute ASR
    results = compute_asr(
        model,
        tokenizer,
        dataset,
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
