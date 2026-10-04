#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import warnings

from tqdm import tqdm

from evals.judge import GemmaJudge
from evals.scores import compute_metrics
from utils.counts import exp, log
from utils.env import set_seed, store_logs
from utils.inference_artifacts import load_model_and_tokenizer

warnings.simplefilter('ignore')

def compute_asr_step(
    model,
    tokenizer,
    prompts_file_name: str,
    log_file_name: str,
    step_count_func: str,
    judge: Any = None,
    verbose: bool = False,
) -> dict:
    """
    Compute cost-aware stepwise Attack Success Rate (ASR_STEP) on jailbreak dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        judge: Judge model
        prompts_file_name: Path to jailbreak prompts
        log_file_name: Path to store logs
        step_count_func: Function to calculate step count
        verbose: Print details for each sample
    
    Returns:
        Dictionary with metrics
    """
    with open(prompts_file_name, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    num_samples = len(dataset)
    print(f"\nEvaluating on {num_samples} samples...")
    outputs = []
    successful_total= 0
    score_total = 0.0
    for _, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        i=0
        success = 0
        score_per_sample = 0.0
        # Sort suffixes by loss in ascending order
        suffixes_sorted_by_loss = sorted(zip(sample["all_losses"], sample["all_strings"]), key=lambda x: x[0])
        suffixes_asc = [v for _, v in suffixes_sorted_by_loss]

        while i<len(suffixes_asc):
            new_question = f"{sample["question"]}{suffixes_asc[i]}"
            metrics = compute_metrics(model, tokenizer, new_question, sample["target"], judge)
            outputs.append({
                        "jailbreak_status": metrics["status"],
                        "jailbreak_index": i if metrics["status"] == "jailbreak" else -1,
                        "jailbreak_score": metrics["score"],
                        "jailbreak_question": new_question,
                        "jailbreak_response": metrics["response"],
                        "judge_reason": metrics["judge_reason"],
                    },)

            if metrics["status"] == "jailbreak":
                success += 1
                # Compute sample ASR for successful jailbreaks only
                score_per_sample += (metrics["score"] * globals()[step_count_func](i/len(suffixes_asc)))
                break
            i += 1
        successful_total += success
        score_total += score_per_sample
        
    if verbose:
        store_logs(outputs, log_file_name)
    results = {
        "final_asr_step_dataset": (score_total / num_samples) * 100,
        "success": successful_total,
        "total_samples": num_samples,
    }
    
    return results

def main():
    """Main function to compute cost-aware stepwise Attack Success Rate (ASR_STEP) on jailbreak dataset."""
    parser = argparse.ArgumentParser(
        description="Compute Stepwise Attack Success Rate (ASR) on jailbreak prompts"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--prompts_file_name",
        type=str,
        default="nanogcg_prompts.jsonl",
        help="Path to jailbreak prompts"
    )
    parser.add_argument(
        "--use_judge",
        action="store_false",
        help="Whether to use judge model for evaluation"
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="gcg",
        help="Type of attack to use, default: gcg"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed" 
    )
    parser.add_argument(
        "--step_count_func",
        type=str,
        default="log",
        help="Function to calculate step count (basic, exp, log)"
    )
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Setup log file
    model_name = args.model_name.split("/")[1].replace("-", "").lower()
    judge = None
    if args.use_judge:
        judge = GemmaJudge(device=model.device)
        log_file_name = f"asr_step_{model_name}_{args.attack_type}_judge_{args.seed}.json"
    else:
        log_file_name = f"asr_step_{model_name}_{args.attack_type}_nojudge_{args.seed}.json"
    print(f"\nLogs path: {log_file_name}")
    
    # Compute ASR
    results = compute_asr_step(
        model,
        tokenizer,
        args.prompts_file_name,
        log_file_name,
        args.step_count_func,
        judge,
        args.verbose
    )
    
    # Print results
    print("\n" + "="*50)
    print("STEPWISE ATTACK SUCCESS RATE (ASR) RESULTS")
    print("="*50)
    print(f"Model: {args.model_name}")
    print("-"*50)
    print(f"Query aware ASR_STEP: {results['final_asr_step_dataset']:.2f}%")
    print(f"Successful jailbreaks: {results['success']}")
    print(f"Total Samples: {results['total_samples']}")
    print("="*50)

if __name__ == "__main__":
    main()
