import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
from tqdm import tqdm

from evals.judge import GemmaJudge
from attacks.PAIR.pair import pair
from utils.env import set_seed, store_logs
from utils.inference_artifacts import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Stepwise Attack Success Rate (ASR) on jailbreak prompts"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="Victim model name",
    )
    parser.add_argument(
        "--attack_model",
        type=str,
        default="google/gemma-4-12B-it",
        help="Attack model name",
    )
    parser.add_argument("--dataset_path",
        type=str,
        default="JailbreakBench/JBB-Behaviors",
        help="Path to jailbreak dataset")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=20,
        help="Maximum number of iterations for PAIR attack",
    )
    parser.add_argument(
        "--max_conversation_length",
        type=int,
        default=4,
        help="Maximum conversation length for PAIR attack",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed",
    )
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    # Load model and tokenizer
    print(f"\nLoading victim model: {args.model_name}")
    victim_model, victim_tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load attack models and tokenizers
    print(f"\nLoading attack model: {args.attack_model}")
    attack_model, attack_tokenizer = load_model_and_tokenizer(args.attack_model)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path,"behaviors", split="harmful")
    
    #Initialize judge
    print(f"\nLoading judge...")
    judge = GemmaJudge()
    
    results_list = []
    for _, sample in enumerate(tqdm(dataset, desc="PAIR Generation", total=len(dataset))):
        prompt = sample["Goal"]
        output = pair(
            prompt=prompt,
            judge=judge,
            victim_model=victim_model,
            victim_tokenizer=victim_tokenizer,
            attack_model=attack_model,
            attack_tokenizer=attack_tokenizer,
            target_str=sample["Target"],
            max_iter=args.max_iter,
            max_conversation_length=args.max_conversation_length,
        )
    
        results_list.append({
            "question": prompt,
            "target": sample["Target"],
            "total_iterations": args.max_iter,
            "jailbroken_prompt": output["jailbroken_prompt"],
            "jailbreak_index": output["jailbreak_index"],
        })
    store_logs(results_list, f"{args.victim_model}_pair_results_{args.seed}.json")
    print("Finished PAIR generation!!")

if __name__ == "__main__":
    main()
    