"""
GCG with the default settings. Code from:
https://github.com/GraySwanAI/nanoGCG/blob/main/examples/simple.py
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from datasets import load_dataset
from gcg import GCGConfig, ProbeSamplingConfig, run
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.env import set_seed, store_logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="JailbreakBench/JBB-Behaviors", help="Path to jailbreak dataset")
    parser.add_argument("--max_samples",
                        type=int,
                        default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--device",
                        type=str,
                        default="cuda")
    parser.add_argument("--dtype",
                        type=str,
                        default="float16")
    parser.add_argument("--probe_sampling",
                        action="store_true")
    parser.add_argument("--seed",
                        type=int,
                        default=2026,
                        help="Random seed")
        
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"\nLoaded {args.model} on {args.device} ")

    probe_sampling_config = None
    if args.probe_sampling:
        draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=getattr(torch, args.dtype)).to(args.device)
        draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        probe_sampling_config = ProbeSamplingConfig(
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
        )
        
    config = GCGConfig(
                verbosity="WARNING",
                probe_sampling_config=probe_sampling_config,)
    print("\nSet GCG Config.")

    # Load dataset
    dataset = load_dataset(args.dataset_path,"behaviors", split="harmful")
    subset = dataset.select([i for i in list(range(args.max_samples))]) if args.max_samples else dataset
    print(f"\nLoaded dataset from: {args.dataset_path}")

    result_list = []
    for _, sample in enumerate(tqdm(subset, desc="GCG Evaluation", total=len(subset))):
        question = sample["Goal"]
        target = sample["Target"]
    
        messages = [{"role": "user", "content": question}]

        result = run(
            model,
            tokenizer,
            messages,
            target,
            config,
        )
        min_loss_index = result.losses.index(min(result.losses))
        result_list.append({
                    "question": question,
                    "target": target,
                    "min_loss_index": min_loss_index,
                    "best_loss": result.best_loss,
                    "best_string": result.best_string,
                    "all_losses": result.losses,
                    "all_strings": result.strings,
                    "with_probe_sampling": args.probe_sampling,
        })

    model_name = args.model_name.split("/")[1].replace("-", "").lower()
    store_logs(result_list, f"{model_name}_gcg_results_{args.seed}.json")
    print("Finished GCG generation!!")

if __name__ == "__main__":
    main()
