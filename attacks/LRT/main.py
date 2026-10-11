"""
GCG with the default settings. Code from:
https://github.com/GraySwanAI/nanoGCG/blob/main/examples/simple.py
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
from tqdm import tqdm

from attacks.LRT.low_resource_translate import run_translate_attack
from utils.env import set_seed, store_logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
                        type=str,
                        default="JailbreakBench/JBB-Behaviors", help="Path to jailbreak dataset")
    parser.add_argument("--seed",
                        type=int,
                        default=2026,
                        help="Random seed")
        
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load dataset
    dataset = load_dataset(args.dataset_path,"behaviors", split="harmful")
    print(f"\nLoaded dataset from: {args.dataset_path}")

    result_list = []
    for _, sample in enumerate(tqdm(dataset, desc="LRT Generation", total=len(dataset))):
        question = sample["Goal"]
        translations = run_translate_attack(question)
        result_list.append({
                            "question": question,
                            "target": sample["Target"],
                            "translations": translations
                })
    store_logs(result_list, f"lrt_results_{args.seed}.json")
    print("Finished Low-resoure translation generation!!")

if __name__ == "__main__":
    main()
