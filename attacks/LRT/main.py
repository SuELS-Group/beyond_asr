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

from attacks.LRT.low_resource_translate import safe_translate
from utils.env import set_seed, store_logs
from utils.inference_artifacts import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="lmsys/vicuna-7b-v1.5")
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

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print(f"tokenizer.chat_template: {tokenizer.chat_template}")

    # Load dataset
    dataset = load_dataset(args.dataset_path,"behaviors", split="harmful")
    print(f"\nLoaded dataset from: {args.dataset_path}")

    result_list = []
    for _, sample in enumerate(tqdm(dataset, desc="LRT Generation", total=len(dataset))):
        question = sample["Goal"]
    
        language, translated_question = safe_translate(question)
        result_list.append({
                    "question": question,
                    "target": sample["Target"],
                    "target_language": language,
                    "translated_question": translated_question
        })

    model_name = args.model_name.split("/")[1].replace("-", "").lower()
    store_logs(result_list, f"{model_name}_lrt_results_{args.seed}.json")
    print("Finished Low-resource translation generation!!")

if __name__ == "__main__":
    main()
