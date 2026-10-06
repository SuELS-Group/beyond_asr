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

from attacks.GCG.gcg import GCGConfig, ProbeSamplingConfig, run
from utils.env import set_seed, store_logs
from utils.inference_artifacts import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="JailbreakBench/JBB-Behaviors", help="Path to jailbreak dataset")
    parser.add_argument("--max_samples",
                        type=int,
                        default=None,
                        help="Max samples to evaluate (default: all)")
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

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print(f"tokenizer.chat_template: {tokenizer.chat_template}")

    probe_sampling_config = None
    if args.probe_sampling:
        draft_model, draft_tokenizer = load_model_and_tokenizer("openai-community/gpt2")
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
    print(f"\nLoaded dataset from: {args.dataset_path}")

    result_list = []
    for _, sample in enumerate(tqdm(dataset, desc="GCG Generation", total=len(dataset))):
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
