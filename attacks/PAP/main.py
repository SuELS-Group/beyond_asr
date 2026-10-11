import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset

from attacks.PAP.pap import run_pap
from utils.env import set_seed, store_logs
from utils.inference_artifacts import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="JailbreakBench/JBB-Behaviors", help="Path to jailbreak dataset")
    parser.add_argument("--device",
                        type=str,
                        default="cuda")
    parser.add_argument("--seed",
                        type=int,
                        default=2026,
                        help="Random seed")
        
    args = parser.parse_args()
    return args

def remove_quotes(sentences) -> list[str]:
    cleaned_sentences = []
    for s in sentences:
        if s.startswith('"""') and s.endswith('"""'):
            cleaned_sentences.append(s[3:-3])  # Remove triple quotes
        elif s.startswith('"') and s.endswith('"'):
            cleaned_sentences.append(s[1:-1])  # Remove double quotes
        elif s.startswith("'") and s.endswith("'"):
            cleaned_sentences.append(s[1:-1])  # Remove single quotes
        else:
            cleaned_sentences.append(s)
    return cleaned_sentences

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path,"behaviors", split="harmful")
    print(f"\nLoaded dataset from: {args.dataset_path}")
    
    # Load PAP persuasion taxonomy
    persuasion_taxonomy = []
    with open("attacks/PAP/persuasion_taxonomy.jsonl", 'r') as file:
        for line in file:
            persuasion_taxonomy.append(json.loads(line))
    print(f"Loaded persuasion taxonomy with {len(persuasion_taxonomy)} entries.")
    
    result_list = []
    for _, sample in enumerate(tqdm(dataset, desc="PAP Generation", total=len(dataset))):
        output = run_pap(model,
                        tokenizer,
                        persuasion_taxonomy,
                        prompt=sample["Goal"],)
        result_list.append({
                            "question": sample["Goal"],
                            "target": sample["Target"],
                            "output": output,
                        })
    model_name = args.model_name.split("/")[1].replace("-", "").lower()
    store_logs(result_list, f"{model_name}_pap_prompts_{args.seed}.json")
    print("Finished PAP generation!!")
            
if __name__ == "__main__":
    main()
    