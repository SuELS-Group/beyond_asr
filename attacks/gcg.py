import nanogcg

CONFIG = nanogcg.GCGConfig(use_prefix_cache=False,
                       num_steps=100,
                       verbosity="ERROR",)   

def gcg_attack(model, tokenizer, question: str, target: str) -> nanogcg.gcg.GCGResult:
    """Perform GCG attack on the given model and tokenizer."""
    messages = [{"role": "user", "content": question}]
    result = nanogcg.run(model,
                        tokenizer,
                        messages,
                        target,
                        CONFIG,)
    return result
