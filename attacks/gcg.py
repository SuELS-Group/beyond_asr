import nanogcg  

def gcg_attack(model, tokenizer, question: str, target: str, config: nanogcg.GCGConfig) -> nanogcg.gcg.GCGResult:
    """Perform GCG attack on the given model and tokenizer."""
    messages = [{"role": "user", "content": question}]
    result = nanogcg.run(model,
                        tokenizer,
                        messages,
                        target,
                        config,)
    return result
