# Beyond ASR

Scripts:
`jailbreak_baseline.py`- to set the baseline ASR scores
`jailbreak.py`- to compute stepwise ASR scores

## Environment setup

1. Clone this project
2. Make sure uv is installed on your target system
3. Run `uv sync` to install dependencies into the virtual env
4. Run `source .venv/bin/activate` to activate it
5. Run `uv pip list` to check installed libraries. Follow uv troubleshooting if list is empty.

Now you are ready to run scripts!

To run a script use:
Basic use: `uv run scripts/jailbreak.py --verbose`
`ARGS` are specified in each script.

Example command (evaluate Qwen3.5 0.8B on GCG attack with a judge):
`uv run scripts/jailbreak_baseline.py --model_name "unsloth/Qwen3.5-0.8B" --verbose --judge`

## Readable GCG suffixes

`attacks/readable_gcg.py` runs the same GCG objective with a candidate-token
allow-list that decodes to printable, human-readable text. It writes the raw
optimizer results plus filtered candidates to JSONL; use `best_readable_string`
or `readable_strings` when evaluating the attack.

```bash
uv run attacks/readable_gcg.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --max-samples 2 \
  --output readable_nanogcg_prompts.jsonl
```

## DeepEval Answer Relevancy ASR

`scripts/deepeval_asr.py` loads a local Llama 3.1 8B Instruct model as the
DeepEval judge, scores each response with `AnswerRelevancyMetric`, and defines
ASR as the fraction of responses whose score is at least the threshold.

For the included prompt records, `target` is used as the response by default:

```bash
uv run scripts/deepeval_asr.py \
--input-file nanogcg_prompts.jsonl \
--output-file outputs/deepeval_asr.json \
--append-best-string
```

For saved inference logs, select the response field explicitly, for example:

```bash
uv run scripts/deepeval_asr.py \
--input-file outputs/responses.json \
--response-key jailbreak_response
```
