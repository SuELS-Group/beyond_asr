### Beyond ASR

Scripts:
`jailbreak_baseline.py`- to set the baseline ASR scores
`jailbreak.py`- to compute stepwise ASR scores

## Environment setup

1. Clone this project
2. Make sure uv is installed on your target system
3. Run `uv sync` to install dependencies into the virtual env and activate it
4. Run `uv pip list` to check installed libraries. Follow uv troubleshooting if list is empty.

Now you are ready to run scripts!

To run a script use:
Basic use: `uv run scripts/jailbreak.py --verbose`
`ARGS` are specified in each script.

Example command (evaluate Qwen3.5 0.8B on GCG attack with a judge):
`uv run scripts/jailbreak_baseline.py --model_name "unsloth/Qwen3.5-0.8B" --verbose --judge`
