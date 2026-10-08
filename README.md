# Beyond ASR

Eval Scripts:
`jailbreak_baseline.py`- to set the baseline ASR scores
`jailbreak.py`- to compute stepwise ASR scores

## Environment setup

1. Clone this project
2. Make sure uv is installed on your target system
3. Run `uv sync` to install dependencies into the virtual env
4. Run `source .venv/bin/activate` to activate it
5. Run `uv pip list` to check installed libraries. Follow uv troubleshooting if list is empty.

Now you are ready to run scripts!

To run an attack, use(example with GCG, Vicuna 7B):

`uv run attacks/GCG/main.py`

To run a eval script, use:

Baseline: `uv run scripts/jailbreak_baseline.py`

ASR_Step: `uv run scripts/jailbreak.py`

`ARGS` are specified in each script.

