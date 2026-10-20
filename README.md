# Beyond ASR
Code for paper: http://arxiv.org/abs/2610.26589 submitted to NAACL'27.

### Environment setup:

1. Clone this project
2. Make sure uv is installed on your target system
3. Run `uv sync` to install dependencies into the virtual env
4. Run `source .venv/bin/activate` to activate it
5. Run `uv pip list` to check installed libraries. Follow uv troubleshooting if list is empty.

Now you are ready to run scripts!

### Running the code:
Eval Script:
`jailbreak.py`- to compute baseline and stepwise ASR scores

#### To run an attack, use (example with GCG, Vicuna 7B):

`uv run attacks/GCG/main.py`

#### To run a eval script, use:

ASR and ASR_Step computation (default: Vicuna): `uv run scripts/jailbreak.py`

`ARGS` are specified in each script.
