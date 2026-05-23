# AgenticDocAI Static SLEUTH Baseline

This repository implements a static SLEUTH-style long-document understanding
baseline inspired by "Resolving Evidence Sparsity: Agentic Context Engineering
for Long-Document Understanding."

The baseline uses:

- `Qwen/Qwen3-VL-8B-Instruct` as the local VLM target
- `ColPali-v1.3` for top-5 visual page retrieval
- one-shot retrieval followed by four-agent evidence refinement
- local markdown prompt loading from `agent_prompts.md`
- SOL-specific instruction loading from `sol_instructions.md`

This is research prototype wiring. It does not implement training, SFT, RL,
GRPO, LangGraph, iterative retrieval, or benchmark score reproduction.

## Prompt Sources

`agent_prompts.md` is the canonical source for SLEUTH-style agent prompts. The
code reads it at runtime and saves the prompt text used for reproducibility.

`sol_instructions.md` is mandatory in SOL mode. Its contents are included in
every agent prompt under a clearly marked `SOL-SPECIFIC INSTRUCTIONS` section
and copied to the run output.

## Conda Setup

```bash
conda env create -f environment.yml
conda activate sleuth-static
```

## Venv Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ColPali may require separate installation:

```bash
pip install colpali-engine
```

If that fails, install the official ColPali package or repository according to
the current ColPali documentation.

## Mock Mode

Mock mode runs without GPU, Qwen, or ColPali:

```bash
python scripts/run_sleuth_baseline.py \
  --pdf data/example.pdf \
  --question "What is the main result?" \
  --out-dir runs/mock_test \
  --mode mock \
  --retriever dummy \
  --llm mock
```

The repository also includes a small MMLongBench-Doc PDF/question sample:

```bash
python scripts/run_sleuth_baseline.py \
  --pdf samples/mmlongbench_doc/a4f3ced0696009fec3179f493e4f28c4.pdf \
  --question "WHAT IS USCA CASE NUMBER?" \
  --out-dir runs/mmlongbench_doc_mock \
  --mode mock
```

## SOL Mode

SOL mode requires `sol_instructions.md`, Qwen3-VL dependencies, and ColPali:

```bash
python scripts/run_sleuth_baseline.py \
  --pdf data/example.pdf \
  --question "According to Table II, which datasets have exactly three methods?" \
  --out-dir runs/sol_test \
  --mode sol
```

Or submit the included MMLongBench-Doc sample as a one-line Slurm job:

```bash
sbatch slurm/sleuth_mmlongbench_doc_sol.sbatch
```

For compatibility with a previous command shape, this alias launches the same
SLEUTH job:

```bash
sbatch slurm/gdpo_debug_upstream.sbatch
```

## Outputs

A run writes rendered pages, agent artifacts, and final outputs:

```text
runs/example/
  pages/
  agents/
  final/
    retrieved_pages.json
    evidence_context.json
    evidence_summary.txt
    difficulty.json
    final_answer.json
    agent_prompts_used.md
    sol_instructions_used.md
```

## Future Extensions

- replace one-shot retrieval with iterative retrieval
- add tool actions
- collect trajectories
- train a bridge-query/evidence agent
- add RL/GRPO later
