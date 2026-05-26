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

## MMLongBench-Doc Evaluation

The repository includes a minimal MMLongBench-Doc evaluation harness for two
methods:

- `base`: ColPali top-5 retrieval, then Qwen3-VL answers directly from retrieved
  pages
- `sleuth`: ColPali top-5 retrieval, clue discovery, page screening, evidence
  context, difficulty assessment, and core decision

The harness targets the official repository layout:

```text
MMLongBench-Doc/
  data/
    samples.json
    documents/
      *.pdf
```

It writes `predictions.jsonl`, `metrics.json`, `metrics_by_category.csv`,
`failed_examples.jsonl`, and `run_config.json`.

Mock smoke test with the included sample:

```bash
python scripts/run_mmlongbench_eval.py \
  --data_dir . \
  --method sleuth \
  --mode mock \
  --limit 1 \
  --output_dir runs/mmlongbench_eval_mock
```

Real SLEUTH evaluation on Sol or another GPU host:

```bash
python scripts/run_mmlongbench_eval.py \
  --data_dir /path/to/MMLongBench-Doc \
  --method sleuth \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --retriever vidore/colpali-v1.3-hf \
  --top_k 5 \
  --temperature 0.1 \
  --limit 3 \
  --output_dir runs/mmlongbench_debug
```

Use `--method base` to run the direct retrieved-page baseline. Do not treat mock
or tiny debug metrics as benchmark results.

On Sol, submit the same evaluation through Slurm with the hardened wrappers:

```bash
DATA_DIR=/path/to/MMLongBench-Doc LIMIT=3 sbatch slurm/mmlongbench_eval_sleuth_sol.sbatch
DATA_DIR=/path/to/MMLongBench-Doc LIMIT=3 sbatch slurm/mmlongbench_eval_base_sol.sbatch
```

The wrappers default to `~/mamba-envs/sleuth-static/bin/python`, scratch caches,
`public` partition/QoS, one A100 GPU, top-5 retrieval, temperature `0.1`, and
`vidore/colpali-v1.3-hf`.

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
