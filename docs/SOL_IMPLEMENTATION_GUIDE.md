# SOL Implementation Guide

This guide is for the static SLEUTH-style baseline in this repository. The
intended workflow is:

1. Push this project to GitHub from your local machine.
2. Pull or clone it on ASU Sol.
3. Build the Python environment from a compute allocation.
4. Smoke-test mock mode.
5. Run SOL mode with Qwen3-VL and ColPali on a GPU allocation.

Official ASU RC references used here:

- [New User Guide](https://docs.rc.asu.edu/new-user-guide/)
- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)
- [Python overview](https://docs.rc.asu.edu/python/)
- [Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos)
- [Helpful Slurm Commands](https://docs.rc.asu.edu/helpful-slurm-commands)
- [Supercomputer Hardware](https://docs.rc.asu.edu/supercomputer-hardware)

## 1. Local GitHub Upload

From the project root on your local machine:

```bash
cd /Users/god/Documents/AgenticDocAI
git init
git status
git add .
git commit -m "Add static SLEUTH baseline"
git branch -M main
```

The current GitHub repository is:

```text
https://github.com/MalveauxLuke/Agentic-Document-understanidng.git
```

If you are creating a fresh fork instead, connect and push:

```bash
git remote add origin git@github.com:<YOUR_GITHUB_USER>/Agentic-Document-understanidng.git
git push -u origin main
```

If using HTTPS instead of SSH:

```bash
git remote add origin https://github.com/<YOUR_GITHUB_USER>/Agentic-Document-understanidng.git
git push -u origin main
```

Do not commit PDFs, model weights, run outputs, or scratch artifacts. The
project `.gitignore` already excludes `data/` and `runs/`.

## 2. Clone Or Pull On Sol

Connect to ASU VPN if required, then SSH:

```bash
ssh <ASURITE>@sol.asu.edu
```

Keep code in home:

```bash
cd ~
git clone https://github.com/MalveauxLuke/Agentic-Document-understanidng.git
cd Agentic-Document-understanidng
```

For future updates:

```bash
cd ~/Agentic-Document-understanidng
git pull --ff-only
```

Use the login node only for light work: navigation, editing, Git operations,
and job submission. Do not run model inference or heavy installs on login nodes.

## 3. Create The Environment

Request a light compute allocation for environment creation:

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4
```

Load Mamba and create the environment:

```bash
module load mamba/latest
mkdir -p $HOME/mamba-envs
mamba env create -p $HOME/mamba-envs/sleuth-static -f environment.yml

export SLEUTH_ENV_NAME=$HOME/mamba-envs/sleuth-static
export PYTHON_BIN=$SLEUTH_ENV_NAME/bin/python
$PYTHON_BIN -V
$PYTHON_BIN -c "import sys; print(sys.executable)"
```

Install ColPali into that environment:

```bash
$PYTHON_BIN -m pip install colpali-engine
```

If ColPali installation fails, follow the current official ColPali installation
instructions and keep all ColPali-specific fixes isolated to the environment or
`sleuth/retrieval/colpali_retriever.py`.

If compiled packages fail with `libstdc++` or `CXXABI_*` errors:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 4. Preflight Checks

Still inside the active environment:

```bash
$PYTHON_BIN - <<'PY'
import fitz
import pydantic
import yaml
import rank_bm25
print("core imports ok")
PY
```

Check the heavy stack:

```bash
$PYTHON_BIN - <<'PY'
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import ColPaliForRetrieval, ColPaliProcessor
import qwen_vl_utils
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("heavy imports ok")
PY
```

Run the unit tests from a compute allocation:

```bash
$PYTHON_BIN -m pytest -q
```

## 5. Place Data And Outputs On Scratch

Use scratch for PDFs, model cache, and run outputs:

```bash
mkdir -p /scratch/$USER/agenticdocai/data
mkdir -p /scratch/$USER/agenticdocai/runs
mkdir -p /scratch/$USER/huggingface
mkdir -p /scratch/$USER/tmp
```

Install the official MMLongBench-Doc data layout from your interactive VSCode
shell:

```bash
bash scripts/install_mmlongbench_doc.sh
```

This downloads:

```text
/scratch/$USER/agenticdocai/data/MMLongBench-Doc/data/samples.json
/scratch/$USER/agenticdocai/data/MMLongBench-Doc/data/documents/*.pdf
```

For a quick downloader test without all PDFs:

```bash
LIMIT_EXAMPLES=2 bash scripts/install_mmlongbench_doc.sh
```

`LIMIT_EXAMPLES=2` downloads `samples.json` plus the PDFs needed by the first
two benchmark rows, so it is suitable for a tiny eval smoke test. A full install
uses no limit.

To force a re-download:

```bash
FORCE=1 bash scripts/install_mmlongbench_doc.sh
```

If you already downloaded the dataset somewhere else, point eval jobs at it:

```bash
export DATA_DIR=/path/to/MMLongBench-Doc
```

For single-PDF experiments, move or upload PDFs to:

```text
/scratch/$USER/agenticdocai/data/
```

Set cache paths before real runs:

```bash
export HF_HOME=/scratch/$USER/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=/scratch/$USER/tmp
```

Scratch is temporary. Move important final outputs to persistent storage after
the run.

## 6. Mock Smoke Test On Sol

Mock mode does not need GPU, Qwen, or ColPali. Use it to confirm repo wiring:

```bash
$PYTHON_BIN scripts/run_sleuth_baseline.py \
  --pdf /scratch/$USER/agenticdocai/data/example.pdf \
  --question "What is the main result?" \
  --out-dir /scratch/$USER/agenticdocai/runs/mock_test \
  --mode mock \
  --retriever dummy \
  --llm mock
```

Expected final files:

```text
/scratch/$USER/agenticdocai/runs/mock_test/final/retrieved_pages.json
/scratch/$USER/agenticdocai/runs/mock_test/final/evidence_context.json
/scratch/$USER/agenticdocai/runs/mock_test/final/evidence_summary.txt
/scratch/$USER/agenticdocai/runs/mock_test/final/difficulty.json
/scratch/$USER/agenticdocai/runs/mock_test/final/final_answer.json
/scratch/$USER/agenticdocai/runs/mock_test/final/agent_prompts_used.md
```

For the evaluation harness, smoke-test the included MMLongBench-Doc sample:

```bash
$PYTHON_BIN scripts/run_mmlongbench_eval.py \
  --data_dir . \
  --method sleuth \
  --mode mock \
  --limit 1 \
  --output_dir /scratch/$USER/agenticdocai/runs/mmlongbench_eval_mock
```

This should create:

```text
/scratch/$USER/agenticdocai/runs/mmlongbench_eval_mock/predictions.jsonl
/scratch/$USER/agenticdocai/runs/mmlongbench_eval_mock/metrics.json
/scratch/$USER/agenticdocai/runs/mmlongbench_eval_mock/metrics_by_category.csv
/scratch/$USER/agenticdocai/runs/mmlongbench_eval_mock/failed_examples.jsonl
/scratch/$USER/agenticdocai/runs/mmlongbench_eval_mock/run_config.json
```

You can also submit the eval harness through Slurm. These wrappers use the same
fixes that solved the earlier Sol issues: direct environment Python, cleared
`PYTHONPATH`/`PYTHONHOME`, `PYTHONNOUSERSITE=1`, env-local `LD_LIBRARY_PATH`,
scratch HF caches, `public` partition/QoS, one A100 GPU, 24-hour walltime, and
HF-native ColPali.

Before submitting a long run, test the allocation and model loading:

```bash
ANSWER_EXTRACTOR_API_KEY=... \
sbatch slurm/mmlongbench_eval_sleuth_debug_sol.sbatch
```

This uses `htc` + `debug`, one A100, 30 minutes, and `LIMIT=1` by default.
SOL evaluation requires an OpenAI-compatible answer-extraction key; the default
extractor model is `gpt-4.1-mini` unless `ANSWER_EXTRACTOR_MODEL` is set.

SLEUTH eval, defaulting to:

```text
/scratch/$USER/agenticdocai/data/MMLongBench-Doc
```

```bash
ANSWER_EXTRACTOR_API_KEY=... \
sbatch slurm/mmlongbench_eval_sleuth_sol.sbatch
```

Official MMLongBench-Doc SLEUTH debug run:

```bash
ANSWER_EXTRACTOR_API_KEY=... \
LIMIT=3 \
sbatch slurm/mmlongbench_eval_sleuth_sol.sbatch
```

For a full run after the debug job works, use `LIMIT=ALL`:

```bash
ANSWER_EXTRACTOR_API_KEY=... \
LIMIT=ALL \
sbatch slurm/mmlongbench_eval_sleuth_sol.sbatch
```

Official MMLongBench-Doc base debug run:

```bash
ANSWER_EXTRACTOR_API_KEY=... \
LIMIT=3 \
sbatch slurm/mmlongbench_eval_base_sol.sbatch
```

Optional category filter:

```bash
ANSWER_EXTRACTOR_API_KEY=... \
CATEGORY=Table \
LIMIT=3 \
sbatch slurm/mmlongbench_eval_sleuth_sol.sbatch
```

To sweep render resolution, keep every other setting fixed and vary only
`RENDER_DPI`:

```bash
for dpi in 144 180 200; do
  ANSWER_EXTRACTOR_API_KEY=... \
  RENDER_DPI="$dpi" \
  LIMIT=75 \
  OUT_DIR="/scratch/$USER/agenticdocai/runs/mmlongbench_eval_sleuth_dpi_${dpi}" \
  sbatch slurm/mmlongbench_eval_sleuth_sol.sbatch
done
```

If your environment is under `~/.conda/envs` instead of `~/mamba-envs`:

```bash
SLEUTH_ENV_NAME=$HOME/.conda/envs/sleuth-static \
LIMIT=3 \
sbatch slurm/mmlongbench_eval_sleuth_sol.sbatch
```

The Slurm outputs are written next to your submission directory:

```bash
cat slurm-sleuth-mmlb-sleuth-<JOB_ID>.out
cat slurm-sleuth-mmlb-sleuth-<JOB_ID>.err
```

The evaluation artifacts default to:

```text
/scratch/$USER/agenticdocai/runs/mmlongbench_eval_<method>_<JOB_ID>/
```

Each eval run prints a readable result summary when it finishes. To display a
completed run again:

```bash
$PYTHON_BIN scripts/show_mmlongbench_results.py /scratch/$USER/agenticdocai/runs/mmlongbench_eval_sleuth_<JOB_ID>
```

## 7. Interactive SOL Run

Request a GPU allocation. For a first real test, use one GPU and a short
walltime:

```bash
salloc -p public -q public -t 04:00:00 -c 8 --mem=80G -G 1
```

Then:

```bash
export SLEUTH_ENV_NAME=$HOME/mamba-envs/sleuth-static
export PYTHON_BIN=$SLEUTH_ENV_NAME/bin/python

export HF_HOME=/scratch/$USER/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=/scratch/$USER/tmp

$PYTHON_BIN scripts/run_sleuth_baseline.py \
  --pdf /scratch/$USER/agenticdocai/data/example.pdf \
  --question "According to Table II, which datasets have exactly three methods?" \
  --out-dir /scratch/$USER/agenticdocai/runs/sol_test \
  --mode sol
```

SOL mode enforces:

- `Qwen/Qwen3-VL-8B-Instruct`
- `vidore/colpali-v1.3-hf` by default for the current HF-native ColPali path
- top-5 retrieval
- temperature `0.1`
- mandatory `sol_instructions.md`

To run a tiny real MMLongBench-Doc evaluation interactively after cloning or
staging the official dataset:

```bash
$PYTHON_BIN scripts/run_mmlongbench_eval.py \
  --data_dir /path/to/MMLongBench-Doc \
  --method sleuth \
  --mode sol \
  --top_k 5 \
  --temperature 0.1 \
  --limit 3 \
  --output_dir /scratch/$USER/agenticdocai/runs/mmlongbench_eval_sleuth_debug
```

Switch to the direct baseline with:

```bash
$PYTHON_BIN scripts/run_mmlongbench_eval.py \
  --data_dir /path/to/MMLongBench-Doc \
  --method base \
  --mode sol \
  --limit 3 \
  --output_dir /scratch/$USER/agenticdocai/runs/mmlongbench_eval_base_debug
```

## 8. Batch SOL Run

Use the included Slurm template:

```bash
cd ~/Agentic-Document-understanidng

export PDF=/scratch/$USER/agenticdocai/data/example.pdf
export QUESTION="According to Table II, which datasets have exactly three methods?"
export OUT_DIR=/scratch/$USER/agenticdocai/runs/sol_batch_test

sbatch --export=ALL scripts/submit_sol.sbatch
```

For the included MMLongBench-Doc sample, use the one-line wrapper:

```bash
sbatch slurm/sleuth_mmlongbench_doc_sol.sbatch
```

The wrapper defaults to the environment at:

```text
~/mamba-envs/sleuth-static
```

It calls that environment's `bin/python` directly instead of relying on
`source activate`, and it clears `PYTHONPATH`/`PYTHONHOME` before running. This
avoids Sol module Python leaking into the batch job.

If your environment lives somewhere else, override it at submit time:

```bash
SLEUTH_ENV_NAME=$HOME/.conda/envs/sleuth-static sbatch slurm/sleuth_mmlongbench_doc_sol.sbatch
```

There is also a compatibility alias for users who want the old single-line
shape from another project:

```bash
sbatch slurm/gdpo_debug_upstream.sbatch
```

That alias launches the same SLEUTH MMLongBench-Doc job; it is not a GDPO/VERL
training job.

Check job status:

```bash
myjobs
sq -u $USER
```

After completion:

```bash
seff <JOB_ID>
```

Inspect outputs:

```bash
ls -R "$OUT_DIR"
cat "$OUT_DIR/final/final_answer.json"
```

## 9. Expected Real Output Layout

```text
runs/sol_test/
  pages/
    page_0001.png
    page_0002.png
  agents/
    clue_page_0000.json
    screen_page_0000.json
  final/
    retrieved_pages.json
    evidence_context.json
    evidence_summary.txt
    difficulty.json
    final_answer.json
    agent_prompts_used.md
    sol_instructions_used.md
```

## 10. Common Failure Fixes

If `ModuleNotFoundError` appears:

```bash
export SLEUTH_ENV_NAME=$HOME/mamba-envs/sleuth-static
export PYTHON_BIN=$SLEUTH_ENV_NAME/bin/python
$PYTHON_BIN -V
$PYTHON_BIN -c "import sys; print(sys.executable)"
$PYTHON_BIN -c "import pydantic; print('pydantic ok')"
```

If Sol still runs old code:

```bash
cd ~/Agentic-Document-understanidng
git status
git pull --ff-only
```

If model downloads fill home, confirm cache paths:

```bash
echo "$HF_HOME"
echo "$TRANSFORMERS_CACHE"
```

If ColPali API errors occur, update only `sleuth/retrieval/colpali_retriever.py`
so the rest of the pipeline interface remains stable.

If Qwen loading fails, likely causes are insufficient GPU memory, incompatible
`transformers` / `qwen-vl-utils`, or model identifier changes. First verify
imports, then try a GPU node with more memory.

If jobs stay pending, inspect:

```bash
myjobs
thisjob <JOB_ID>
myfairshare
```

For syntax/preflight testing, use shorter debug-style allocations before running
full jobs.
