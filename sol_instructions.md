# SOL Generic Rules, Known Issues, and Practical Fixes

This note is intentionally **not project-specific**.

It is meant to be reusable across different Sol projects and captures two kinds
of information:

1. Official Sol / ASU Research Computing rules and operating constraints
2. Practical cross-project issues we have hit on Sol and how to address them

Official references used for this note:

- [ASU RC New User Guide](https://docs.rc.asu.edu/new-user-guide/)
- [ASU RC Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos/)
- [ASU RC Resource Limits](https://docs.rc.asu.edu/resource-limits/)
- [ASU RC Helpful Slurm Commands](https://docs.rc.asu.edu/helpful-slurm-commands/)
- [ASU RC Browser-based Web Portal](https://docs.rc.asu.edu/web-portal/)
- [ASU RC Acceptable Use Policy](https://cores.research.asu.edu/research-computing/policies)
- [ASU RC Python Envs and Mamba](https://docs.rc.asu.edu/mamba)
- [ASU RC Python overview](https://docs.rc.asu.edu/python/)
- [ASU RC Brief Example](https://docs.rc.asu.edu/a-brief-example)

## 1. Official Sol rules and constraints

These are the main rules that matter in practice.

### 1.1 Use login nodes only for light, non-computational work

ASU RC defines login nodes as shared nodes for simple, non-computational tasks
and says compute and storage-heavy activity should go to compute nodes. The
Acceptable Use Policy is explicit that running jobs on login nodes is prohibited,
will generate a warning, and may be terminated without advance notice.

What this means operationally:

- do file navigation, editing, small inspections, and job submission on login nodes
- do not run long Python jobs, model inference, training, or heavy compilation there
- request a compute allocation with `salloc` or submit with `sbatch` for anything substantial

Relevant official sources:

- [New User Guide](https://docs.rc.asu.edu/new-user-guide/)
- [Acceptable Use Policy](https://cores.research.asu.edu/research-computing/policies)

### 1.2 Idle or wasteful jobs can be terminated

The AUP says jobs may be terminated if they:

- impact other users' performance
- are excessively idle
- leave interactive sessions idle for four or more hours
- use "sleep loops"
- create unattended listening services

Practical implication:

- do not leave long idle interactive GPU sessions running
- do not leave background services listening on ports unless explicitly allowed
- clean up stale interactive jobs

Relevant official source:

- [Acceptable Use Policy](https://cores.research.asu.edu/research-computing/policies)

### 1.3 Scratch is temporary, shared, and should be actively used

The official docs say:

- `/scratch/[asurite]` is available to all users
- scratch is a temporary, shared resource
- users must actively use files stored there
- inactive files may be removed after 90 days
- scratch is intended for immediate computational use

The policy page also notes:

- scratch has a per-user limit
- scratch has a 20 million file limit per user
- scratch does not provide the same protection guarantees as home storage

Practical implication:

- keep active job outputs and large temporary artifacts on scratch
- do not treat scratch as permanent archival storage
- periodically move important artifacts elsewhere

Relevant official sources:

- [Resource Limits](https://docs.rc.asu.edu/resource-limits/)
- [Acceptable Use Policy](https://cores.research.asu.edu/research-computing/policies)

### 1.4 Home is persistent but smaller

ASU RC documents that:

- `/home` has a quota of 100 GiB
- it is intended for important files and persistent storage
- it is the right place for software installations and package environments
- unlike scratch, it is not a purge-style temporary area

Practical implication:

- keep code, small configs, environments, and important scripts in home
- do not park large experiment artifacts there unless you understand your quota usage

Relevant official sources:

- [Resource Limits](https://docs.rc.asu.edu/resource-limits/)
- [Acceptable Use Policy](https://cores.research.asu.edu/research-computing/policies)

### 1.5 Use the right partition and QoS

The official partition guidance matters a lot.

Key public-facing rules:

- `public` is the general default for most users and can run up to 7 days
- `htc` is for jobs that complete within 4 hours and has scheduling advantages
- `lightwork` is for lighter tasks like building environments, compiling software,
  VSCode tunnels, or bulk file operations
- `lightwork` has a 24-hour limit and a maximum of 8 CPU cores per node
- misuse of `lightwork` can result in cancellation and loss of eligibility
- `private` jobs on private nodes are preemptible

Practical implication:

- use `lightwork` for env creation, lightweight setup, or low-duty-cycle tasks
- use `htc` for short real compute jobs
- use `public` for normal longer-running research jobs
- do not burn full cores at high sustained utilization on `lightwork`

Relevant official source:

- [Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos/)

### 1.6 VPN and authentication matter

The current ASU RC getting-started guidance says continued connectivity to the
ASU SSLVPN is required for stable access to Research Computing resources.

Practical implication:

- if connections are flaky, reconnect the ASU SSLVPN before assuming Sol itself is down
- if SSH or portal behavior seems inconsistent, check VPN first

Relevant official source:

- [New User Guide](https://docs.rc.asu.edu/new-user-guide/)

### 1.7 The web portal is a first-class access path

The Sol web portal supports:

- shell access
- file browsing for home and scratch
- job monitoring
- job composer
- interactive apps

Practical implication:

- it is a good fallback when SSH is awkward
- it is useful for file transfers and job inspection
- it is not the best path for moving very large files

Relevant official source:

- [Browser-based Web Portal](https://docs.rc.asu.edu/web-portal/)

## 2. How to set up and run a Python environment on Sol

This is the generic environment pattern we have been using.

### 2.1 Do environment creation and package installation from a compute allocation

ASU RC's Python docs explicitly warn not to install packages on the login nodes
or inside Jupyter notebooks. Their examples use an interactive allocation first.

Good generic pattern:

```bash
ssh <asurite>@sol.asu.edu
```

Then request light setup resources. Two good options are:

For lightweight env creation or setup:

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4
```

For short real compute/setup work:

```bash
salloc -p htc -q public -t 04:00:00 -c 4
```

Why:

- `lightwork` is explicitly documented as a good fit for creating Mamba environments
- `htc` is a good fit when the setup task is short but computationally real

Relevant official sources:

- [Python overview](https://docs.rc.asu.edu/python/)
- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)
- [Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos/)

### 2.2 Load Mamba

Once you are in the shell where you want to work:

```bash
module load mamba/latest
```

Or:

```bash
ml mamba
```

This is the official ASU RC path for Python environment management on Sol.

Relevant official source:

- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)

### 2.3 See what environments already exist

```bash
mamba info --envs
```

This helps confirm:

- whether the environment already exists
- where it lives
- whether you are about to recreate something unnecessarily

Relevant official source:

- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)

### 2.4 Create a new environment

Generic example in your home directory:

```bash
mamba create -n myenv -c conda-forge python=3.12
```

Then activate it:

```bash
source activate myenv
```

If you need to install packages during creation:

```bash
mamba create -n myenv -c conda-forge python=3.12 numpy pandas
```

If you need a reproducible env from a file:

```bash
mamba env create -n myenv --file environment.yml
```

Relevant official source:

- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)

### 2.5 Install more packages into the active environment

After activation:

```bash
mamba install -c conda-forge scipy matplotlib
```

If a package only exists via `pip`, activate the env first and then:

```bash
pip install <package>
```

Practical rule:

- prefer `mamba` when possible
- use `pip` inside the active env only when needed

### 2.6 Typical session pattern we have actually used

This is the compact pattern that has worked well in practice:

```bash
ssh <asurite>@sol.asu.edu
salloc -p lightwork -q public -t 02:00:00 -c 4
module load mamba/latest
mamba create -n myenv -c conda-forge python=3.12
source activate myenv
python -V
which python
python -c "import sys; print(sys.executable)"
```

Then install what you need:

```bash
mamba install -c conda-forge <packages>
```

Or:

```bash
pip install <packages>
```

### 2.7 If the environment contains compiled Python packages

If imports later fail with errors mentioning:

- `libstdc++`
- `CXXABI_*`
- compiled extensions

then a useful runtime fix is:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

This is not an official Mamba creation step, but it is a real cross-project fix
we have had to use when compiled dependencies picked up the wrong shared
libraries at runtime.

### 2.8 Sanity-check the environment before a long run

Before submitting an expensive job:

```bash
python -V
which python
python -c "import numpy; print(numpy.__version__)"
```

And for a project-specific stack, test a few critical imports:

```bash
python - <<'PY'
import numpy
import torch
print("ok")
PY
```

Practical rule:

- never assume the environment is healthy just because activation succeeded

### 2.9 Where environments usually live

ASU RC documents that user environments are usually installed under:

```text
~/.conda/envs
```

You can also create them elsewhere with `-p <path>` if you want a shared or
custom location.

Relevant official source:

- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)

### 2.10 Jupyter note

ASU RC provides a helper command:

```bash
mkjupy <envName>
```

to turn a Mamba env into a Jupyter kernel.

Relevant official source:

- [Helpful Slurm Commands](https://docs.rc.asu.edu/helpful-slurm-commands/)

## 3. Officially useful commands and habits

The docs highlight several Slurm and Sol helper commands.

Useful ones:

- `myjobs`
- `sq`
- `thisjob <jobID>`
- `seff <jobID>`
- `myfairshare`
- `myquota`
- `sbatch`
- `salloc`
- `sinfo`

Relevant official source:

- [Helpful Slurm Commands](https://docs.rc.asu.edu/helpful-slurm-commands/)

Practical habit:

- after a failed or unexpectedly slow run, check `thisjob`, `seff`, and `myfairshare`

## 4. Cross-project issues we have repeatedly hit on Sol

These are not official Sol rules. These are practical issues we have hit across
real work on Sol that are portable to other projects.

### 3.1 Wrong shell when sourcing environment scripts

Common symptom:

- shell setup scripts assume `bash`, but the current shell is `zsh`

Typical fix:

```bash
exec bash -l
```

Then source the env helper again.

Practical rule:

- if a helper script behaves strangely at startup, check the shell first

### 3.2 Wrong Python environment or wrong interpreter

Common symptoms:

- `ModuleNotFoundError`
- packages that are "definitely installed" appear missing
- commands behave differently between login sessions

Root causes:

- wrong env was activated
- no env was activated
- shell startup did not restore the expected interpreter

Typical fixes:

- explicitly activate the intended env every session
- print `which python`
- print `python -V`
- run a short import test before submitting a long job

Practical rule:

- do not assume the right env is still active just because the prompt looks familiar

### 3.3 Local machine and Sol checkout drift

Common symptom:

- code works locally, but Sol behaves like the new file or new default does not exist

Root cause:

- the Sol checkout was not updated after local edits

Typical fixes:

- `git pull` on Sol
- or re-sync the changed files explicitly

Practical rule:

- if Sol still shows an old default or old error signature after a local fix,
  suspect remote checkout drift before debugging the runtime

### 3.4 Walltime / partition mismatch

Common symptom:

- Slurm rejects the submission due to walltime/QoS constraints

Root cause:

- using the wrong partition or requesting a time that queue does not allow

Typical fixes:

- shorten the walltime
- switch to the correct partition/QoS
- reuse a known-good sbatch shape

Practical rule:

- if a job is short, prefer `htc`
- if it is just setup/build/low-intensity work, prefer `lightwork`

### 3.5 Login-node misuse

Common symptom:

- commands get killed, warnings appear, or other users are impacted

Root cause:

- trying to do real compute on a login node

Typical fix:

- move the work behind `salloc` or `sbatch`

Practical rule:

- assume heavy model inference, training, or data processing belongs on a compute node

### 3.6 C++ ABI / shared library mismatch

Common symptoms:

- import errors mentioning `libstdc++`
- missing `CXXABI_*`
- Python package imports fail deep inside compiled dependencies

Root cause:

- Python is using the wrong shared libraries at runtime
- the system library path wins over the active environment's library path

Typical fix:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Practical rule:

- if pure Python imports work but compiled packages fail, inspect `LD_LIBRARY_PATH`

### 3.7 Python package version mismatch

Common symptoms:

- API attributes missing
- package imports succeed but runtime objects do not match expectations
- one library is too new or too old for another

Typical example class:

- `transformers` version too new for the installed `vllm`

Typical fix:

- pin the package set to a known-compatible range
- record the working versions after the fix

Practical rule:

- for ML stacks on Sol, treat package compatibility as part of the runtime, not an afterthought

### 3.8 Accelerator backend mismatches

Common symptoms:

- CUDA kernels try to JIT-compile unexpectedly
- incompatible backend codepaths activate
- runtime errors mention sampler backends, CUDA headers, or unsupported kernels

Root cause:

- a library selected an accelerator backend that exists in principle but is not
  stable for the exact cluster image / package combination

Typical fix:

- explicitly set environment variables to force the known-good backend

Practical rule:

- if a library offers multiple attention/sampler/kernel backends, make the choice explicit

### 3.9 Hidden runtime caps in installed libraries

Common symptom:

- a parameter that looks valid in code is rejected at runtime

Example class:

- a `logprobs` or `top-k` request exceeds what the installed server/runtime allows

Typical fix:

- lower the parameter to the actual supported maximum
- make the default match the real Sol runtime limit

Practical rule:

- confirm the real runtime capability with a smoke test instead of assuming docs
  from another machine apply unchanged

### 3.10 Empty environment variables causing fake path bugs

Common symptom:

- paths unexpectedly resolve to `/token_data`, `/run_summary.txt`, or other root-level paths

Root cause:

- a variable like `OUTDIR` or `RUN_DIR` was empty

Typical fix:

- `echo` the variable before using it
- re-export it explicitly

Practical rule:

- if a path suddenly starts at `/`, check whether a variable expansion vanished

### 3.11 Artifact storage confusion

Common symptom:

- uncertainty about whether results are "saved"
- accidental mixing of code, scratch outputs, and long-term artifacts

Typical fix:

- keep active outputs on scratch
- move or copy important results to a stable location later
- avoid tracking bulk artifacts in git

Practical rule:

- treat code, runtime scratch, and archival copies as three different layers

## 5. Practical Sol workflow that generalizes well

For a new project, a conservative workflow looks like this:

1. Keep code in `~/some_project`
2. Keep large runtime outputs on `/scratch/$USER/...`
3. Build or activate environments from a light setup context, not a login-node abuse pattern
4. Smoke-test on a small allocation before scaling up
5. Record working package versions once the stack is healthy
6. Explicitly set backend env vars when using complex ML runtimes
7. Move finished artifacts off scratch if they matter long-term

## 6. Recommended generic preflight checklist

Before launching an expensive job on Sol, check:

1. Am I on VPN if needed?
2. Am I using a login node only for setup and submission?
3. Is the correct environment activated?
4. Does a 10-second import sanity check pass?
5. Is the project checkout on Sol actually up to date?
6. Am I using the right partition/QoS for the workload?
7. Are my output paths going to scratch, not accidentally to home or root?
8. If this stack uses multiple GPU backends, have I forced the one I trust?
9. If the runtime has hidden caps, have I smoke-tested the chosen settings?

## 7. Recommended generic post-failure checklist

If a Sol job fails, check in this order:

1. Was the failure at submit time, startup time, import time, model-init time, or true runtime?
2. Did Slurm reject the job because of partition/QoS/walltime?
3. Was the correct env active?
4. Is the Sol checkout stale relative to local changes?
5. Is this a compiled-library / `LD_LIBRARY_PATH` issue?
6. Did the runtime silently choose a bad accelerator backend?
7. Did the job fail because a requested setting exceeds what the installed runtime supports?
8. Are the output and log paths actually what you think they are?

## 8. Bottom line

The biggest Sol lessons that generalize across projects are:

- do not compute on login nodes
- use the right partition for the workload
- treat scratch as temporary working storage
- keep environments and compiled-library paths explicit
- assume ML runtime compatibility issues are real until proven otherwise
- smoke-test backend and parameter assumptions on Sol itself
- suspect remote checkout drift early when local and remote behavior disagree
