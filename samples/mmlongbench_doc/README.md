# MMLongBench-Doc Sample

This folder contains one lightweight sample from
[`yubo2333/MMLongBench-Doc`](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc)
for smoke-testing the real PDF + question path.

Source dataset row:

- dataset: `yubo2333/MMLongBench-Doc`
- split: `train`
- document: `documents/a4f3ced0696009fec3179f493e4f28c4.pdf`
- question: `WHAT IS USCA CASE NUMBER?`
- answer: `21-13199`
- evidence pages: `[1]`

Run mock mode locally:

```bash
python scripts/run_sleuth_baseline.py \
  --pdf samples/mmlongbench_doc/a4f3ced0696009fec3179f493e4f28c4.pdf \
  --question "WHAT IS USCA CASE NUMBER?" \
  --out-dir runs/mmlongbench_doc_mock \
  --mode mock
```

Run SOL mode on Sol:

```bash
python scripts/run_sleuth_baseline.py \
  --pdf samples/mmlongbench_doc/a4f3ced0696009fec3179f493e4f28c4.pdf \
  --question "WHAT IS USCA CASE NUMBER?" \
  --out-dir /scratch/$USER/agenticdocai/runs/mmlongbench_doc_sol \
  --mode sol
```
