import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_slurm_scripts_parse_with_bash():
    scripts = [
        ROOT / "slurm" / "mmlongbench_eval_sol.sbatch",
        ROOT / "slurm" / "mmlongbench_eval_sleuth_sol.sbatch",
        ROOT / "slurm" / "mmlongbench_eval_base_sol.sbatch",
        ROOT / "slurm" / "install_mmlongbench_doc.sbatch",
        ROOT / "slurm" / "sleuth_mmlongbench_doc_sol.sbatch",
        ROOT / "slurm" / "gdpo_debug_upstream.sbatch",
        ROOT / "scripts" / "submit_sol.sbatch",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], cwd=ROOT, check=True)
