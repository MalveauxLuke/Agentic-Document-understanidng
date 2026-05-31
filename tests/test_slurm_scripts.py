import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_slurm_scripts_parse_with_bash():
    scripts = [
        ROOT / "slurm" / "mmlongbench_eval_sol.sbatch",
        ROOT / "slurm" / "mmlongbench_eval_sleuth_sol.sbatch",
        ROOT / "slurm" / "mmlongbench_eval_sleuth_debug_sol.sbatch",
        ROOT / "slurm" / "mmlongbench_eval_base_sol.sbatch",
        ROOT / "slurm" / "mmlongbench_colpali_cache_sol.sbatch",
        ROOT / "slurm" / "install_mmlongbench_doc.sbatch",
        ROOT / "slurm" / "sleuth_mmlongbench_doc_sol.sbatch",
        ROOT / "slurm" / "gdpo_debug_upstream.sbatch",
        ROOT / "scripts" / "submit_sol.sbatch",
        ROOT / "scripts" / "install_mmlongbench_doc.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], cwd=ROOT, check=True)


def test_eval_wrappers_resolve_repo_root_from_slurm_submit_dir(tmp_path):
    spool_dir = tmp_path / "slurm_spool"
    spool_dir.mkdir()
    fake_main = tmp_path / "repo" / "slurm" / "mmlongbench_eval_sol.sbatch"
    fake_main.parent.mkdir(parents=True)
    fake_main.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$METHOD:$PROJECT_ROOT:$1\" > \"$CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    fake_main.chmod(0o755)

    for wrapper_name, expected_method in [
        ("mmlongbench_eval_sleuth_sol.sbatch", "sleuth"),
        ("mmlongbench_eval_base_sol.sbatch", "base"),
    ]:
        wrapper_copy = spool_dir / wrapper_name
        wrapper_copy.write_text((ROOT / "slurm" / wrapper_name).read_text(encoding="utf-8"), encoding="utf-8")
        wrapper_copy.chmod(0o755)
        capture_path = tmp_path / f"{expected_method}.txt"
        subprocess.run(
            ["bash", str(wrapper_copy), "arg1"],
            check=True,
            env={
                "SLURM_SUBMIT_DIR": str(fake_main.parents[1]),
                "CAPTURE_PATH": str(capture_path),
                "PATH": "/usr/bin:/bin",
            },
        )
        assert capture_path.read_text(encoding="utf-8").strip() == f"{expected_method}:{fake_main.parents[1]}:arg1"
