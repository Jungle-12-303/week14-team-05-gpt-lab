"""Colab helper for the D0 sentiment baseline 20-epoch run.

Run this from Colab after cloning the repository, or copy the command below
into a notebook cell:

    !python experiments/scripts/run_d0_20epoch_colab.py

The script mounts Google Drive, checks out the experiment branch, installs
requirements, finds the A0_basic pretrain run on Drive, and writes all D0
artifacts back to Drive.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BRANCH = os.environ.get("GPT_LAB_BRANCH", "experiment/d0-baseline-20epoch")
REPO_URL = os.environ.get(
    "GPT_LAB_REPO_URL",
    "https://github.com/Jungle-12-303/week14-team-05-gpt-lab.git",
)
REPO_DIR = Path(os.environ.get("GPT_LAB_REPO_DIR", "/content/week14-team-05-gpt-lab"))
DRIVE_ROOT = Path(os.environ.get("GPT_LAB_DRIVE_ROOT", "/content/drive/MyDrive/gpt-lab"))
PRETRAIN_ROOT = DRIVE_ROOT / "experiment_outputs" / "pretrain"
SENTIMENT_ROOT = DRIVE_ROOT / "experiment_outputs" / "sentiment"


def run(cmd: list[str | Path], cwd: Path | None = None) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True, env=env)


def mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("This helper is intended for Google Colab.") from exc
    drive.mount("/content/drive")


def prepare_repo() -> None:
    if not REPO_DIR.exists():
        run(["git", "clone", REPO_URL, REPO_DIR])
    run(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR)
    run(["git", "checkout", BRANCH], cwd=REPO_DIR)
    run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=REPO_DIR)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REPO_DIR)


def find_pretrain_run_dir() -> Path:
    explicit = os.environ.get("GPT_LAB_PRETRAIN_RUN_DIR")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"GPT_LAB_PRETRAIN_RUN_DIR does not exist: {path}")

    preferred_names = [
        "A0_basic_20260602_JAEHWAN",
        "A0_basic_20260603_JAEHWAN",
    ]
    for name in preferred_names:
        candidate = PRETRAIN_ROOT / name
        if (candidate / "summary.json").exists() and any((candidate / "checkpoints").glob("*_best.pt")):
            return candidate

    candidates = [
        path
        for path in sorted(PRETRAIN_ROOT.glob("A0_basic_*_JAEHWAN*"), reverse=True)
        if (path / "summary.json").exists() and any((path / "checkpoints").glob("*_best.pt"))
    ]
    if not candidates:
        raise FileNotFoundError(
            "No A0_basic pretrain run found under "
            f"{PRETRAIN_ROOT}. Set GPT_LAB_PRETRAIN_RUN_DIR to the desired run directory."
        )
    return candidates[0]


def main() -> None:
    mount_drive()
    prepare_repo()

    pretrain_run_dir = find_pretrain_run_dir()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SENTIMENT_ROOT / f"D0_baseline_20epoch_{run_stamp}_HYEONGMIN"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("pretrain_run_dir:", pretrain_run_dir)
    print("output_dir:", output_dir)

    run(
        [
            sys.executable,
            "experiments/scripts/run_d_sentiment.py",
            "--experiment",
            "D0",
            "--epochs",
            "20",
            "--pretrain-run-dir",
            pretrain_run_dir,
            "--output-dir",
            output_dir,
            "--device",
            "cuda",
            "--require-cuda",
            "--num-workers",
            "2",
            "--pin-memory",
            "--persistent-workers",
            "--prefetch-factor",
            "2",
            "--enable-tf32",
            "--log-every-steps",
            "20",
        ],
        cwd=REPO_DIR,
    )

    print("D0 20-epoch run complete.")
    print("summary:", output_dir / "summary.json")
    print("metrics:", output_dir / "metrics")
    print("checkpoints:", output_dir / "checkpoints")
    print("logs:", output_dir / "logs")


if __name__ == "__main__":
    main()
