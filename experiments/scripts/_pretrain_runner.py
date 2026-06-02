"""Shared automation for A/B/C pretraining experiment runners."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bpe import BPETokenizer  # noqa: E402
from dataset import create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
from train import train_model  # noqa: E402


DATA_DIR = ROOT / "data"
LM_TRAIN_PATH = DATA_DIR / "nsmc_lm_train.txt"
LM_VAL_PATH = DATA_DIR / "nsmc_lm_val.txt"

BASE_CONFIG = {
    "batch_size": 8,
    "learning_rate": 3e-4,
    "drop_rate": 0.1,
    "context_length": 64,
    "n_layers": 2,
    "emb_dim": 128,
    "n_heads": 4,
    "optimizer": "AdamW",
    "weight_decay": 0.0,
    "num_epochs": 2,
    "qkv_bias": False,
}

SAVE_CONFIG = {
    "log_every_steps": 20,
    "eval_every_steps": 100,
    "save_every_steps": 100,
    "keep_latest": 2,
}

SMOKE_OVERRIDES = {
    "batch_size": 2,
    "context_length": 32,
    "emb_dim": 64,
    "n_layers": 1,
    "num_epochs": 1,
    "log_every_steps": 10,
    "eval_every_steps": 20,
    "save_every_steps": 20,
}


class Tee:
    """stdout/stderr를 console과 log file에 동시에 씁니다."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            try:
                stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def default_output_root() -> str:
    if Path("/content").exists():
        return "/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain"
    return str(ROOT / "local" / "experiment_outputs" / "pretrain")


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    experiment_choices: list[str],
    default_owner: str,
) -> None:
    parser.add_argument("--experiment", choices=experiment_choices, default="all")
    parser.add_argument("--owner", default=default_owner)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output-root", type=Path, default=Path(default_output_root()))
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--force-tokenizer-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Run a smoke-test sized experiment.")
    parser.add_argument("--install", action="store_true", help="Install requirements before running.")
    parser.add_argument("--skip-download", action="store_true", help="Do not run download_data.py.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--eval-iter", type=int, default=10)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--start-context", default="영화")
    parser.add_argument("--train-token-limit", type=int, default=None)
    parser.add_argument("--val-token-limit", type=int, default=None)
    parser.add_argument("--train-char-limit", type=int, default=None)
    parser.add_argument("--val-char-limit", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print the plan without training.")


def selected_experiments(args: argparse.Namespace, experiments: list[dict]) -> list[dict]:
    if args.experiment == "all":
        return experiments
    return [
        item
        for item in experiments
        if item["id"] == args.experiment or item.get("group") == args.experiment
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_command(cmd: list[str], cwd: Path = ROOT) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def install_requirements() -> None:
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def prepare_data(skip_download: bool) -> None:
    required = [LM_TRAIN_PATH, LM_VAL_PATH]
    if all(path.exists() for path in required):
        return
    if skip_download:
        missing = ", ".join(str(path) for path in required if not path.exists())
        raise FileNotFoundError(f"missing data files: {missing}")
    run_command([sys.executable, "download_data.py"])


def get_git_commit() -> str:
    try:
        result = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True)
        return result.strip()
    except Exception:
        return "unknown"


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_env(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": get_git_commit(),
    }


def read_lm_texts() -> tuple[str, str]:
    return (
        LM_TRAIN_PATH.read_text(encoding="utf-8"),
        LM_VAL_PATH.read_text(encoding="utf-8"),
    )


def default_tokenizer_path(vocab_size: int) -> Path:
    return ROOT / "artifacts" / "tokenizers" / f"nsmc_bpe_vocab{vocab_size}_full.json"


def quick_tokenizer_path(output_dir: Path, vocab_size: int) -> Path:
    return output_dir / "tokenizers" / f"nsmc_bpe_vocab{vocab_size}_full.json"


def planned_tokenizer_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.tokenizer_path is not None:
        return args.tokenizer_path
    if args.quick:
        return quick_tokenizer_path(output_dir, args.vocab_size)
    return default_tokenizer_path(args.vocab_size)


def load_or_train_tokenizer(
    args: argparse.Namespace,
    *,
    train_text: str,
    output_dir: Path,
) -> tuple[BPETokenizer, Path]:
    tokenizer_path = planned_tokenizer_path(args, output_dir)

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    if tokenizer_path.exists() and not args.force_tokenizer_train:
        tokenizer.load(tokenizer_path)
        if tokenizer.vocab_size != args.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size mismatch: file={tokenizer.vocab_size}, args={args.vocab_size}"
            )
        print("loaded tokenizer:", tokenizer_path)
        return tokenizer, tokenizer_path

    tokenizer.train(train_text)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print("saved tokenizer:", tokenizer_path)
    return tokenizer, tokenizer_path


def make_config(args: argparse.Namespace, experiment: dict) -> dict:
    config = {**BASE_CONFIG, **SAVE_CONFIG, **experiment.get("overrides", {})}
    config["seed"] = args.seed
    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.quick:
        config.update(SMOKE_OVERRIDES)
        if args.vocab_size == 3000:
            args.vocab_size = 300
        if args.train_char_limit is None:
            args.train_char_limit = 5000
        if args.val_char_limit is None:
            args.val_char_limit = 2000
        args.eval_iter = min(args.eval_iter, 2)
    return config


def limit_text(text: str, char_limit: int | None) -> str:
    if char_limit is None:
        return text
    return text[:char_limit]


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict, total_steps: int):
    if config.get("scheduler") != "warmup_cosine":
        return None

    warmup_steps = int(config.get("warmup_steps") or max(1, total_steps // 10))
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        current_step = step + 1
        if current_step <= warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_output_dirs(args: argparse.Namespace, experiment_id: str) -> dict[str, Path]:
    run_name = f"{experiment_id}_{args.date}_{args.owner}"
    output_dir = args.output_root / run_name
    dirs = {
        "output": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "logs": output_dir / "logs",
        "metrics": output_dir / "metrics",
        "plots": output_dir / "plots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one_experiment(
    args: argparse.Namespace,
    *,
    suite_name: str,
    experiment: dict,
    train_text: str,
    val_text: str,
) -> dict:
    config = make_config(args, experiment)
    dirs = build_output_dirs(args, experiment["id"])
    log_path = dirs["logs"] / f"{experiment['id']}_{args.date}.out"

    with log_path.open("a", encoding="utf-8") as log_file:
        tee_stdout = Tee(sys.stdout, log_file)
        tee_stderr = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            return _run_one_experiment_logged(
                args,
                suite_name=suite_name,
                experiment=experiment,
                config=config,
                dirs=dirs,
                log_path=log_path,
                train_text=train_text,
                val_text=val_text,
            )


def _run_one_experiment_logged(
    args: argparse.Namespace,
    *,
    suite_name: str,
    experiment: dict,
    config: dict,
    dirs: dict[str, Path],
    log_path: Path,
    train_text: str,
    val_text: str,
) -> dict:
    started_at = time.time()
    set_seed(args.seed)
    device = choose_device(args)
    tokenizer_path = planned_tokenizer_path(args, dirs["output"])
    metrics_path = dirs["metrics"] / f"{experiment['id']}_{args.date}_metrics.jsonl"

    if args.dry_run:
        model_config = {
            "vocab_size": args.vocab_size,
            "context_length": config["context_length"],
            "emb_dim": config["emb_dim"],
            "n_heads": config["n_heads"],
            "n_layers": config["n_layers"],
            "drop_rate": config["drop_rate"],
            "qkv_bias": config.get("qkv_bias", False),
        }
        run_config = {
            "suite": suite_name,
            "experiment": experiment,
            "config": config,
            "model_config": model_config,
            "tokenizer_path": str(tokenizer_path),
            "input_limits": {
                "train_char_limit": args.train_char_limit,
                "val_char_limit": args.val_char_limit,
                "train_token_limit": args.train_token_limit,
                "val_token_limit": args.val_token_limit,
            },
            "output_dir": str(dirs["output"]),
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "env": collect_env(device),
            "dry_run": True,
        }
        write_json(dirs["output"] / "run_config.json", run_config)
        summary = {
            "suite": suite_name,
            "experiment_id": experiment["id"],
            "group": experiment.get("group"),
            "change": experiment.get("change"),
            "value": experiment.get("value"),
            "output_dir": str(dirs["output"]),
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "tokenizer_path": str(tokenizer_path),
            "best_checkpoint_path": None,
            "latest_checkpoint_paths": [],
            "best_val_loss": None,
            "final_global_step": None,
            "elapsed_min": (time.time() - started_at) / 60,
            "dry_run": True,
        }
        write_json(dirs["output"] / "summary.json", summary)
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    tokenizer, tokenizer_path = load_or_train_tokenizer(args, train_text=train_text, output_dir=dirs["output"])
    model_train_text = limit_text(train_text, args.train_char_limit)
    model_val_text = limit_text(val_text, args.val_char_limit)
    train_ids = tokenizer.encode(model_train_text)
    val_ids = tokenizer.encode(model_val_text)
    if args.train_token_limit is not None:
        train_ids = train_ids[: args.train_token_limit]
    if args.val_token_limit is not None:
        val_ids = val_ids[: args.val_token_limit]

    if len(train_ids) <= config["context_length"] + 1:
        raise ValueError("training token sequence is too short for context_length.")
    if len(val_ids) <= config["context_length"] + 1:
        raise ValueError("validation token sequence is too short for context_length.")

    stride = args.stride or config["context_length"]
    train_loader = create_dataloader(
        train_ids,
        context_length=config["context_length"],
        batch_size=config["batch_size"],
        stride=stride,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=config["context_length"],
        batch_size=config["batch_size"],
        stride=stride,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": config["context_length"],
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "drop_rate": config["drop_rate"],
        "qkv_bias": config.get("qkv_bias", False),
    }
    model = GPTModel(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    total_steps = max(1, len(train_loader) * config["num_epochs"])
    scheduler = make_scheduler(optimizer, config, total_steps)
    run_config = {
        "suite": suite_name,
        "experiment": experiment,
        "config": config,
        "model_config": model_config,
        "tokenizer_path": str(tokenizer_path),
        "input_limits": {
            "train_char_limit": args.train_char_limit,
            "val_char_limit": args.val_char_limit,
            "train_token_limit": args.train_token_limit,
            "val_token_limit": args.val_token_limit,
        },
        "output_dir": str(dirs["output"]),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "env": collect_env(device),
        "dry_run": args.dry_run,
    }
    write_json(dirs["output"] / "run_config.json", run_config)

    print("== pretrain experiment ==")
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    history: dict = {}
    if not args.dry_run:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=config["num_epochs"],
            eval_freq=config["eval_every_steps"],
            eval_iter=args.eval_iter,
            start_context=args.start_context,
            tokenizer=tokenizer,
            ckpt_dir=dirs["checkpoints"],
            experiment_id=experiment["id"],
            run_date=args.date,
            history=history,
            log_every_steps=config["log_every_steps"],
            save_every_steps=config["save_every_steps"],
            keep_latest=config["keep_latest"],
            metrics_path=metrics_path,
            grad_clip_norm=config.get("grad_clip_norm"),
            lr_scheduler=scheduler,
        )

    elapsed_min = (time.time() - started_at) / 60
    summary = {
        "suite": suite_name,
        "experiment_id": experiment["id"],
        "group": experiment.get("group"),
        "change": experiment.get("change"),
        "value": experiment.get("value"),
        "output_dir": str(dirs["output"]),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "tokenizer_path": str(tokenizer_path),
        "best_checkpoint_path": history.get("best_checkpoint_path"),
        "latest_checkpoint_paths": history.get("latest_checkpoint_paths", []),
        "best_val_loss": history.get("best_val_loss"),
        "final_global_step": history.get("final_global_step"),
        "elapsed_min": elapsed_min,
        "dry_run": args.dry_run,
    }
    write_json(dirs["output"] / "summary.json", summary)
    print("== summary ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_experiment_suite(
    args: argparse.Namespace,
    *,
    suite_name: str,
    experiments: list[dict],
) -> list[dict]:
    if args.install:
        install_requirements()
    if args.dry_run:
        train_text, val_text = "", ""
    else:
        prepare_data(args.skip_download)
        train_text, val_text = read_lm_texts()

    summaries = []
    for experiment in selected_experiments(args, experiments):
        summaries.append(
            run_one_experiment(
                args,
                suite_name=suite_name,
                experiment=experiment,
                train_text=train_text,
                val_text=val_text,
            )
        )

    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))
    return summaries
