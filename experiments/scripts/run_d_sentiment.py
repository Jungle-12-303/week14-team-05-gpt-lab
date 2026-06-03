"""Run D sentiment fine-tuning experiments.

The runner trains D0/D2/D3 candidates, selects the lowest validation-loss
checkpoint as D4, and evaluates the test set exactly once for that selected
checkpoint.

Examples:
    python experiments/scripts/run_d_sentiment.py --quick
    python experiments/scripts/run_d_sentiment.py --epochs 2 --batch-size 8
    python experiments/scripts/run_d_sentiment.py --pretrain-checkpoint /content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/{BEST_RUN}/checkpoints/{BEST_CHECKPOINT}.pt

CPU-only debug example:
    python experiments/scripts/run_d_sentiment.py --quick --vocab-size 300 --max-length 32 --emb-dim 32 --n-layers 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bpe import BPETokenizer  # noqa: E402
from finetune import (  # noqa: E402
    GPTForSequenceClassification,
    ReviewSentimentDataset,
    evaluate_sentiment,
    train_sentiment_model,
)
from model import GPTModel  # noqa: E402


DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "local" / "experiment_outputs" / "d_sentiment"


@dataclass
class ExperimentResult:
    experiment_id: str
    change: str
    mode: str
    train_loss: float
    train_acc: float
    best_val_loss: float
    best_val_acc: float
    checkpoint_path: Path
    metrics_path: Path
    final_global_step: int
    elapsed_min: float
    test_loss: float | None = None
    test_acc: float | None = None

    def to_json(self, include_test: bool = True) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "change": self.change,
            "mode": self.mode,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "checkpoint_path": str(self.checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "final_global_step": self.final_global_step,
            "elapsed_min": self.elapsed_min,
            "test_loss": self.test_loss if include_test else None,
            "test_acc": self.test_acc if include_test else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use small data/model for a smoke run.")
    parser.add_argument("--dry-run", action="store_true", help="Write config/output paths without training.")
    parser.add_argument("--install", action="store_true", help="Install requirements before running.")
    parser.add_argument("--skip-download", action="store_true", help="Do not run download_data.py.")
    parser.add_argument(
        "--experiment",
        choices=("all", "D0", "D2", "D3"),
        default="all",
        help="Run all D candidates or only one validation candidate.",
    )
    parser.add_argument("--pretrain-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--pretrain-run-dir",
        type=Path,
        default=None,
        help=(
            "Local A/B/C pretrain run directory containing run_config.json, "
            "summary.json, and checkpoints/. Missing model args are inferred from it."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device. Use cuda for local GPU runs that should fail fast without CUDA.",
    )
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable.")
    parser.add_argument(
        "--local-gpu",
        action="store_true",
        help="Use practical local GPU defaults: cuda device, TF32, pin_memory, and DataLoader workers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=3e-4)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--force-tokenizer-train", action="store_true")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--freeze-blocks", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", help="Pin DataLoader memory for CUDA transfer.")
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs when num_workers > 0.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--enable-tf32", action="store_true", help="Enable TF32 matmul/cuDNN on CUDA.")
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="Run full validation every N train steps. Use 0 to evaluate only at epoch end.",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Save latest checkpoints every N train steps. Use 0 to keep only best checkpoints.",
    )
    parser.add_argument("--keep-latest", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    return parser.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.epochs = min(args.epochs, 1)
    args.batch_size = min(args.batch_size, 2)
    args.train_limit = args.train_limit or 128
    args.val_limit = args.val_limit or 64
    args.test_limit = args.test_limit or 64
    args.max_length = min(args.max_length, 32)
    args.emb_dim = min(args.emb_dim, 32)
    args.n_layers = min(args.n_layers, 1)
    args.n_heads = min(args.n_heads, 4)
    args.vocab_size = min(args.vocab_size, 300)
    args.log_every_steps = min(args.log_every_steps, 10)
    if args.eval_every_steps > 0:
        args.eval_every_steps = min(args.eval_every_steps, 20)
    if args.save_every_steps > 0:
        args.save_every_steps = min(args.save_every_steps, 20)


def cli_has(flag: str) -> bool:
    return flag in sys.argv[1:]


def apply_local_gpu_defaults(args: argparse.Namespace) -> None:
    if not args.local_gpu:
        return
    if args.device == "auto":
        args.device = "cuda"
    args.require_cuda = True
    args.pin_memory = True
    args.enable_tf32 = True
    if args.num_workers == 0 and not cli_has("--num-workers"):
        args.num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    if args.num_workers > 0 and not cli_has("--persistent-workers"):
        args.persistent_workers = True
    if args.num_workers > 0 and args.prefetch_factor is None:
        args.prefetch_factor = 2


def run_command(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True, check=check)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def install_requirements() -> None:
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def prepare_data(skip_download: bool) -> None:
    if skip_download:
        return
    import download_data

    download_data.main()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def limit_rows(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    return rows[:limit]


def label_counts(rows: Iterable[dict]) -> Counter:
    return Counter(int(row["label"]) for row in rows)


def ratio(counts: Counter) -> float:
    total = counts[0] + counts[1]
    return counts[1] / total if total else float("nan")


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def collect_env(device: torch.device) -> dict:
    return {
        "python": sys.version.replace("\n", " "),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "device": str(device),
        "git_commit": get_git_commit(),
    }


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return {"0": value[0], "1": value[1], "total": value[0] + value[1], "positive_ratio": ratio(value)}
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def resolve_local_path(candidate: str | Path | None, *, run_dir: Path | None = None) -> Path | None:
    if candidate is None:
        return None

    raw = Path(candidate)
    if raw.exists():
        return raw.resolve()

    if run_dir is not None:
        by_name = run_dir / "checkpoints" / raw.name
        if by_name.exists():
            return by_name.resolve()
        relative = run_dir / raw
        if relative.exists():
            return relative.resolve()

    text = str(candidate).replace("\\", "/")
    if "/artifacts/tokenizers/" in text:
        local = ROOT / "artifacts" / "tokenizers" / Path(text).name
        if local.exists():
            return local.resolve()
    if "/checkpoints/" in text and run_dir is not None:
        local = run_dir / "checkpoints" / Path(text).name
        if local.exists():
            return local.resolve()

    if not raw.is_absolute():
        local = ROOT / raw
        if local.exists():
            return local.resolve()

    return raw


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_pretrain_run_dir(args: argparse.Namespace) -> None:
    if args.pretrain_run_dir is None:
        return

    run_dir = args.pretrain_run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"pretrain run directory not found: {run_dir}")

    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}
    summary = read_json(summary_path) if summary_path.exists() else {}

    if args.pretrain_checkpoint is None:
        checkpoint_value = summary.get("best_checkpoint_path")
        if checkpoint_value is None:
            checkpoints = sorted((run_dir / "checkpoints").glob("*_best.pt"))
            if len(checkpoints) == 1:
                args.pretrain_checkpoint = checkpoints[0]
            elif not checkpoints:
                raise FileNotFoundError(f"no *_best.pt checkpoint found under {run_dir / 'checkpoints'}")
            else:
                raise ValueError(
                    f"multiple best checkpoints found under {run_dir / 'checkpoints'}; "
                    "pass --pretrain-checkpoint explicitly."
                )
        else:
            args.pretrain_checkpoint = resolve_local_path(checkpoint_value, run_dir=run_dir)
    else:
        args.pretrain_checkpoint = resolve_local_path(args.pretrain_checkpoint, run_dir=run_dir)

    if args.pretrain_checkpoint is None or not args.pretrain_checkpoint.exists():
        raise FileNotFoundError(f"pretrain checkpoint not found: {args.pretrain_checkpoint}")

    model_config = run_config.get("model_config") or {}
    if model_config:
        option_map = {
            "vocab_size": ("--vocab-size", int),
            "context_length": ("--max-length", int),
            "emb_dim": ("--emb-dim", int),
            "n_heads": ("--n-heads", int),
            "n_layers": ("--n-layers", int),
            "drop_rate": ("--drop-rate", float),
        }
        for key, (flag, caster) in option_map.items():
            if key in model_config and not cli_has(flag):
                attr = "max_length" if key == "context_length" else key
                setattr(args, attr, caster(model_config[key]))

    if args.tokenizer_path is None:
        tokenizer_value = run_config.get("tokenizer_path") or summary.get("tokenizer_path")
        tokenizer_path = resolve_local_path(tokenizer_value, run_dir=run_dir)
        if tokenizer_path is not None:
            args.tokenizer_path = tokenizer_path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")


def default_tokenizer_path(args: argparse.Namespace) -> Path:
    if args.quick:
        return args.output_dir.resolve() / "tokenizers" / f"D_tokenizer_vocab{args.vocab_size}_quick.json"
    return ROOT / "artifacts" / "tokenizers" / f"nsmc_bpe_vocab{args.vocab_size}_full.json"


def load_or_train_tokenizer(args: argparse.Namespace) -> BPETokenizer:
    tokenizer_path = args.tokenizer_path or default_tokenizer_path(args)
    lm_train_path = DATA_DIR / "nsmc_lm_train.txt"
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    if tokenizer_path.exists() and not args.force_tokenizer_train:
        tokenizer.load(tokenizer_path)
        if tokenizer.vocab_size != args.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size mismatch: file={tokenizer.vocab_size}, "
                f"args={args.vocab_size}. Use --force-tokenizer-train or another --tokenizer-path."
            )
        print("loaded tokenizer:", tokenizer_path)
        return tokenizer

    corpus = lm_train_path.read_text(encoding="utf-8")
    if args.quick:
        corpus = corpus[:5000]
    tokenizer.train(corpus)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print("saved tokenizer:", tokenizer_path)
    return tokenizer


def load_pretrain_state(checkpoint_path: Path | None, device: torch.device) -> tuple[str, dict | None, dict | None]:
    if checkpoint_path is None:
        return "none (random initialized backbone)", None, None

    resolved_path = checkpoint_path.resolve()
    state = torch.load(resolved_path, map_location=device)
    state_dict = state.get("model_state_dict", state.get("state_dict", state))
    config = state.get("config") if isinstance(state, dict) else None
    return str(resolved_path), state_dict, config


def build_backbone(args: argparse.Namespace, checkpoint_config: dict | None = None) -> GPTModel:
    if checkpoint_config is not None:
        config = dict(checkpoint_config)
        if config["vocab_size"] != args.vocab_size:
            raise ValueError(
                f"pretrain vocab_size={config['vocab_size']} does not match tokenizer vocab_size={args.vocab_size}."
            )
        if args.max_length > config["context_length"]:
            raise ValueError(
                f"max_length={args.max_length} exceeds pretrain context_length={config['context_length']}."
            )
        return GPTModel(config)

    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.max_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False,
    }
    return GPTModel(config)


def apply_pretrain(backbone: GPTModel, state_dict: dict | None) -> None:
    if state_dict is None:
        return

    try:
        backbone.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {
            key.removeprefix("gpt."): value
            for key, value in state_dict.items()
            if key.startswith("gpt.")
        }
        if not stripped:
            raise
        backbone.load_state_dict(stripped)


def make_loader(
    rows: list[dict],
    tokenizer: BPETokenizer,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = ReviewSentimentDataset(
        rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_id=tokenizer.get_pad_id(),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def resolve_device(args: argparse.Namespace) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is False.")
    if args.device == "cuda":
        if not cuda_available:
            raise RuntimeError("requested --device cuda, but CUDA is unavailable.")
        return torch.device("cuda")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if cuda_available else "cpu")


def configure_torch_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def freeze_backbone_part(model: GPTForSequenceClassification, freeze_blocks: int) -> None:
    for param in model.gpt.embedding.parameters():
        param.requires_grad = False
    for block in model.gpt.blocks[:freeze_blocks]:
        for param in block.parameters():
            param.requires_grad = False


def optimizer_for(
    model: GPTForSequenceClassification,
    args: argparse.Namespace,
    mode: str,
) -> torch.optim.Optimizer:
    if mode == "split_lr":
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("gpt."):
                backbone_params.append(param)
            else:
                head_params.append(param)
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.backbone_lr},
                {"params": head_params, "lr": args.classifier_lr},
            ]
        )

    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(params, lr=args.lr)


def build_sentiment_model(
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> tuple[GPTForSequenceClassification, str]:
    pretrain_used, pretrain_state_dict, pretrain_config = load_pretrain_state(args.pretrain_checkpoint, device)
    backbone = build_backbone(args, pretrain_config)
    apply_pretrain(backbone, pretrain_state_dict)
    model = GPTForSequenceClassification(
        backbone,
        num_labels=2,
        drop_rate=args.drop_rate,
        pad_id=tokenizer.get_pad_id(),
    ).to(device)
    return model, pretrain_used


def run_experiment(
    experiment_id: str,
    change: str,
    mode: str,
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    train_rows: list[dict],
    val_rows: list[dict],
    device: torch.device,
    checkpoint_dir: Path,
    metrics_path: Path,
    run_date: str,
    freeze: bool = False,
    split_lr: bool = False,
) -> ExperimentResult:
    start = time.perf_counter()
    model, pretrain_used = build_sentiment_model(args, tokenizer, device)

    if freeze:
        freeze_backbone_part(model, args.freeze_blocks)

    optimizer = optimizer_for(model, args, mode="split_lr" if split_lr else "single_lr")
    train_loader = make_loader(train_rows, tokenizer, args, shuffle=True)
    val_loader = make_loader(val_rows, tokenizer, args, shuffle=False)

    print(f"\n== {experiment_id}: {change} ==")
    print("pretrain checkpoint:", pretrain_used)
    summary = train_sentiment_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        experiment_id=experiment_id,
        run_date=run_date,
        ckpt_dir=checkpoint_dir,
        metrics_path=metrics_path,
        log_every_steps=args.log_every_steps,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
        keep_latest=args.keep_latest,
        grad_clip_norm=args.grad_clip_norm,
        extra_state={
            "change": change,
            "mode": mode,
            "pretrain_checkpoint": pretrain_used,
            "max_length": args.max_length,
            "freeze_blocks": args.freeze_blocks if freeze else 0,
        },
    )
    elapsed_min = (time.perf_counter() - start) / 60.0

    print(
        f"best: val_loss={summary['best_val_loss']:.4f}, "
        f"val_acc={summary['best_val_acc']:.4f}, "
        f"checkpoint={summary['best_checkpoint_path']}, "
        f"elapsed_min={elapsed_min:.2f}"
    )

    return ExperimentResult(
        experiment_id=experiment_id,
        change=change,
        mode=mode,
        train_loss=float(summary["train_loss"]),
        train_acc=float(summary["train_acc"]),
        best_val_loss=float(summary["best_val_loss"]),
        best_val_acc=float(summary["best_val_acc"]),
        checkpoint_path=Path(summary["best_checkpoint_path"]),
        metrics_path=metrics_path,
        final_global_step=int(summary["final_global_step"]),
        elapsed_min=elapsed_min,
    )


def load_candidate_for_eval(
    checkpoint_path: Path,
    tokenizer: BPETokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> GPTForSequenceClassification:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config")
    backbone = build_backbone(args, checkpoint_config)
    model = GPTForSequenceClassification(
        backbone,
        num_labels=2,
        drop_rate=args.drop_rate,
        pad_id=tokenizer.get_pad_id(),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_d4_selected(
    selected: ExperimentResult,
    tokenizer: BPETokenizer,
    test_rows: list[dict],
    args: argparse.Namespace,
    device: torch.device,
    metrics_path: Path,
    run_date: str,
) -> None:
    model = load_candidate_for_eval(selected.checkpoint_path, tokenizer, args, device)
    test_loader = make_loader(test_rows, tokenizer, args, shuffle=False)
    test_loss, test_acc = evaluate_sentiment(model, test_loader, device)
    selected.test_loss = test_loss
    selected.test_acc = test_acc
    append_jsonl(
        metrics_path,
        {
            "event": "final_test",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": "D4",
            "run_date": run_date,
            "selected_experiment_id": selected.experiment_id,
            "checkpoint_path": selected.checkpoint_path,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
    )
    print(
        f"\n== D4 selected test ==\n"
        f"selected={selected.experiment_id}, "
        f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
    )


def write_report(
    args: argparse.Namespace,
    output_dir: Path,
    env: dict,
    counts: dict[str, Counter],
    results: list[ExperimentResult],
    selected: ExperimentResult,
    run_date: str,
    metrics_path: Path,
) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / f"D_{run_date}_HYEONGMIN.md"

    def md_cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    def opt_float(value: float | None) -> str:
        return "" if value is None else f"{value:.4f}"

    lines = [
        "# D experiment log: sentiment fine-tuning",
        "",
        "## Environment",
        "",
        "| item | value |",
        "| --- | --- |",
    ]
    for key, value in env.items():
        lines.append(f"| {md_cell(key)} | {md_cell(value)} |")

    lines.extend(
        [
            f"| seed | {md_cell(args.seed)} |",
            f"| data_dir | {md_cell(DATA_DIR)} |",
            f"| output_dir | {md_cell(output_dir)} |",
            f"| metric_jsonl | {md_cell(metrics_path)} |",
            f"| pretrain_checkpoint | {md_cell(args.pretrain_checkpoint or 'none')} |",
            "",
            "## D1 class imbalance",
            "",
            "| split | label 0 | label 1 | total | positive ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, split_counts in counts.items():
        total = split_counts[0] + split_counts[1]
        lines.append(
            f"| {split} | {split_counts[0]} | {split_counts[1]} | "
            f"{total} | {ratio(split_counts):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Fixed settings",
            "",
            "| item | value |",
            "| --- | --- |",
            f"| max_length | {md_cell(args.max_length)} |",
            f"| batch_size | {md_cell(args.batch_size)} |",
            f"| num_epochs | {md_cell(args.epochs)} |",
            f"| baseline lr | {md_cell(args.lr)} |",
            f"| backbone lr | {md_cell(args.backbone_lr)} |",
            f"| classifier lr | {md_cell(args.classifier_lr)} |",
            f"| freeze range | {md_cell(f'embedding + first {args.freeze_blocks} block(s)')} |",
            f"| log_every_steps | {md_cell(args.log_every_steps)} |",
            f"| eval_every_steps | {md_cell(args.eval_every_steps)} |",
            f"| save_every_steps | {md_cell(args.save_every_steps)} |",
            f"| keep_latest | {md_cell(args.keep_latest)} |",
            f"| quick | {md_cell(args.quick)} |",
            "",
            "## D0/D2/D3 validation candidates",
            "",
            "| experiment_id | change | train loss | train acc | best val loss | val acc | checkpoint | global step | elapsed min | conclusion |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )

    for result in results:
        conclusion = "selected for D4 test" if result.experiment_id == selected.experiment_id else "drop"
        lines.append(
            f"| {md_cell(result.experiment_id)} | {md_cell(result.change)} | "
            f"{result.train_loss:.4f} | {result.train_acc:.4f} | "
            f"{result.best_val_loss:.4f} | {result.best_val_acc:.4f} | "
            f"{md_cell(result.checkpoint_path)} | {result.final_global_step} | "
            f"{result.elapsed_min:.2f} | {md_cell(conclusion)} |"
        )

    lines.extend(
        [
            "",
            "## D4 final test",
            "",
            "| selected candidate | source checkpoint | best val loss | val acc | test loss | test acc |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            (
                f"| {md_cell(selected.experiment_id)} | {md_cell(selected.checkpoint_path)} | "
                f"{selected.best_val_loss:.4f} | {selected.best_val_acc:.4f} | "
                f"{opt_float(selected.test_loss)} | {opt_float(selected.test_acc)} |"
            ),
            "",
            "Test set was evaluated once, only after selecting the lowest validation-loss checkpoint.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_run_config(args: argparse.Namespace, output_dir: Path, run_date: str, metrics_path: Path) -> Path:
    config_path = output_dir / "run_config.json"
    if args.experiment == "all":
        candidate_policy = "D0/D2/D3 use validation only; D4 evaluates test once for selected checkpoint."
    else:
        candidate_policy = f"{args.experiment} only; D4 evaluates test once for that checkpoint."
    write_json(
        config_path,
        {
            "run_date": run_date,
            "args": vars(args),
            "output_dir": output_dir,
            "metrics_path": metrics_path,
            "candidate_policy": candidate_policy,
        },
    )
    return config_path


def selected_experiment_specs(args: argparse.Namespace) -> list[dict]:
    specs = [
        {
            "experiment_id": "D0",
            "change": "sentiment baseline",
            "mode": "baseline",
            "freeze": False,
            "split_lr": False,
        },
        {
            "experiment_id": "D2",
            "change": "freeze embedding and early block(s)",
            "mode": "freeze",
            "freeze": True,
            "split_lr": False,
        },
        {
            "experiment_id": "D3",
            "change": "split learning rates for backbone and classifier",
            "mode": "split_lr",
            "freeze": False,
            "split_lr": True,
        },
    ]
    if args.experiment == "all":
        return specs
    return [spec for spec in specs if spec["experiment_id"] == args.experiment]


def write_summary(
    output_dir: Path,
    env: dict,
    args: argparse.Namespace,
    counts: dict[str, Counter],
    results: list[ExperimentResult],
    selected: ExperimentResult | None,
    report_path: Path | None,
    metrics_path: Path,
    run_date: str,
) -> Path:
    summary_path = output_dir / "summary.json"
    write_json(
        summary_path,
        {
            "run_date": run_date,
            "env": env,
            "args": vars(args),
            "class_counts": counts,
            "candidates": [result.to_json(include_test=False) for result in results],
            "selected": selected.to_json() if selected is not None else None,
            "final_test": {
                "evaluated": selected is not None and selected.test_loss is not None,
                "selected_experiment_id": selected.experiment_id if selected is not None else None,
                "test_loss": selected.test_loss if selected is not None else None,
                "test_acc": selected.test_acc if selected is not None else None,
            },
            "report_path": report_path,
            "metrics_path": metrics_path,
        },
    )
    return summary_path


def main() -> int:
    args = parse_args()
    apply_quick_defaults(args)
    apply_local_gpu_defaults(args)
    apply_pretrain_run_dir(args)

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"
    run_date = datetime.now().strftime("%Y%m%d")
    metrics_path = metrics_dir / f"D_{run_date}_metrics.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.install:
        install_requirements()

    set_seed(args.seed)
    device = resolve_device(args)
    configure_torch_runtime(args, device)
    env = collect_env(device)
    print(json.dumps(env, ensure_ascii=False, indent=2))
    if args.pretrain_run_dir is not None:
        print("pretrain run dir:", args.pretrain_run_dir.resolve())
    print("pretrain checkpoint:", args.pretrain_checkpoint or "none")
    print("tokenizer path:", args.tokenizer_path or default_tokenizer_path(args))
    print(
        "local runtime:",
        {
            "device": str(device),
            "local_gpu": args.local_gpu,
            "pin_memory": args.pin_memory,
            "num_workers": args.num_workers,
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
            "enable_tf32": args.enable_tf32,
        },
    )

    write_run_config(args, output_dir, run_date, metrics_path)
    if args.dry_run:
        summary_path = write_summary(
            output_dir=output_dir,
            env=env,
            args=args,
            counts={},
            results=[],
            selected=None,
            report_path=None,
            metrics_path=metrics_path,
            run_date=run_date,
        )
        print("dry run complete:", summary_path)
        return 0

    prepare_data(args.skip_download)

    train_rows_full = load_jsonl(DATA_DIR / "nsmc_sentiment_train.jsonl")
    val_rows_full = load_jsonl(DATA_DIR / "nsmc_sentiment_val.jsonl")
    test_rows_full = load_jsonl(DATA_DIR / "nsmc_sentiment_test.jsonl")
    counts = {
        "train": label_counts(train_rows_full),
        "validation": label_counts(val_rows_full),
        "test": label_counts(test_rows_full),
    }

    print("\n== D1: class imbalance ==")
    for split, split_counts in counts.items():
        total = split_counts[0] + split_counts[1]
        print(
            f"{split}: label0={split_counts[0]}, label1={split_counts[1]}, "
            f"total={total}, positive_ratio={ratio(split_counts):.4f}"
        )

    train_rows = limit_rows(train_rows_full, args.train_limit)
    val_rows = limit_rows(val_rows_full, args.val_limit)
    test_rows = limit_rows(test_rows_full, args.test_limit)
    tokenizer = load_or_train_tokenizer(args)

    results = [
        run_experiment(
            spec["experiment_id"],
            spec["change"],
            spec["mode"],
            args,
            tokenizer,
            train_rows,
            val_rows,
            device,
            checkpoint_dir,
            metrics_path,
            run_date,
            freeze=spec["freeze"],
            split_lr=spec["split_lr"],
        )
        for spec in selected_experiment_specs(args)
    ]

    selected = min(results, key=lambda item: item.best_val_loss)
    evaluate_d4_selected(
        selected=selected,
        tokenizer=tokenizer,
        test_rows=test_rows,
        args=args,
        device=device,
        metrics_path=metrics_path,
        run_date=run_date,
    )

    report_path = write_report(args, output_dir, env, counts, results, selected, run_date, metrics_path)
    summary_path = write_summary(output_dir, env, args, counts, results, selected, report_path, metrics_path, run_date)
    print("\nreport:", report_path)
    print("summary:", summary_path)
    print("metrics:", metrics_path)
    print("checkpoints:", checkpoint_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
