# -*- coding: utf-8 -*-
"""Colab용 A0 baseline 사전학습 smoke test.

실행:
    python experiments/pretrain_smoke.py
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import download_data
from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel
from src.train import calc_loss_loader, train_model


BASE_CONFIG = {
    "vocab_size": 3000,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TRAIN_CONFIG = {
    "seed": 42,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
    "num_epochs": 2,
    "eval_freq": 100,
    "eval_iter": 10,
    "start_context": "영화",
}

SMOKE_CONFIG = {
    **TRAIN_CONFIG,
    "batch_size": 2,
    "num_epochs": 1,
    "eval_freq": 20,
    "eval_iter": 2,
    "train_char_limit": 25_000,
    "val_char_limit": 8_000,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_data() -> tuple[Path, Path]:
    train_path = ROOT / "data" / "nsmc_lm_train.txt"
    val_path = ROOT / "data" / "nsmc_lm_val.txt"
    if not train_path.exists() or not val_path.exists():
        print("LM 데이터가 없어 download_data.py를 실행합니다.")
        download_data.main()
    return train_path, val_path


def finite_or_raise(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite: {value}")


def main() -> None:
    started_at = time.time()
    set_seed(SMOKE_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    print("BASE_CONFIG:", BASE_CONFIG)
    print("TRAIN_CONFIG:", TRAIN_CONFIG)
    print("SMOKE_CONFIG:", SMOKE_CONFIG)

    train_path, val_path = ensure_data()
    train_text = train_path.read_text(encoding="utf-8")[: SMOKE_CONFIG["train_char_limit"]]
    val_text = val_path.read_text(encoding="utf-8")[: SMOKE_CONFIG["val_char_limit"]]

    tokenizer_path = ROOT / "data" / f"vocab_bpe_{BASE_CONFIG['vocab_size']}_smoke.json"
    tokenizer = BPETokenizer(vocab_size=BASE_CONFIG["vocab_size"])
    if tokenizer_path.exists():
        tokenizer.load(tokenizer_path)
        print("loaded tokenizer:", tokenizer_path)
    else:
        tokenizer.train(train_text)
        tokenizer.save(tokenizer_path)
        print("saved tokenizer:", tokenizer_path)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    train_loader = create_dataloader(
        train_ids,
        context_length=BASE_CONFIG["context_length"],
        batch_size=SMOKE_CONFIG["batch_size"],
        stride=BASE_CONFIG["context_length"],
        drop_last=True,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=BASE_CONFIG["context_length"],
        batch_size=SMOKE_CONFIG["batch_size"],
        stride=BASE_CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
    )

    model = GPTModel(BASE_CONFIG)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SMOKE_CONFIG["learning_rate"],
        weight_decay=SMOKE_CONFIG["weight_decay"],
    )

    initial_val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=SMOKE_CONFIG["eval_iter"]
    )
    finite_or_raise("initial_val_loss", initial_val_loss)
    print(f"initial val loss: {initial_val_loss:.4f}")

    history: dict = {}
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=SMOKE_CONFIG["num_epochs"],
            eval_freq=SMOKE_CONFIG["eval_freq"],
            eval_iter=SMOKE_CONFIG["eval_iter"],
            start_context=SMOKE_CONFIG["start_context"],
            tokenizer=tokenizer,
            ckpt_dir=ROOT / "checkpoints",
            experiment_id="A0_smoke",
            history=history,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA OOM during smoke test") from exc

    for idx, loss in enumerate(history.get("train_losses", []), start=1):
        finite_or_raise(f"train_loss_epoch_{idx}", loss)
    for idx, loss in enumerate(history.get("val_losses", []), start=1):
        finite_or_raise(f"val_loss_epoch_{idx}", loss)

    if torch.cuda.is_available():
        print("cuda max memory allocated MB:", torch.cuda.max_memory_allocated() / 1024**2)

    print("best val loss:", history.get("best_val_loss"))
    print("best checkpoint:", history.get("best_checkpoint_path"))
    print(f"elapsed sec: {time.time() - started_at:.1f}")
    print("SMOKE TEST PASSED: no error, no NaN/Inf, no OOM")


if __name__ == "__main__":
    main()
