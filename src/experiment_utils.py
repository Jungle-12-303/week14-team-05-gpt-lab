# -*- coding: utf-8 -*-
"""로컬/Colab 공용 실험 유틸리티."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

try:
    from .bpe import BPETokenizer
    from .experiment_config import (
        BASE_CONFIG,
        DEFAULT_TOKENIZER_NAME,
        SMOKE_CONFIG,
        TRAIN_CONFIG,
    )
except ImportError:
    from bpe import BPETokenizer
    from experiment_config import (
        BASE_CONFIG,
        DEFAULT_TOKENIZER_NAME,
        SMOKE_CONFIG,
        TRAIN_CONFIG,
    )


def read_text_file(path: str | Path) -> str:
    """UTF-8 텍스트 파일을 읽어 문자열로 반환합니다."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return file.read()


def load_pretrain_texts(
    train_text_path: str | Path = "data/nsmc_lm_train.txt",
    val_text_path: str | Path = "data/nsmc_lm_val.txt",
) -> tuple[str, str]:
    """사전학습용 train/validation 텍스트를 읽어 반환합니다."""
    train_text = read_text_file(train_text_path)
    val_text = read_text_file(val_text_path)
    return train_text, val_text


def get_tokenizer_path(
    vocab_size: int,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    output_dir: str | Path = "tokenizers",
) -> Path:
    """토크나이저 파일 경로를 반환합니다."""
    output_dir = Path(output_dir)
    return output_dir / f"{tokenizer_name}_vocab{vocab_size}.json"


def load_or_train_tokenizer(
    train_text: str,
    vocab_size: int | None = None,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    output_dir: str | Path = "tokenizers",
) -> tuple[BPETokenizer, Path, bool]:
    """저장된 토크나이저를 불러오거나 새로 학습한 뒤 저장합니다."""
    if vocab_size is None:
        vocab_size = BASE_CONFIG["vocab_size"]

    tokenizer_path = get_tokenizer_path(
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
    )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    if tokenizer_path.exists():
        tokenizer.load(tokenizer_path)
        return tokenizer, tokenizer_path, False

    tokenizer.train(train_text)
    tokenizer.save(tokenizer_path)
    return tokenizer, tokenizer_path, True


def encode_pretrain_corpus(
    tokenizer: BPETokenizer,
    train_text: str,
    val_text: str,
) -> tuple[list[int], list[int]]:
    """train/validation 텍스트를 token id 리스트로 변환합니다."""
    train_token_ids = tokenizer.encode(train_text)
    val_token_ids = tokenizer.encode(val_text)
    return train_token_ids, val_token_ids


def get_token_ids_paths(
    vocab_size: int,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    output_dir: str | Path = "tokenizers",
) -> tuple[Path, Path]:
    """train/validation token id 캐시 파일 경로를 반환합니다."""
    tokenizer_path = get_tokenizer_path(
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
    )
    cache_stem = tokenizer_path.with_suffix("")
    return (
        cache_stem.with_name(f"{cache_stem.name}_train_ids.pt"),
        cache_stem.with_name(f"{cache_stem.name}_val_ids.pt"),
    )


def _save_token_ids(token_ids: list[int], path: Path) -> None:
    """token id 리스트를 LongTensor로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(token_ids, dtype=torch.long), path)


def _load_token_ids(path: Path) -> list[int]:
    """저장된 token id tensor를 Python 리스트로 불러옵니다."""
    token_ids = torch.load(path, map_location="cpu")
    if isinstance(token_ids, torch.Tensor):
        return token_ids.tolist()
    return list(token_ids)


def load_or_encode_pretrain_corpus(
    tokenizer: BPETokenizer,
    train_text: str,
    val_text: str,
    vocab_size: int | None = None,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    output_dir: str | Path = "tokenizers",
) -> tuple[list[int], list[int], Path, Path, bool]:
    """저장된 token id를 불러오거나 encode 후 캐시로 저장합니다."""
    if vocab_size is None:
        vocab_size = BASE_CONFIG["vocab_size"]

    train_ids_path, val_ids_path = get_token_ids_paths(
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
    )

    if train_ids_path.exists() and val_ids_path.exists():
        train_token_ids = _load_token_ids(train_ids_path)
        val_token_ids = _load_token_ids(val_ids_path)
        return train_token_ids, val_token_ids, train_ids_path, val_ids_path, False

    train_token_ids, val_token_ids = encode_pretrain_corpus(
        tokenizer,
        train_text,
        val_text,
    )
    _save_token_ids(train_token_ids, train_ids_path)
    _save_token_ids(val_token_ids, val_ids_path)
    return train_token_ids, val_token_ids, train_ids_path, val_ids_path, True


def merged_config(base_config: dict, overrides: dict | None = None) -> dict:
    """기본 설정에 override를 덮어쓴 새 설정 dict를 반환합니다."""
    config = deepcopy(base_config)
    if overrides:
        config.update(overrides)
    return config


def get_train_config(
    use_smoke: bool = False,
    overrides: dict | None = None,
) -> dict:
    """TRAIN_CONFIG 또는 SMOKE_CONFIG를 기반으로 실행 설정을 반환합니다."""
    base_train_config = SMOKE_CONFIG if use_smoke else TRAIN_CONFIG
    return merged_config(base_train_config, overrides)
