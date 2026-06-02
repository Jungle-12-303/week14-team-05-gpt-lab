# -*- coding: utf-8 -*-
"""추가 미션 실험용 공통 설정."""

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
}

DEFAULT_TOKENIZER_NAME = "nsmc_bpe"
