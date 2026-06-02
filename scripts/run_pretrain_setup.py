# -*- coding: utf-8 -*-
"""토크나이저 재사용과 공통 설정을 확인하는 예시 스크립트."""

from src.experiment_config import BASE_CONFIG, DEFAULT_TOKENIZER_NAME
from src.experiment_utils import (
    encode_pretrain_corpus,
    load_or_train_tokenizer,
    load_pretrain_texts,
)

TOKENIZER_NAME = DEFAULT_TOKENIZER_NAME


def main() -> None:
    train_text, val_text = load_pretrain_texts()

    print("train chars:", len(train_text))
    print("val chars:", len(val_text))
    print(train_text[:200])

    tokenizer, tokenizer_path, was_trained = load_or_train_tokenizer(
        train_text=train_text,
        vocab_size=BASE_CONFIG["vocab_size"],
        tokenizer_name=TOKENIZER_NAME,
    )

    print("tokenizer ready")
    print("tokenizer path:", tokenizer_path)
    print("tokenizer trained now:", was_trained)
    print("vocab_size:", tokenizer.vocab_size)
    print("num_merges:", len(tokenizer.merges))

    train_token_ids, val_token_ids = encode_pretrain_corpus(
        tokenizer,
        train_text,
        val_text,
    )

    print("train token_ids ready")
    print("train num_tokens:", len(train_token_ids))
    print(train_token_ids[:30])

    print("val token_ids ready")
    print("val num_tokens:", len(val_token_ids))
    print(val_token_ids[:30])

    decoded_preview = tokenizer.decode(train_token_ids[:100])
    print(decoded_preview)


if __name__ == "__main__":
    main()
