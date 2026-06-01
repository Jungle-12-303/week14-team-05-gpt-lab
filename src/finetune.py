# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """

    # 연속된 공백을 한 칸으로 정리하고 앞뒤 공백을 제거합니다.
    def _clean_text(text: str | None) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    # 탭(\t)으로 구분된 NSMC TSV 파일을 읽어 {"text": ..., "label": ...} 목록으로 변환합니다.
    def _read_nsmc_tsv(path: str | Path) -> list[dict]:
        rows: list[dict] = []
        # newline=""은 csv 모듈이 줄바꿈을 올바르게 처리하도록 돕습니다.
        with open(path, "r", encoding="utf-8", newline="") as f:
            # 이 파일은 쉼표가 아니라 탭(\t)으로 컬럼이 구분됩니다.
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = _clean_text(row.get("document"))
                label = row.get("label")
                # 빈 리뷰이거나 라벨이 0/1이 아니면 건너뜁니다.
                if not text or label not in {"0", "1"}:
                    continue
                # 이후 학습에서 바로 쓰기 좋은 형태로 맞춰 저장합니다.
                rows.append({"text": text, "label": int(label)})
        return rows

    def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
        # 각 샘플을 한 줄에 하나씩 저장하는 JSONL 형식으로 기록합니다.
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 원본 TSV를 읽어 train/test 샘플 목록으로 변환합니다.
    train_rows = _read_nsmc_tsv(train_tsv_path)
    test_rows = _read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    # seed를 고정한 랜덤 셔플로 항상 같은 분할 결과를 재현할 수 있게 합니다.
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    # train 데이터 일부를 validation으로 떼고, 나머지를 실제 학습용으로 사용합니다.
    val_size = max(1, int(len(train_rows) * val_ratio)) if train_rows else 0
    val_rows = train_rows[:val_size]
    train_rows_for_cls = train_rows[val_size:]

    if output_dir is not None:
        # 요청 시 분리된 데이터를 JSONL 파일로도 저장합니다.
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path / "nsmc_sentiment_train.jsonl", train_rows_for_cls)
        _write_jsonl(output_path / "nsmc_sentiment_val.jsonl", val_rows)
        _write_jsonl(output_path / "nsmc_sentiment_test.jsonl", test_rows)

    # 메모리에서 바로 쓸 수 있도록 train/val/test를 함께 반환합니다.
    return train_rows_for_cls, val_rows, test_rows


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        row = self.data[idx]
        text = row["text"]
        label = row["label"]

        token_ids = self.tokenizer.encode(text, add_bos_eos=True)
        token_ids = token_ids[: self.max_length]

        if len(token_ids) < self.max_length:
            pad_length = self.max_length - len(token_ids)
            token_ids.extend([self.pad_id] * pad_length)

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        return input_ids, label


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # 입력 토큰 ID를 GPT가 처리할 수 있는 임베딩 시퀀스로 바꿉니다.
        hidden_states = self.gpt.embedding(input_ids)

        # 각 Transformer block을 통과시키며 문맥이 반영된 hidden state를 만듭니다.
        for block in self.gpt.blocks:
            hidden_states = block(hidden_states)

        # 마지막 정규화까지 적용해 분류에 사용할 최종 hidden state를 얻습니다.
        hidden_states = self.gpt.final_layernorm(hidden_states)
        # 마지막 토큰 위치의 벡터를 문장 대표 표현으로 사용합니다.
        pooled = hidden_states[:, -1, :]
        # 분류 직전에 dropout을 적용해 과적합을 줄입니다.
        pooled = self.dropout(pooled)
        # 문장 표현을 각 라벨의 점수(logits)로 변환합니다.
        logits = self.classifier(pooled)

        if labels is not None:
            # 정답 라벨이 있으면 cross entropy loss를 함께 계산합니다.
            loss = F.cross_entropy(logits, labels)
            return loss, logits

        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    raise NotImplementedError("train_epoch_sentiment를 구현하세요.")


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    raise NotImplementedError("evaluate_sentiment를 구현하세요.")
