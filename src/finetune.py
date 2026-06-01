# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
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
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터 생성.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    # TSV 파일에서 document가 비어 있지 않은 행만 {"text", "label"} 형식으로 변환.
    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        rows: list[dict] = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("document") or "").strip()
                label = row.get("label")
                if not text or label is None:
                    continue
                rows.append({"text": text, "label": int(label)})
        return rows

    # seed를 고정한 shuffle로 train/validation 분할을 재현 가능하게 유지.
    train_rows = read_nsmc_tsv(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    val_size = int(len(train_rows) * val_ratio)
    if val_ratio > 0 and len(train_rows) > 0:
        val_size = max(1, val_size)
    val_size = min(val_size, len(train_rows))

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]

    # test 파일이 없으면 빈 test set으로 두고, 있으면 같은 형식으로 읽음.
    test_data = read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        # 필요하면 세 split을 JSON 파일로 저장해 이후 실험에서 재사용 가능하게 함.
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("train.json", train_data),
            ("validation.json", val_data),
            ("test.json", test_data),
        ):
            (output_path / name).write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return train_data, val_data, test_data


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
        # Dataset 전체 샘플 수를 DataLoader가 알 수 있게 반환.
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환."""
        # 리뷰 문자열을 토큰 ID로 바꾼 뒤 모델 입력 길이에 맞게 자르거나 padding.
        item = self.data[idx]
        token_ids = self.tokenizer.encode(item["text"], add_bos_eos=True)
        token_ids = token_ids[: self.max_length]

        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        return input_ids, int(item["label"])


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
        # GPT hidden state의 마지막 토큰 표현을 분류 logits로 바꾸는 head 구성.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits 생성.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # LM head를 거치기 전 hidden state를 직접 구해 문장 분류용 표현으로 사용.
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.final_layernorm(x)

        # causal GPT에서는 마지막 위치가 앞 문맥을 모두 본 대표 벡터 역할.
        pooled = x[:, -1, :]
        logits = self.classifier(self.dropout(pooled))

        if labels is not None:
            # 분류 문제이므로 각 샘플의 class label에 대해 cross entropy 계산.
            loss = nn.functional.cross_entropy(logits, labels)
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
