# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import csv
import json
import random

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
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1).")
    
    train_val_data = _read_tsv(train_tsv_path)

    random_num_generator = random.Random(seed)
    random_num_generator.shuffle(train_val_data)

    val_size = int(len(train_val_data) * val_ratio)
    val_data = train_val_data[:val_size]
    train_data = train_val_data[val_size:]

    test_data = _read_tsv(test_tsv_path) if test_tsv_path is not None else []

    # output_dir가 주어졌을 때만 JSONL 파일을 저장
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path / "nsmc_sentiment_train.jsonl", train_data)
        _write_jsonl(output_path / "nsmc_sentiment_val.jsonl", val_data)
        _write_jsonl(output_path / "nsmc_sentiment_test.jsonl", test_data)

    return train_data, val_data, test_data


# [헬퍼 함수] NSMC TSV 파일을 읽어서 텍스트와 레이블을 붙인 리스트로 변환
def _read_tsv(path: str | Path) -> list[dict]:
    data = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("document") or "").strip()
            label = row.get("label")

            # 빈 리뷰와 label이 "0" 또는 "1"이 아닌 행은 패스
            if not text:
                continue
            if label not in {"0", "1"}:
                continue

            data.append({"text": text, "label": int(label)})
    return data


# [헬퍼 함수] dict 리스트를 JSONL 파일로 저장
def _write_jsonl(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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
        
        # self.data[idx]에서 {"text": ..., "label": ...} 하나를 꺼냄
        item = self.data[idx]

        # text를 tokenizer.encode()로 token id list로 변환
        token_ids = self.tokenizer.encode(item["text"])

        # max_length보다 길면 자름
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            # max_length보다 짧으면 pad_id로 채움
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        # token id list를 torch.long tensor로 변환
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # label은 int로 변환해서 반환
        label = int(item["label"])

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
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        raise NotImplementedError("GPTForSequenceClassification.__init__을 구현하세요.")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        raise NotImplementedError("GPTForSequenceClassification.forward를 구현하세요.")


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
