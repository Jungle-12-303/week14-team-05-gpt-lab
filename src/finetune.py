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
        self.gpt = gpt_model  # 기존 GPT backbone 저장
        self.num_labels = num_labels  # 분류 클래스 개수 저장(NSMC는 긍정/부정 예측이라 2)
        # dropout과 classifier를 정의. classifier 입력 차원은 gpt_model.config["emb_dim"]
        emb_dim = gpt_model.config["emb_dim"]
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(emb_dim, num_labels)  # 문장 대표 벡터를 num_labels개의 logits로 변환하는 Linear layer

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # GPTModel.forward()는 LM head까지 통과하므로, 분류에서는 hidden state를 직접 만든다.
        x = self.gpt.embedding(input_ids)
        
        # TransformerBlock을 통과해 각 token 위치의 문맥 표현을 만든다.    
        for block in self.gpt.blocks:
            x = block(x)

        # GPT의 마지막 LayerNorm까지 적용한 hidden state를 사용한다.
        x = self.gpt.final_layernorm(x)

        # 오른쪽 padding이 있는 경우 마지막 실제 token이 아니라 pad 위치일 수 있다.
        pooled = x[:, -1, :]  # 단순 구현에서는 sequence의 마지막 위치를 문장 대표 벡터로 사용
        
        # 분류 head에 넣기 전에 dropout으로 과적합을 줄인다.
        pooled = self.dropout(pooled)

        # 문장 대표 벡터를 num_labels개의 class logits로 변환
        logits = self.classifier(pooled)

        # labels가 주어지면 학습/평가용 loss까지 함께 반환
        if labels is not None:
            labels = labels.long()
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits

        # 추론 시에는 logits만 반환
        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        # 모델과 같은 device에서 계산되도록 batch를 이동
        input_ids = input_ids.to(device)
        labels = labels.to(device).long()

        # 이전 batch의 gradient가 누적되지 않도록 초기화
        optimizer.zero_grad()

        # forward에서 classification loss와 logits를 함께 계산
        loss, logits = model(input_ids, labels=labels)

        # loss를 기준으로 gradient를 계산하고 parameter를 갱신
        loss.backward()
        optimizer.step()

        # 마지막 batch 크기가 다를 수 있으므로 sample 수 기준으로 평균을 낸다.
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("train_loader must contain at least one batch.")

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy

def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    raise NotImplementedError("evaluate_sentiment를 구현하세요.")
