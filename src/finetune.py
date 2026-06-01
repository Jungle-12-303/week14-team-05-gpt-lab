# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
import re
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
    val_ratio: float = 0.08, # 학습 데이터 중 validation 데이터로 떼어낼 비율
    seed: int = 42, # 데이터 섞을 때 사용하는 랜덤 시드
    output_dir: str | Path | None = None, # 가공한 데이터셋을 파일로 저장할 디렉토리
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def clean_text(text: str | None) -> str:
        # NSMC 리뷰에는 빈 문자열이나 여러 공백이 섞일 수 있으므로 한 칸 공백으로 정리합니다.
        if text is None:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        # NSMC 원본 TSV는 id, document, label 컬럼을 갖습니다.
        # 모델에는 id가 필요 없으므로 리뷰 텍스트와 0/1 label만 남깁니다.
        rows: list[dict] = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = clean_text(row.get("document"))
                label = row.get("label")

                # 빈 리뷰나 감성 라벨이 아닌 값은 학습 데이터에서 제외합니다.
                if not text or label not in {"0", "1"}:
                    continue

                rows.append({"text": text, "label": int(label)})
        return rows

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        # 한 줄에 샘플 하나씩 저장하면 나중에 큰 데이터도 줄 단위로 읽기 쉽습니다.
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_rows = read_nsmc_tsv(train_tsv_path)
    test_rows = read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    # seed를 고정한 전용 Random 객체를 사용해 전역 random 상태를 건드리지 않습니다.
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    # val_ratio만큼 train TSV에서 validation 데이터를 분리합니다.
    # 데이터가 있고 비율도 양수라면 최소 1개는 validation으로 보냅니다.
    if train_rows and val_ratio > 0:
        val_size = max(1, int(len(train_rows) * val_ratio))
        val_size = min(val_size, len(train_rows))
    else:
        val_size = 0

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = test_rows

    if output_dir is not None:
        # output_dir이 주어지면 재사용하기 쉬운 JSONL 파일 3개로 저장합니다.
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_path / "nsmc_sentiment_train.jsonl", train_data)
        write_jsonl(output_path / "nsmc_sentiment_val.jsonl", val_data)
        write_jsonl(output_path / "nsmc_sentiment_test.jsonl", test_data)

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
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        # data의 한 샘플은 {"text": 리뷰 문자열, "label": 0 또는 1} 형태입니다.
        sample = self.data[idx]
        text = sample["text"]
        label = int(sample["label"])

        # 분류에서는 문장 전체의 시작/끝을 알려주기 위해 BOS/EOS 토큰을 함께 붙입니다.
        token_ids = self.tokenizer.encode(text, add_bos_eos=True)

        # 모델 입력 길이를 넘는 리뷰는 max_length까지만 사용합니다.
        token_ids = token_ids[: self.max_length]

        # DataLoader가 여러 샘플을 하나의 배치로 묶을 수 있도록 길이를 고정합니다.
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_id] * padding_length

        # PyTorch 모델 입력으로 바로 사용할 수 있게 정수 Tensor로 변환합니다.
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
        # GPT의 hidden state 차원은 config["emb_dim"]입니다.
        hidden_size = gpt_model.config["emb_dim"]
        # 분류 head 앞에 dropout을 두어 fine-tuning 중 과적합을 줄입니다.
        self.dropout = nn.Dropout(drop_rate)
        # 문장 대표 벡터를 긍정/부정 같은 class 개수만큼의 logits로 바꿉니다.
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # GPTModel.forward()는 언어모델용 vocab logits를 반환하므로,
        # 감성 분류에서는 backbone의 hidden state까지만 직접 계산합니다.
        x = self.gpt.embedding(input_ids)

        # 사전학습 때와 같은 Transformer block들을 통과시켜 문맥 표현을 만듭니다.
        for block in self.gpt.blocks:
            x = block(x)

        # GPT의 마지막 LayerNorm까지 적용한 값이 분류에 사용할 hidden state입니다.
        hidden_states = self.gpt.final_layernorm(x)

        # GPT는 causal 구조라 마지막 위치가 앞 토큰들의 정보를 모두 볼 수 있습니다.
        # 그래서 마지막 토큰 hidden state를 문장 대표 벡터로 사용합니다.
        pooled = hidden_states[:, -1, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if labels is not None:
            # labels가 주어지면 분류 loss와 logits를 함께 반환해 학습 루프에서 바로 씁니다.
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits

        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    # dropout과 gradient 계산이 켜지도록 모델을 학습 모드로 전환합니다.
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        # 배치 데이터를 모델과 같은 CPU/GPU 장치로 옮깁니다.
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # 이전 배치에서 계산된 gradient가 누적되지 않도록 초기화합니다.
        optimizer.zero_grad()

        # labels를 함께 넘기면 모델이 cross entropy loss와 logits를 같이 반환합니다.
        loss, logits = model(input_ids, labels=labels)

        # loss를 기준으로 gradient를 계산하고 optimizer로 파라미터를 업데이트합니다.
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        # 배치 평균 loss에 배치 크기를 곱해 샘플 수 기준 평균을 낼 수 있게 누적합니다.
        total_loss += loss.item() * batch_size

        # 가장 큰 logit을 가진 class를 예측값으로 보고 정답 개수를 셉니다.
        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    # 비어 있는 loader가 들어오면 0으로 나누지 않고 NaN을 반환합니다.
    if total_examples == 0:
        return float("nan"), float("nan")

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    # dropout 같은 학습 전용 동작을 끄기 위해 평가 모드로 전환합니다.
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # 평가에서는 gradient가 필요 없으므로 메모리 사용과 계산 부담을 줄입니다.
    with torch.no_grad():
        for input_ids, labels in data_loader:
            # 배치 데이터를 모델과 같은 CPU/GPU 장치로 옮깁니다.
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # labels를 함께 넘겨 평가 loss와 분류 logits를 한 번에 얻습니다.
            loss, logits = model(input_ids, labels=labels)

            batch_size = labels.size(0)
            # 배치 평균 loss를 샘플 수만큼 가중해 전체 평균을 정확히 계산합니다.
            total_loss += loss.item() * batch_size

            # 가장 높은 logit의 class를 예측값으로 사용해 accuracy를 계산합니다.
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

    # 비어 있는 loader가 들어오면 0으로 나누지 않고 NaN을 반환합니다.
    if total_examples == 0:
        return float("nan"), float("nan")

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy
