# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import csv
from datetime import datetime
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

    if val_ratio == 0 or len(train_val_data) < 2:
        val_size = 0
    else:
        val_size = max(1, int(len(train_val_data) * val_ratio))
        val_size = min(val_size, len(train_val_data) - 1)

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

            # 빈 리뷰와 label이 "0" 또는 "1"이 아닌 행은 건너뛴다.
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


def _append_metric_jsonl(path: str | Path | None, record: dict) -> None:
    """metric record를 JSONL 파일에 한 줄 추가합니다."""
    if path is None:
        return

    metric_path = Path(path)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _current_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    """optimizer parameter group별 learning rate를 반환합니다."""
    return [float(group.get("lr", float("nan"))) for group in optimizer.param_groups]


def _format_step_path(
    ckpt_dir: Path,
    experiment_id: str,
    run_date: str,
    global_step: int,
    suffix: str,
) -> Path:
    """step 기반 checkpoint 파일명을 만듭니다."""
    return ckpt_dir / f"{experiment_id}_{run_date}_step{global_step:04d}_{suffix}.pt"


def _cleanup_old_checkpoints(paths: list[Path], keep_latest: int) -> list[Path]:
    """최근 checkpoint path만 남기고 오래된 파일을 삭제합니다."""
    if keep_latest <= 0:
        for path in paths:
            path.unlink(missing_ok=True)
        return []

    keep = paths[-keep_latest:]
    for path in paths[:-keep_latest]:
        path.unlink(missing_ok=True)
    return keep


def _save_sentiment_checkpoint(
    model: "GPTForSequenceClassification",
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str | Path,
    *,
    experiment_id: str,
    best_val_loss: float,
    best_val_acc: float,
    extra_state: dict | None = None,
) -> None:
    """감성 분류 model/optimizer 상태와 학습 위치를 저장합니다."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "experiment_id": experiment_id,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "config": model.gpt.config,
        "extra_state": extra_state or {},
    }
    torch.save(checkpoint, Path(path))


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
        pad_id: int = 0,
    ):
        super().__init__()
        self.gpt = gpt_model  # 기존 GPT backbone 저장
        self.num_labels = num_labels  # 분류 클래스 개수 저장(NSMC는 긍정/부정 예측이라 2)
        self.pad_id = pad_id
        emb_dim = gpt_model.config["emb_dim"]
        
        # GPT hidden state를 분류 label 개수만큼의 logits로 변환
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
        # GPTModel.forward()는 LM head까지 통과하므로, 분류에서는 LM head 전 hidden state를 직접 계산
        x = self.gpt.embedding(input_ids)
        
        # TransformerBlock을 통과해 각 token 위치의 문맥 표현을 만든다.    
        for block in self.gpt.blocks:
            x = block(x)

        # GPT의 마지막 LayerNorm까지 적용한 hidden state를 사용
        x = self.gpt.final_layernorm(x)

        # padding이 아닌 마지막 token 위치의 hidden state를 문장 대표 벡터로 사용한다.
        non_pad_mask = input_ids != self.pad_id
        if not non_pad_mask.any(dim=1).all().item():
            raise ValueError("input_ids must contain at least one non-padding token per example.")

        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        last_token_indices = (positions * non_pad_mask).max(dim=1).values
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = x[batch_indices, last_token_indices]
        
        # 분류 head에 넣기 전에 dropout을 적용해 과적합 위험을 낮춘다.
        pooled = self.dropout(pooled)

        # 문장 대표 벡터를 num_labels개의 class logits로 변환
        logits = self.classifier(pooled)

        # labels가 주어지면 학습/평가용 loss까지 함께 반환
        if labels is not None:
            labels = labels.to(logits.device).long()
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
    """감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    try:
        with torch.no_grad():
            for input_ids, labels in data_loader:
                # 모델과 같은 device에서 계산되도록 batch를 이동
                input_ids = input_ids.to(device)
                labels = labels.to(device).long()

                # 평가에서는 parameter를 갱신하지 않고 loss와 logits만 계산
                loss, logits = model(input_ids, labels=labels)

                # 마지막 batch 크기가 다를 수 있으므로 sample 수 기준으로 평균을 낸다.
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size

                # 가장 큰 logit을 예측 class로 보고 정답 개수를 누적
                preds = logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_examples += batch_size

        if total_examples == 0:
            raise ValueError("data_loader must contain at least one batch.")

        avg_loss = total_loss / total_examples
        accuracy = total_correct / total_examples

        return avg_loss, accuracy
    finally:
        if was_training:
            model.train()


def train_sentiment_model(
    model: GPTForSequenceClassification,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    *,
    experiment_id: str,
    run_date: str,
    ckpt_dir: str | Path,
    metrics_path: str | Path | None = None,
    log_every_steps: int | None = None,
    eval_every_steps: int | None = None,
    save_every_steps: int | None = None,
    keep_latest: int = 2,
    start_epoch: int = 0,
    global_step: int = 0,
    grad_clip_norm: float | None = None,
    extra_state: dict | None = None,
) -> dict:
    """감성 분류 fine-tuning을 step 단위 metric/checkpoint 저장과 함께 실행합니다.

    D0/D2/D3 후보는 validation loss로만 비교하고, test 평가는 runner의 D4 단계에서
    선택된 best checkpoint에 대해서만 수행합니다.
    """
    if num_epochs < start_epoch:
        raise ValueError("num_epochs must be greater than or equal to start_epoch.")
    if len(train_loader) == 0:
        raise ValueError("train_loader must contain at least one batch.")
    if len(val_loader) == 0:
        raise ValueError("val_loader must contain at least one batch.")

    checkpoint_dir = Path(ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_checkpoint_path: Path | None = None
    latest_checkpoint_paths: list[Path] = []
    last_train_loss = float("nan")
    last_train_acc = float("nan")
    last_val_loss = float("nan")
    last_val_acc = float("nan")

    _append_metric_jsonl(
        metrics_path,
        {
            "event": "run_start",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": experiment_id,
            "run_date": run_date,
            "start_epoch": start_epoch,
            "global_step": global_step,
        },
    )

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch_idx, (input_ids, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            loss, logits = model(input_ids, labels=labels)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += batch_size
            global_step += 1

            running_loss = total_loss / total_examples
            running_acc = total_correct / total_examples

            if log_every_steps is not None and log_every_steps > 0 and global_step % log_every_steps == 0:
                _append_metric_jsonl(
                    metrics_path,
                    {
                        "event": "train_step",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment_id": experiment_id,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "train_loss": running_loss,
                        "train_acc": running_acc,
                        "lrs": _current_lrs(optimizer),
                    },
                )

            if eval_every_steps is not None and eval_every_steps > 0 and global_step % eval_every_steps == 0:
                val_loss, val_acc = evaluate_sentiment(model, val_loader, device)
                last_val_loss = val_loss
                last_val_acc = val_acc
                improved = val_loss < best_val_loss
                if improved:
                    old_best = best_checkpoint_path
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_checkpoint_path = _format_step_path(
                        checkpoint_dir,
                        experiment_id,
                        run_date,
                        global_step,
                        "best",
                    )
                    _save_sentiment_checkpoint(
                        model,
                        optimizer,
                        epoch + 1,
                        global_step,
                        best_checkpoint_path,
                        experiment_id=experiment_id,
                        best_val_loss=best_val_loss,
                        best_val_acc=best_val_acc,
                        extra_state=extra_state,
                    )
                    if old_best is not None and old_best != best_checkpoint_path:
                        old_best.unlink(missing_ok=True)

                _append_metric_jsonl(
                    metrics_path,
                    {
                        "event": "eval",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment_id": experiment_id,
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        "improved": improved,
                        "checkpoint_path": str(best_checkpoint_path) if improved else None,
                    },
                )

            if save_every_steps is not None and save_every_steps > 0 and global_step % save_every_steps == 0:
                latest_checkpoint_path = _format_step_path(
                    checkpoint_dir,
                    experiment_id,
                    run_date,
                    global_step,
                    "latest",
                )
                _save_sentiment_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    latest_checkpoint_path,
                    experiment_id=experiment_id,
                    best_val_loss=best_val_loss,
                    best_val_acc=best_val_acc,
                    extra_state=extra_state,
                )
                latest_checkpoint_paths.append(latest_checkpoint_path)
                latest_checkpoint_paths = _cleanup_old_checkpoints(
                    latest_checkpoint_paths,
                    keep_latest=keep_latest,
                )
                _append_metric_jsonl(
                    metrics_path,
                    {
                        "event": "checkpoint",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment_id": experiment_id,
                        "epoch": epoch,
                        "global_step": global_step,
                        "checkpoint_type": "latest",
                        "checkpoint_path": str(latest_checkpoint_path),
                        "kept_latest": [str(path) for path in latest_checkpoint_paths],
                    },
                )

        if total_examples == 0:
            raise ValueError("train_loader must contain at least one batch.")

        last_train_loss = total_loss / total_examples
        last_train_acc = total_correct / total_examples
        last_val_loss, last_val_acc = evaluate_sentiment(model, val_loader, device)
        improved = last_val_loss < best_val_loss
        if improved:
            old_best = best_checkpoint_path
            best_val_loss = last_val_loss
            best_val_acc = last_val_acc
            best_checkpoint_path = _format_step_path(
                checkpoint_dir,
                experiment_id,
                run_date,
                global_step,
                "best",
            )
            _save_sentiment_checkpoint(
                model,
                optimizer,
                epoch + 1,
                global_step,
                best_checkpoint_path,
                experiment_id=experiment_id,
                best_val_loss=best_val_loss,
                best_val_acc=best_val_acc,
                extra_state=extra_state,
            )
            if old_best is not None and old_best != best_checkpoint_path:
                old_best.unlink(missing_ok=True)

        _append_metric_jsonl(
            metrics_path,
            {
                "event": "epoch_end",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "experiment_id": experiment_id,
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": last_train_loss,
                "train_acc": last_train_acc,
                "val_loss": last_val_loss,
                "val_acc": last_val_acc,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
                "improved": improved,
            },
        )

    if best_checkpoint_path is None:
        raise RuntimeError("training finished without saving a best checkpoint.")

    summary = {
        "experiment_id": experiment_id,
        "train_loss": last_train_loss,
        "train_acc": last_train_acc,
        "last_val_loss": last_val_loss,
        "last_val_acc": last_val_acc,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_checkpoint_path": str(best_checkpoint_path),
        "latest_checkpoint_paths": [str(path) for path in latest_checkpoint_paths],
        "final_global_step": global_step,
        "completed_epochs": num_epochs,
    }

    _append_metric_jsonl(
        metrics_path,
        {
            "event": "run_end",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **summary,
        },
    )

    return summary
