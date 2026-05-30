# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # PyTorch의 nn.Module은 객체를 함수처럼 호출하면 내부적으로 forward()를 호출하도록 만들어져 있고, 
    # forward() 함수에서 cross entropy loss 자동 계산
    loss, logits = model(input_batch, targets=target_batch)
    return loss

# 배치 손실을 누적해 데이터로더 평균 손실 계산
def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    
    total_loss = 0.0  # 손실 합산

    # 배치 길이 검증
    if len(data_loader) == 0:
        return float("nan")  # not a number: 정상적인 숫자로 표현할 수 없는 값
    elif num_batches is None:
        # num_batches가 지정되지 않으면 모든 배치를 순회
        num_batches = len(data_loader)
    else:
        # num_batches가 데이터 로더에 있는 배치 개수보다 크면 배치 횟수를 데이터 로더에 있는 총 배치 개수로 맞춘다.
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # 각 배치의 손실을 합산
        else:
            break
    
    return total_loss / num_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """ torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    
    # 저장 당시 device와 현재 실행 device가 다를 수 있으므로 현재 device에 맞춰 로드한다.
    checkpoint = torch.load(path, map_location=device)

    # checkpoint에서 모델 가중치만 꺼내 현재 model 인스턴스에 복원한다.
    model.load_state_dict(checkpoint["model_state_dict"])

    # 이어서 학습할 경우 optimizer 내부 상태도 복원한다.
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # 학습을 중단한 위치부터 재개할 수 있도록 위치 정보를 반환한다.
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    
    return epoch, global_step


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    raise NotImplementedError("generate를 구현하세요.")


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    raise NotImplementedError("generate_and_print_sample을 구현하세요.")


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
