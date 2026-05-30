# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


# 입력 배치에 대한 모델의 예측 logits와 정답 토큰을 비교해 한 배치의 평균 cross entropy 손실을 계산합니다.
def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    # 입력 배치와 정답 배치를 모델과 같은 device로 옮겨 연산 가능하게 맞춥니다.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # 입력 배치를 모델에 넣어 각 위치의 logits를 계산합니다.
    logits = model(input_batch)  # (batch_size, seq_len, vocab_size)

    # 다음 토큰 예측용 cross entropy loss를 계산합니다.
    batch_size, seq_len, vocab_size = logits.shape
    loss = F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        target_batch.reshape(batch_size * seq_len),
    )
    return loss


# 데이터 로더 전체 또는 일부 배치의 평균 손실을 계산합니다.
def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    # 배치가 없으면 평균 손실을 계산할 수 없으므로 NaN을 반환합니다.
    if len(data_loader) == 0:
        return float("nan")

    # 평가할 배치 수를 정합니다.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        if num_batches <= 0:
            return float("nan")
        num_batches = min(num_batches, len(data_loader))

    # 선택한 배치들의 손실을 누적합니다.
    total_loss = 0.0

    for batch_idx, (input_batch, target_batch) in enumerate(
        data_loader
    ):  # data_loader의 각 원소는 (input_batch, target_batch)이고, enumerate()를 쓰면 앞에 배치 번호가 붙습니다.
        if batch_idx >= num_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return float(total_loss / num_batches)


# 학습 재개에 필요한 모델과 옵티마이저 상태를 저장합니다.
def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,  # 몇 번째 epoch까지 끝냈는지
    global_step: int,  # 전체 배치 업데이트를 몇 번 했는지
    path: str,
) -> None:
    """model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    torch.save(
        {
            "model_state_dic": model.state_dict(),
            "opoptimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


# 저장된 체크포인트에서 모델과 옵티마이저 상태를 복원합니다.
def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dic"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["opoptimizer_state_dict"])

    model.to(
        device
    )  # PyTorch에서는 모델과 입력 텐서가 보통 같은 device에 있어야 연산할 수 있음

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    # 저장된 학습 진행 상태(epoch, global_step)를 복원해 호출자에게 반환합니다.
    return epoch, global_step


# temperature와 top-k를 적용해 다음 토큰을 순차적으로 생성합니다.
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


# 시작 문맥으로 텍스트를 생성하고 사람이 읽을 수 있는 문자열로 출력합니다.
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


# 주기적으로 평가와 샘플 생성을 수행하며 전체 학습 루프를 실행합니다.
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


# epoch별 훈련 및 선택적 검증 손실 곡선을 시각화합니다.
def plot_losses(
    train_losses: list[float], val_losses: list[float] | None = None
) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
