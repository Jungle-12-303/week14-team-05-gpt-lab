# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch
from bpe import BPETokenizer as bpe_tkzr

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
    """temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""

    for _ in range(max_new_tokens):
        # 모델이 처리할 수 있는 최대 context 길이에 맞춰 최근 토큰만 사용
        idx_cond = idx[:, -context_size:]

        # 생성 단계에서는 학습하지 않으므로 gradient 계산을 기록하지 않는다.
        with torch.no_grad():
            logits = model(idx_cond)

        # 마지막 위치의 logits가 현재 문맥 다음 토큰의 분포를 나타낸다.
        logits = logits[:, -1, :]

        # top-k 밖의 토큰은 softmax 후 확률이 0이 되도록 -inf로 마스킹한다.
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))  # 너무 큰 top-k 값은 가용한 최대값으로 자동 보정
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1, None]  # shape를 (B, 1)로 유지하기 위해 새로운 차원 하나 추가
            negative_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
            logits = torch.where(logits < min_val, negative_inf, logits)  # torch.where(cond, A, B)는 cond의 True/False 여부에 따라 A/B 적용
        
        # temperature가 0 이하이면 확률 샘플링 대신 greedy 선택을 사용
        if temperature <= 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        
        # 단일 prompt 기준: EOS가 생성되면 반복을 중단
        # (참고) idx_next는 tensor라서 batch size가 커지면 바로 if에 쓰기 어렵다.
        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


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
    """start_context를 encode하고 generate 후 decode하여 출력합니다."""

    model.eval()  # PyTorch 모델을 평가 모드로 전환(Dropout 미적용)

    # tokenizer 출력은 list[int]이므로 모델 입력 형태인 (1, T) LongTensor로 변환
    input_token_ids = tokenizer.encode(start_context)
    input_tensor = torch.tensor(input_token_ids, dtype=torch.long).unsqueeze(0).to(device)

    output_tensor = generate(
        model=model,
        idx=input_tensor,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
    )

    # batch 차원을 제거한 뒤 decode가 받을 수 있는 token id list로 변환
    output_token_ids = output_tensor.squeeze(0).tolist()
    output_text = tokenizer.decode(output_token_ids)
    
    print(output_text)


# [헬퍼 함수] train_model 내부에서 사용할 모델 평가용 함수
def evaluate_model(
    model: GPTModel,
    train_loader,
    val_loader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """train/validation loader의 평균 loss를 평가합니다."""

    was_training = model.training  # 현재 모델이 train 모드인지 확인
    model.eval()  # 평가 모드로 전환

    # 계산 그래프 생성을 끄고 train/val loss 평가
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    if was_training:
        model.train()  # 평가 전 train 모드 → 평가 후 train 모드로 복귀
    else:
        model.eval()  # 평가 전 eval 모드  → 평가 후 eval 모드 유지

    return train_loss, val_loss


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
    """사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""

    train_losses = []  # epoch별 평균 train loss를 저장할 리스트 (최종 반환값)

    model.to(device)  # 모델 파라미터를 CPU/GPU 같은 지정된 device로 이동
    model.train()  # 학습 모드로 전환

    # epoch 반복
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0  # 현재 epoch의 batch loss 합계
        num_batches = 0  # batch 개수

        # 훈련 데이터로부터 input/target batch 하나씩 가져오기
        for input_batch, target_batch in train_loader:
            # 이전 batch에서 계산된 gradient 초기화
            # PyTorch는 gradient를 누적하므로 매 batch마다 필요
            optimizer.zero_grad()

            # 현재 batch를 모델에 넣고 cross entropy loss 계산
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()  # loss를 기준으로 각 파라미터의 gradient 계산
            optimizer.step()  # 계산된 gradient로 실제 모델 파라미터 업데이트
            epoch_loss += loss.item()  # 현재 batch loss 값을 epoch loss 합계에 추가
            num_batches += 1  # 현재 epoch 안의 batch 수 증가
            global_step += 1  # 전체 학습 step 수 증가

            # step마다 train/validation loss 평가
            if eval_freq is not None and global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                val_losses.append(val_loss)

                # 현재 epoch, global step, train loss, validation loss 출력
                print(
                    f"Ep {epoch + 1}, Step {global_step}: "
                    f"train loss {train_loss:.3f}, val loss {val_loss:.3f}"
                )
        
        # 한 epoch 동안의 평균 train loss를 계산해서 기록
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)

        # 현재 모델로 샘플 텍스트를 생성해 출력
        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
            context_size=model.config["context_length"],
        )
        model.train()  # 학습 모드로 다시 전환

        # ckpt_freq가 설정되어 있으면 지정한 epoch 간격마다 checkpoint를 저장
        if ckpt_freq is not None and (epoch + 1) % ckpt_freq == 0:
            # 모델 상태, optimizer 상태, 현재 epoch, global step을 파일로 저장
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                path=f"checkpoint_epoch_{epoch + 1}.pt",
            )

    return train_losses


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
