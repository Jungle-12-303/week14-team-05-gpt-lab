# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # GPTModel에서 F.cross_entropy를 가져옴

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
    """한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss 계산."""


    input_batch = input_batch.to(device)  # tensor데이터 -> cpu / gpu 계산 input/target을 하나의 장치로 맞춘다.
    target_batch = target_batch.to(device)  
    logits = model(input_batch)            # (B,T) -> (B,T,V) 다음 토큰 후보 점수표 "logits 생성"
    
    logits_flat = logits.reshape(
        -1, logits.size(-1)
    )  # (B, T, vocab_size) -> (B*T, vocab_size)
    target_flat = target_batch.reshape(
        -1
    )  # -1은 pytorch한테 차원 변경 맡김 -> 1차원으로 펼친다.

    loss = F.cross_entropy(
        logits_flat, target_flat
    )  # cross_entropy(input:예측 점수 , targets: 정답 번호)

    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:  # float: 평균 loss
    """data_loader의 여러 배치 loss 평균 반환."""
    if len(data_loader) == 0:
        return float("nan")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    batches_seen = 0

    # 평가/검증 단계에서는 gradient가 필요 없으므로 메모리와 연산 절약.
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            # num_batches가 있으면 앞에서부터 지정한 개수만 평가.
            if num_batches is not None and batch_idx >= num_batches:
                break

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            batches_seen += 1

    # 평가 전에 train 모드였던 모델은 다시 train 모드로 복구.
    if was_training:
        model.train()

    if batches_seen == 0:
        return float("nan")
    return total_loss / batches_seen


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """model/optimizer 상태와 학습 위치를 파일로 저장."""
    # 체크포인트는 "가중치"뿐 아니라 이어서 학습할 위치도 함께 저장.
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
    """checkpoint를 읽어 model/optimizer 상태 복원 후 epoch, step 반환."""
    # map_location으로 저장 당시 GPU/CPU와 달라도 현재 device에서 읽기 가능.
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["global_step"]


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """temperature와 top-k 샘플링을 지원해 토큰 이어 생성."""
    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        # GPT는 context_size보다 긴 입력을 한 번에 보지 못하므로 최근 토큰만 유지.
        idx_cond = idx[:, -context_size:]

        # 생성은 학습이 아니므로 gradient 기록 없이 다음 토큰 점수만 계산.
        with torch.no_grad():
            logits = model(idx_cond)

        # 마지막 위치의 logits만 "다음 토큰" 선택에 사용.
        logits = logits[:, -1, :]

        if top_k is not None:
            # top-k 밖의 후보는 -inf로 지워 softmax 확률을 0으로 처리.
            top_k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, top_k)
            kth_best = top_logits[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < kth_best, float("-inf"))

        if temperature == 0.0:
            # temperature 0은 가장 큰 logit만 고르는 greedy decoding으로 처리.
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # temperature가 낮을수록 날카로운 분포, 높을수록 다양한 분포.
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

        # 배치의 모든 샘플이 EOS를 만들면 생성 중단.
        if eos_id is not None and torch.all(idx_next == eos_id):
            break

    if was_training:
        model.train()

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
    """start_context를 토큰화한 뒤 생성 결과를 문자열로 출력."""
    model.eval()

    # tokenizer.encode는 문자열을 토큰 ID 리스트로 변환.
    encoded = tokenizer.encode(start_context)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
    )

    # decode는 토큰 ID 리스트를 다시 사람이 읽는 문자열로 변환.
    print(tokenizer.decode(token_ids.squeeze(0).tolist()))


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
    """사전 학습 루프 실행 후 epoch별 평균 train loss 반환."""
    train_losses: list[float] = []

    model.to(device)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        batches_seen = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # loss에서 각 파라미터가 얼마나 바뀌어야 하는지 gradient 계산.
            loss.backward()
            # optimizer가 gradient를 사용해 실제 모델 파라미터를 한 걸음 업데이트.
            optimizer.step()

            epoch_loss += loss.item()
            batches_seen += 1
            global_step += 1

            if eval_freq > 0 and global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                print(
                    f"step {global_step}: train loss {train_loss:.4f}, "
                    f"val loss {val_loss:.4f}"
                )

            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=f"checkpoint_step_{global_step}.pt",
                )

        avg_epoch_loss = epoch_loss / batches_seen if batches_seen > 0 else float("nan")
        train_losses.append(avg_epoch_loss)

        if tokenizer is not None and start_context:
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context,
                context_size=getattr(model, "config", {}).get("context_length", 256),
            )

    return train_losses


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
