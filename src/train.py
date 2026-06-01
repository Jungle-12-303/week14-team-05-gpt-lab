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

    with torch.no_grad():
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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

    model_state = checkpoint.get("model_state_dict", checkpoint.get("model_state_dic"))
    optimizer_state = checkpoint.get(
        "optimizer_state_dict", checkpoint.get("opoptimizer_state_dict")
    )

    model.load_state_dict(model_state)

    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

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
    idx: torch.Tensor,  # 현재까지의 입력 토큰 ID 시퀀스. (batch_size, seq_len)
    max_new_tokens: int,  # 새로 생성할 최대 토큰 수
    context_size: int,  # 모델이 한번에 볼 수 있는 최대 문맥 길이
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,  # 문장 종료 토큰 ID
) -> torch.Tensor:  # idx 뒤에 새 토큰이 붙은 최종 시퀀스
    """temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # 현재 문맥 다음에 올 토큰 1개의 분포만 사용

        if top_k is not None:
            # torch.topk()는 (values, indices)를 반환하며, indices는 선택된 값들의 원래 위치입니다.
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[
                :, -1
            ].unsqueeze(-1)  # 기본적으로 내림차순 정렬이기 때문에 top-k 안에서 가장 작은 값을 가리킴

            # torch.where(조건, 조건이_참일때_값, 거짓일때_값)
            logits = torch.where(
                logits < min_val, torch.full_like(logits, float("-inf")), logits
            )

        if temperature > 0.0:
            logits = logits / temperature  # T < 1 : 뾰족, T > 1 : 완만
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # 확률분포를 따라 다음 토큰의 인덱스 하나를 뽑는 것
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and torch.all(idx_next == eos_id):
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


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
    """start_context를 encode하고 generate 후 decode하여 출력합니다."""
    # 모델을 현재 학습 모드에서 평가 모드로 전환합니다.
    was_training = model.training  # 현재 모델의 상태를 저장
    model.eval()

    # start_context 문자열을 tokenizer로 토큰 ID 시퀀스로 인코딩합니다.
    try:
        start_ids = tokenizer.encode(start_context, add_bos_eos=False)
    except TypeError:
        start_ids = tokenizer.encode(start_context)

    # 인코딩한 토큰을 텐서로 만들고 device로 옮깁니다.
    idx = (
        torch.tensor(start_ids, dtype=torch.long).unsqueeze(0).to(device)
    )  # (batch_size, seq_len)

    # torch.no_grad() 안에서 generate()를 호출해 새 토큰을 생성합니다.
    with torch.no_grad():
        out = generate(
            model,
            idx,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
        )

    # 생성된 토큰 ID 시퀀스를 CPU로 가져오고 1차원으로 펼칩니다.
    out_ids = (
        out.squeeze(0).detach().cpu().tolist()
    )  # detach() : 텐서를 계산 그래프에서 분리

    # tokenizer로 토큰 ID 시퀀스를 다시 문자열로 디코딩합니다.
    try:
        text = tokenizer.decode(out_ids, skip_special=True)
    except TypeError:
        text = tokenizer.decode(out_ids)

    # 디코딩한 텍스트를 출력합니다.
    print(text)

    # 원래 모델이 학습 모드였다면 다시 train 모드로 되돌립니다.
    if was_training:
        model.train()


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
    """사전 학습 루프를 실행하고 epoch별 train loss 리스트를 반환합니다."""
    # epoch별 훈련 손실을 기록합니다.
    train_losses = []

    # 메인 훈련 루프 - epoch 단위로 전체 학습 데이터를 반복
    for epoch in range(start_epoch, num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            # 이전 배치에서 계산된 gradient를 초기화합니다.
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 현재 배치 손실로부터 gradient를 계산합니다.
            loss.backward()
            # 계산한 gradient를 사용해 모델 파라미터를 업데이트합니다.
            optimizer.step()
            global_step += 1

            # 지정한 주기마다 훈련/검증 손실을 평가합니다.
            if eval_freq > 0 and global_step % eval_freq == 0:
                model.eval()
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
                model.train()

            # 지정한 주기마다 체크포인트를 저장합니다.
            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                ckpt_path = f"checkpoint_step_{global_step:06d}.pt"
                save_checkpoint(model, optimizer, epoch, global_step, ckpt_path)

        # epoch 종료 시점의 훈련 손실을 평가 모드에서 기록합니다.
        model.eval()
        train_losses.append(calc_loss_loader(train_loader, model, device, eval_iter))
        # epoch이 끝날 때마다 현재 모델로 샘플 텍스트를 생성해 봅니다.
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses


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
