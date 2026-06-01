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
    # 입력 토큰 ID (B, T)를 모델과 같은 장치(CPU/GPU)로 이동
    input_batch = input_batch.to(device)
    # 정답 토큰 ID (B, T)를 모델과 같은 장치로 이동
    target_batch = target_batch.to(device)
    # 1.InputEmbedding -> 2.TransformerBlock × N개 -> 3.LayerNorm -> 4.LM Head (Linear) ->5. logits: (B, T, vocab_size)
    # forward pass: (B, T) → (B, T, vocab_size) 각 위치의 다음 토큰 점수
    logits = model(input_batch)
    # 내부적으로 softmax → log → 정답 위치 값 추출 → 전체 평균을 한 번에 계산
    loss = torch.nn.functional.cross_entropy(
        # (B, T, vocab_size) → (B*T, vocab_size): cross_entropy가 요구하는 2D 형태로 변환
        logits.flatten(0, 1),
        # (B, T) → (B*T,): 각 위치의 정답 토큰 ID를 1D로 변환
        target_batch.flatten()
    )
    # 배치 전체의 평균 cross entropy loss (scalar Tensor)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    # num_batches를 지정하지 않으면 data_loader에 있는 모든 배치를 평가합니다.
    elif num_batches is None:
        num_batches = len(data_loader)
    # num_batches가 지정된 경우, 실제 배치 개수보다 많이 돌지 않도록 작은 값을 사용합니다.
    else:
        num_batches = min(num_batches, len(data_loader))
    # 검증/평가용 loss 계산에서는 gradient를 저장할 필요가 없으므로 메모리 사용을 줄입니다.
    with torch.no_grad():
        # data_loader에서 입력 배치와 정답 배치를 하나씩 꺼내며 인덱스 i도 함께 받습니다.
        for i, (input_batch, target_batch) in enumerate(data_loader):
            # 현재 배치 번호가 평가할 배치 수보다 작을 때만 loss를 계산합니다.
            if i < num_batches:
                # 한 배치의 next-token prediction loss를 계산합니다.
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                # Tensor loss를 Python 숫자로 바꿔 누적합니다.
                total_loss += loss.item()
            # 지정한 배치 수만큼 이미 계산했다면 반복을 멈춥니다.
            else:
                break
    # 0 나눗셈 예외처리
    if num_batches == 0:
        return float("nan")
    # 누적한 loss를 계산한 배치 수로 나누어 평균 loss를 반환합니다.
    return total_loss / num_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    # 1. 저장할 정보를 하나의 딕셔너리로 묶는다
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    # 2. torch.save를 사용해서 checkpoint 딕셔너리를 path 위치에 저장한다
    torch.save(checkpoint, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    # 저장된 checkpoint 파일을 읽어옵니다.
    # map_location=device를 사용하면 GPU에서 저장한 파일도 CPU/GPU 원하는 장치로 안전하게 불러올 수 있습니다.
    checkpoint = torch.load(path, map_location=device)

    # checkpoint에 저장된 모델 파라미터를 현재 model 객체에 복원합니다.
    model.load_state_dict(checkpoint["model_state_dict"])

    # optimizer가 전달된 경우에만 optimizer 상태도 복원합니다.
    # 추론만 할 때는 optimizer가 필요 없으므로 None일 수 있습니다.
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 이어서 학습할 수 있도록 저장 당시의 epoch와 global_step을 반환합니다.
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
    """temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    # 정해진 개수만큼 새 단어 조각을 하나씩 만듭니다.
    for _ in range(max_new_tokens):
        # 모델이 한 번에 볼 수 있는 만큼만 최근 문장을 잘라냅니다.
        idx_cond = idx[:, -context_size:]
        # 글을 만들 때는 학습하지 않으므로 계산 기록을 남기지 않습니다.
        with torch.no_grad():
            # 현재 문장을 보고 다음에 올 토큰들의 점수를 계산합니다.
            logits = model(idx_cond)
        # 마지막 위치의 점수만 다음 토큰을 고르는 데 사용합니다.
        logits = logits[:, -1, :]
        # top_k가 있으면 후보를 점수가 높은 몇 개로만 줄입니다.
        if top_k is not None:
            # 각 문장마다 점수가 가장 높은 top_k개를 찾습니다.
            top_logits, _ = torch.topk(logits, top_k)
            # top_k 후보 중 가장 낮은 점수를 기준값으로 삼습니다.
            min_val = top_logits[:, -1]
            # 기준보다 낮은 후보는 뽑히지 못하게 막습니다.
            logits = torch.where(logits < min_val, logits.new_tensor(float("-inf")), logits)
        # temperature가 0보다 크면 확률적으로 다음 토큰을 고릅니다.
        if temperature > 0.0:
            # temperature로 점수 차이를 더 부드럽거나 날카롭게 만듭니다.
            logits = logits / temperature
            # 점수를 토큰별 확률로 바꿉니다.
            probs = torch.softmax(logits, dim=-1)
            # 확률에 따라 다음 토큰 하나를 뽑습니다.
            idx_next = torch.multinomial(probs, num_samples=1)
        # temperature가 0이면 항상 가장 점수가 높은 토큰을 고릅니다.
        else:
            # 가장 가능성이 높은 토큰 ID를 선택합니다.
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # 모든 문장이 종료 토큰을 만들면 생성을 멈춥니다.
        if eos_id is not None and torch.all(idx_next == eos_id):
            # 더 만들 필요가 없으므로 반복을 끝냅니다.
            break
        # 새로 고른 토큰을 기존 문장 뒤에 붙입니다.
        idx = torch.cat((idx, idx_next), dim=1)
    # 처음 입력과 새로 만든 토큰을 합친 결과를 돌려줍니다.
    return idx


# 텍스트를 토큰 ID로 변환하기 위한 유틸리티 함수
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

# 토큰 ID를 텍스트로 변환하기 위한 유틸리티 함수
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


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
    # 샘플 생성 중에는 dropout 같은 학습용 동작을 끕니다.
    model.eval()
    try:
        # 시작 문장을 모델이 이해하는 토큰 ID로 바꾸고 CPU/GPU 장치에 올립니다.
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        # 글을 만드는 동안에는 학습하지 않으므로 gradient 계산을 끕니다.
        with torch.no_grad():
            # 시작 문장 뒤에 새 토큰을 이어 붙여 전체 토큰 ID를 만듭니다.
            token_ids = generate(
                model=model,
                idx=encoded,
                max_new_tokens=max_new_tokens,
                context_size=context_size,
                temperature=temperature,
                top_k=top_k,
            )
        # 생성된 토큰 ID를 사람이 읽을 수 있는 문자열로 되돌립니다.
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # 출력이 한 줄로 보이도록 줄바꿈을 공백으로 바꿔 출력합니다.
        print(decoded_text.replace("\n", " "))
    finally:
        # 샘플 출력이 끝나면 다시 학습 모드로 되돌립니다.
        model.train()


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
    # train loss들을 저장할 리스트 만들기
    train_losses = []

    # 전체 epoch 반복
    for epoch in range(start_epoch, num_epochs):

        # 모델을 학습 모드로 변경
        model.train()

        # 현재 epoch의 loss 누적값 초기화
        epoch_loss = 0

        # train_loader에서 batch를 하나씩 꺼내기
        for input_batch, target_batch in train_loader:

            # optimizer의 이전 gradient 초기화
            optimizer.zero_grad()

            # 현재 batch의 loss 계산
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # loss를 기준으로 gradient 계산
            loss.backward()

            # 계산된 gradient로 model parameter 업데이트
            optimizer.step()

            # global_step 1 증가
            global_step += 1

            # 현재 batch loss를 epoch_loss에 더하기
            epoch_loss += loss.item()

            # eval_freq마다 train/val loss 확인
            if eval_freq > 0 and global_step % eval_freq == 0:

                # 모델을 평가 모드로 변경
                model.eval()

                # train_loader 일부 batch의 평균 loss 계산
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)

                # val_loader 일부 batch의 평균 loss 계산
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

                # 현재 상태 출력
                print(f"epoch {epoch}, step {global_step}, train loss {train_loss}, val loss {val_loss}")

                # 다시 학습 모드로 변경
                model.train()

            # ckpt_freq가 있고, global_step이 ckpt_freq의 배수이면 checkpoint 저장
            if ckpt_freq is not None and global_step % ckpt_freq == 0:

                # checkpoint 파일 경로 정하기
                path = f"checkpoint_step_{global_step}.pt"

                # checkpoint 저장
                save_checkpoint(model, optimizer, epoch, global_step, path)

        # epoch 하나가 끝난 뒤 평균 train loss 계산
        avg_epoch_loss = epoch_loss / len(train_loader)

        # epoch 평균 loss 저장
        train_losses.append(avg_epoch_loss)

        # 샘플 문장 생성해서 출력
        generate_and_print_sample(model, tokenizer, device, start_context)

    # 모든 epoch이 끝나면 train loss 리스트 반환
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
