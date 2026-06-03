# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import json
import math
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # GPTModel에서 F.cross_entropy를 가져옴
from tqdm.auto import tqdm

# 패키지로 import될 때와 파일 단독 실행 때 모두 GPTModel을 찾기 위한 import 처리.
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

    # input과 target을 모델이 계산하는 장치와 같은 device로 이동.
    input_batch = input_batch.to(device)  # tensor데이터 -> cpu / gpu 계산 input/target을 하나의 장치로 맞춘다.
    target_batch = target_batch.to(device)  

    # 모델에 input을 넣어 각 위치별 다음 토큰 후보 점수(logits) 계산.
    logits = model(input_batch)            # (B,T) -> (B,T,V) 다음 토큰 후보 점수표 "logits 생성"

    # cross_entropy가 받을 수 있도록 batch와 sequence 차원을 하나로 펼침.
    logits_flat = logits.reshape(
        -1, logits.size(-1)
    )  # (B, T, vocab_size) -> (B*T, vocab_size)
    target_flat = target_batch.reshape(
        -1
    )  # -1은 pytorch한테 차원 변경 맡김 -> 1차원으로 펼친다.

    # 펼친 예측 점수와 정답 토큰 ID를 비교해 평균 loss 계산.
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
    # 배치가 하나도 없으면 평균을 낼 수 없으므로 NaN 반환.
    if len(data_loader) == 0:
        return float("nan")

    # loss 측정 중 dropout 같은 학습용 동작을 끄기 위해 현재 모드 저장.
    was_training = model.training
    model.eval()

    # 여러 배치의 loss 합과 실제 계산한 배치 수를 따로 누적.
    total_loss = 0.0
    batches_seen = 0

    # 평가/검증 단계에서는 gradient가 필요 없으므로 메모리와 연산 절약.
    try:
        with torch.no_grad():
            for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
                # num_batches가 있으면 앞에서부터 지정한 개수만 평가.
                if num_batches is not None and batch_idx >= num_batches:
                    break

                # 한 배치 loss를 구한 뒤 Python 숫자로 바꿔 합산.
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
                batches_seen += 1
    finally:
        # 평가 중 예외가 발생해도 train 모드였던 모델은 다시 train 모드로 복구.
        if was_training:
            model.train()

    if batches_seen == 0:
        return float("nan")

    # 누적 loss를 실제 계산한 배치 수로 나눠 평균 반환.
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
    # state_dict는 모델/옵티마이저 내부 상태를 재현 가능한 딕셔너리로 만든 값.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    # checkpoint 딕셔너리를 path 위치에 PyTorch 형식으로 저장.
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

    # 저장된 모델 파라미터를 현재 model 객체에 주입.
    model.load_state_dict(checkpoint["model_state_dict"])

    # 이어서 학습할 때만 optimizer의 momentum 등 내부 상태도 복원.
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 학습 재개 위치를 호출자에게 반환.
    return checkpoint["epoch"], checkpoint["global_step"]


def _append_metric_jsonl(path: str | Path | None, record: dict) -> None:
    """metric record를 JSONL 파일에 한 줄 추가합니다."""
    if path is None:
        return

    metric_path = Path(path)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """optimizer의 첫 번째 parameter group learning rate를 반환합니다."""
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


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
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite number greater than or equal to 0.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be None or a positive integer.")

    # 생성 중에는 dropout을 끄되, 끝나면 원래 train/eval 모드로 복구하기 위해 저장.
    was_training = model.training
    model.eval()

    try:
        # 새 토큰을 max_new_tokens개까지 한 번에 하나씩 생성.
        for _ in range(max_new_tokens):
            # GPT는 context_size보다 긴 입력을 한 번에 보지 못하므로 최근 토큰만 유지.
            idx_cond = idx[:, -context_size:]

            # 생성은 학습이 아니므로 gradient 기록 없이 다음 토큰 점수만 계산.
            with torch.no_grad():
                logits = model(idx_cond)

            # 마지막 위치의 logits만 "다음 토큰" 선택에 사용.
            logits = logits[:, -1, :]

            # top_k가 있으면 확률 후보를 점수가 높은 k개 토큰으로 제한.
            if top_k is not None:
                # top-k 밖의 후보는 -inf로 지워 softmax 확률을 0으로 처리.
                top_k = min(top_k, logits.size(-1))
                top_logits, _ = torch.topk(logits, top_k)
                kth_best = top_logits[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < kth_best, float("-inf"))

            # temperature 0은 greedy, 그 외에는 확률 분포에서 샘플링.
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

        # 원래 입력 토큰과 새로 생성한 토큰이 이어진 전체 시퀀스 반환.
        return idx
    finally:
        if was_training:
            model.train()


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
    # 샘플 생성은 평가 동작으로 실행하되, 호출 전 모드는 끝나면 복구한다.
    was_training = model.training
    model.eval()

    try:
        # tokenizer.encode는 문자열을 토큰 ID 리스트로 변환.
        encoded = tokenizer.encode(start_context)

        # 모델 입력 형식인 (batch_size=1, seq_len) LongTensor로 변환.
        idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

        # 시작 토큰 뒤에 새 토큰을 생성.
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
    finally:
        if was_training:
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
    ckpt_dir: str | Path = "checkpoints",
    experiment_id: str = "EXP",
    run_date: str | None = None,
    history: dict | None = None,
    log_every_steps: int | None = None,
    save_every_steps: int | None = None,
    keep_latest: int = 2,
    metrics_path: str | Path | None = None,
    metrics_callback=None,
    grad_clip_norm: float | None = None,
    lr_scheduler=None,
) -> list[float]:
    """사전 학습 루프 실행 후 epoch별 평균 train loss 반환.

    num_epochs는 전체 목표 epoch 수이고, start_epoch는 다음에 시작할 epoch입니다.
    checkpoint의 epoch 값도 재개할 다음 epoch을 뜻합니다.
    ckpt_freq는 기존 호환용 epoch 단위 checkpoint 저장 주기입니다.
    save_every_steps를 지정하면 step 단위 latest checkpoint를 저장합니다.
    metrics_path를 지정하면 train/eval/checkpoint metric을 JSONL로 기록합니다.
    best checkpoint는 validation loss가 개선될 때 step 단위 파일명으로 저장합니다.
    """
    if num_epochs < start_epoch:
        raise ValueError("num_epochs must be greater than or equal to start_epoch.")

    # epoch가 끝날 때마다 평균 train loss를 저장할 리스트.
    train_losses: list[float] = []
    # 실험 비교용 validation loss는 epoch 단위로 별도 누적한다.
    val_losses: list[float] = []
    best_val_loss = float("inf")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not experiment_id:
        raise ValueError("experiment_id must be a non-empty string.")
    if run_date is None:
        run_date = date.today().strftime("%Y%m%d")
    best_checkpoint_path: Path | None = None
    latest_checkpoint_paths: list[Path] = []

    # 모델 파라미터를 학습에 사용할 CPU/GPU device로 이동.
    model.to(device)

    # start_epoch부터 전체 목표 epoch 전까지 학습 반복.
    epoch_iter = tqdm(
        range(start_epoch, num_epochs),
        desc=f"{experiment_id} epochs",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_iter:
        # 학습 단계에서는 dropout 등 학습용 동작을 켠 상태로 설정.
        model.train()

        # 현재 epoch의 loss 합과 배치 수 초기화.
        epoch_loss = 0.0
        batches_seen = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"{experiment_id} epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        # train_loader에서 input/target 배치를 하나씩 꺼내 학습.
        for input_batch, target_batch in progress_bar:
            # 이전 배치에서 누적된 gradient 초기화.
            optimizer.zero_grad()

            # 현재 배치의 다음 토큰 예측 loss 계산.
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # loss에서 각 파라미터가 얼마나 바뀌어야 하는지 gradient 계산.
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            # optimizer가 gradient를 사용해 실제 모델 파라미터를 한 걸음 업데이트.
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_loss += loss.item()
            batches_seen += 1
            global_step += 1
            progress_bar.set_postfix(
                step=global_step,
                loss=f"{loss.item():.4f}",
                lr=f"{_current_lr(optimizer):.2e}",
            )

            if log_every_steps is not None and log_every_steps > 0 and global_step % log_every_steps == 0:
                record = {
                    "event": "train_step",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "experiment_id": experiment_id,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": float(loss.item()),
                    "learning_rate": _current_lr(optimizer),
                }
                _append_metric_jsonl(metrics_path, record)
                if metrics_callback is not None:
                    metrics_callback(record)
                tqdm.write(
                    f"step {global_step}: train loss {loss.item():.4f}, "
                    f"lr {_current_lr(optimizer):.2e}"
                )

            # eval_freq마다 train/validation loss를 일부 배치 기준으로 점검.
            if eval_freq > 0 and global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                is_best = val_loss < best_val_loss
                checkpoint_path = None
                if is_best:
                    best_val_loss = val_loss
                    if best_checkpoint_path is not None:
                        best_checkpoint_path.unlink(missing_ok=True)
                    best_checkpoint_path = _format_step_path(
                        ckpt_dir,
                        experiment_id,
                        run_date,
                        global_step,
                        "best",
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch=epoch + 1,
                        global_step=global_step,
                        path=str(best_checkpoint_path),
                    )
                    checkpoint_path = str(best_checkpoint_path)

                record = {
                    "event": "eval",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "experiment_id": experiment_id,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "best_val_loss": float(best_val_loss),
                    "is_best": is_best,
                    "checkpoint_path": checkpoint_path,
                    "learning_rate": _current_lr(optimizer),
                }
                _append_metric_jsonl(metrics_path, record)
                if metrics_callback is not None:
                    metrics_callback(record)
                tqdm.write(
                    f"step {global_step}: train loss {train_loss:.4f}, "
                    f"val loss {val_loss:.4f}"
                )

            if save_every_steps is not None and save_every_steps > 0 and global_step % save_every_steps == 0:
                latest_checkpoint_path = _format_step_path(
                    ckpt_dir,
                    experiment_id,
                    run_date,
                    global_step,
                    "latest",
                )
                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    path=str(latest_checkpoint_path),
                )
                latest_checkpoint_paths.append(latest_checkpoint_path)
                latest_checkpoint_paths = _cleanup_old_checkpoints(
                    latest_checkpoint_paths,
                    keep_latest=keep_latest,
                )
                record = {
                    "event": "checkpoint",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "experiment_id": experiment_id,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "checkpoint_type": "latest",
                    "checkpoint_path": str(latest_checkpoint_path),
                }
                _append_metric_jsonl(metrics_path, record)
                if metrics_callback is not None:
                    metrics_callback(record)

        # epoch 하나가 끝난 뒤 평균 loss를 계산해 기록.
        avg_epoch_loss = epoch_loss / batches_seen if batches_seen > 0 else float("nan")
        train_losses.append(avg_epoch_loss)

        # 실험 비교를 위해 epoch 종료 시점의 validation loss를 기록한다.
        val_loss = calc_loss_loader(val_loader, model, device)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_checkpoint_path is not None:
                best_checkpoint_path.unlink(missing_ok=True)
            best_checkpoint_path = _format_step_path(
                ckpt_dir,
                experiment_id,
                run_date,
                global_step,
                "best",
            )
            save_checkpoint(
                model,
                optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                path=str(best_checkpoint_path),
            )

        completed_epoch = epoch + 1
        record = {
            "event": "epoch_end",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": experiment_id,
            "epoch": completed_epoch,
            "global_step": global_step,
            "train_loss": float(avg_epoch_loss),
            "val_loss": float(val_loss),
            "best_val_loss": float(best_val_loss),
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
            "learning_rate": _current_lr(optimizer),
        }
        _append_metric_jsonl(metrics_path, record)
        if metrics_callback is not None:
            metrics_callback(record)

        if ckpt_freq is not None and ckpt_freq > 0 and completed_epoch % ckpt_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch=completed_epoch,
                global_step=global_step,
                path=str(ckpt_dir / f"checkpoint_epoch_{completed_epoch}.pt"),
            )

        epoch_iter.set_postfix(
            train=f"{avg_epoch_loss:.4f}",
            val=f"{val_loss:.4f}",
            best=f"{best_val_loss:.4f}",
        )
        tqdm.write(
            f"epoch {completed_epoch}: train loss {avg_epoch_loss:.4f}, "
            f"val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}"
        )

        # tokenizer와 시작 문장이 있으면 현재 모델의 샘플 생성 결과 확인.
        if tokenizer is not None and start_context:
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context,
                context_size=getattr(model, "config", {}).get("context_length", 256),
            )

    if history is not None:
        history["train_losses"] = train_losses.copy()
        history["val_losses"] = val_losses.copy()
        history["best_val_loss"] = best_val_loss
        history["best_checkpoint_path"] = str(best_checkpoint_path) if best_checkpoint_path else None
        history["latest_checkpoint_paths"] = [str(path) for path in latest_checkpoint_paths]
        history["final_global_step"] = global_step

    # epoch별 평균 train loss 목록 반환.
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
