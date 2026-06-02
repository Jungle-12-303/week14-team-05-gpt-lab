# A 실험 결과: 사전 학습 안정화

| 항목 | 내용 |
| --- | --- |
| 담당자 | 재환 |
| 담당 영역 | 사전 학습 안정화 |
| 주요 실험 | baseline, warmup, cosine decay, gradient clipping, weight decay |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |
| 실행 스크립트 | `experiments/scripts/run_a_pretrain_stability.py` |

## 1. 실험 목표

사전 학습에서 학습률 스케줄링과 gradient 제어가 validation loss와 학습 안정성에 어떤 영향을 주는지 확인한다.

핵심 질문은 다음과 같다.

- warmup + cosine decay가 baseline보다 validation loss를 낮추는가?
- gradient clipping이 loss spike, NaN, 발산을 줄이는가?
- weight decay가 train/validation loss gap을 줄이는가?

## 2. 공통 환경

| 항목 | 값 |
| --- | --- |
| Colab GPU | T4 런타임 설정, smoke test 실제 실행 device는 CPU (`cuda_available=False`) |
| Colab runtime | GPU, Latest |
| Python version | 3.12.13 |
| PyTorch version | 2.11.0+cpu |
| CUDA version | None |
| git commit | 4dd8f58 |
| seed | 42 |
| 데이터 경로 | `/content/gpt-lab/data/nsmc_lm_train.txt`, `/content/gpt-lab/data/nsmc_lm_val.txt` |
| Google Drive output root | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/{실험ID}_{YYYYMMDD}_JAEHWAN/` |
| 공유 tokenizer 경로 | `artifacts/tokenizers/nsmc_bpe_vocab{vocab_size}_full.json` |
| raw metric JSONL 경로 | `{output_root}/metrics/{실험ID}_{YYYYMMDD}_metrics.jsonl` |
| stdout/stderr log 경로 | `{output_root}/logs/{실험ID}_{YYYYMMDD}.out` |
| latest checkpoint 경로 | `{output_root}/checkpoints/{실험ID}_{YYYYMMDD}_step{global_step}_latest.pt` |
| best checkpoint 경로 | `{output_root}/checkpoints/{실험ID}_{YYYYMMDD}_step{global_step}_best.pt` |

## 3. 고정 설정

| 항목 | 값 |
| --- | --- |
| batch_size | 8 |
| learning_rate | 3e-4 |
| drop_rate | 0.1 |
| context_length | 64 |
| n_layers | 2 |
| emb_dim | 128 |
| n_heads | 4 |
| optimizer | AdamW |
| num_epochs | 2~3 |
| log_every_steps | 20 |
| eval_every_steps | 100 |
| save_every_steps | 100 |
| keep_latest | 2 |

## 4. 실행 규모 기준

| 구분 | 모델 학습 입력 규모 | vocab_size | context_length | 용도 | 결과 반영 |
| --- | ---: | ---: | ---: | --- | --- |
| Smoke | `corpus[:5000]` 수준 | 300 | 32 | 실행 확인 | 공식 비교 제외 |
| Light | `corpus[:500000]` 수준 | 2000 | 64 | 1차 screening | 후보 선별 |
| Basic | `corpus[:1500000]` 수준 | 3000 | 128 | 제출 최소 기준 | 최종 후보 검증 |

위 고정 설정의 `context_length=64`는 1차 screening 기준이다. `A0`는 `--vocab-size 2000 --train-char-limit 500000`로 실행하는 screening baseline이고, `A0_basic`은 `--vocab-size 3000 --train-char-limit 1500000`로 실행하는 Basic 제출 기준 baseline이다. 최종 안정화 후보는 가능하면 Basic 기준으로 재확인하고, Basic 재확인을 하지 못한 경우 그 이유를 결론에 적는다.

## 5. 실험 결과

| 실험 ID | 변경점 | best global_step | best val loss | final train loss | 소요 시간 | best checkpoint | metric JSONL | 결론 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | screening baseline |  |  |  |  |  |  |  |
| A0_basic | Basic submission baseline |  |  |  |  |  |  |  |
| A1 | warmup + cosine decay |  |  |  |  |  |  |  |
| A2 | gradient clipping |  |  |  |  |  |  |  |
| A3 | weight_decay=0.01 |  |  |  |  |  |  |  |
| A4 | combined |  |  |  |  |  |  |  |

## 6. Step metric 기록

| 실험 ID | global_step | epoch | train loss | val loss | checkpoint | 메모 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 |  |  |  |  |  |  |
| A0_basic |  |  |  |  |  |  |
| A1 |  |  |  |  |  |  |
| A2 |  |  |  |  |  |  |
| A3 |  |  |  |  |  |  |
| A4 |  |  |  |  |  |  |

## 7. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 | 보존 로그 |
| --- | --- | --- | --- |
|  |  |  |  |

## 8. 최종 결론

```text
baseline 대비 가장 효과가 있었던 안정화 기법은 무엇이고,
validation loss와 학습 안정성이 어떻게 달라졌는지 적는다.
```
