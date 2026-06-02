# A 실험 결과: 사전 학습 안정화

| 항목 | 내용 |
| --- | --- |
| 담당자 | 재환 |
| 담당 영역 | 사전 학습 안정화 |
| 주요 실험 | baseline, warmup, cosine decay, gradient clipping, weight decay |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |

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
| Python version | 3.12.13 |
| PyTorch version | 2.11.0+cpu |
| CUDA version | None |
| git commit | 4dd8f58 |
| seed | 42 |
| 데이터 경로 | `/content/gpt-lab/data/nsmc_lm_train.txt`, `/content/gpt-lab/data/nsmc_lm_val.txt` |

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
| eval_freq / eval_iter |  |

## 4. 실험 결과

| 실험 ID | 변경점 | best val loss | final train loss | 소요 시간 | checkpoint | 결론 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | baseline |  |  |  |  |  |
| A1 | warmup + cosine decay |  |  |  |  |  |
| A2 | gradient clipping |  |  |  |  |  |
| A3 | weight_decay=0.01 |  |  |  |  |  |
| A4 | combined |  |  |  |  |  |

## 5. Loss 기록

| 실험 ID | epoch/step | train loss | val loss | 메모 |
| --- | --- | --- | --- | --- |
| A0 |  |  |  |  |
| A1 |  |  |  |  |
| A2 |  |  |  |  |
| A3 |  |  |  |  |

## 6. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 |
| --- | --- | --- |
|  |  |  |

## 7. 최종 결론

```text
baseline 대비 가장 효과가 있었던 안정화 기법은 무엇이고,
validation loss와 학습 안정성이 어떻게 달라졌는지 적는다.
```
