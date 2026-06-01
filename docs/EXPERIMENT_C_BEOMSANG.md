# C 실험 결과: 모델 구조 하이퍼파라미터 탐색

| 항목 | 내용 |
| --- | --- |
| 담당자 | 범상 |
| 담당 영역 | 하이퍼파라미터 탐색 2 |
| 주요 실험 | context_length, n_layers, emb_dim |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |

## 1. 실험 목표

모델 구조를 바꾸었을 때 validation loss, 학습 시간, GPU 메모리 사용량이 어떻게 변하는지 확인한다. 시간이 제한되어 있으므로 context_length 또는 n_layers 비교를 필수로 수행하고, emb_dim 비교는 가능한 범위에서 진행한다.

핵심 질문은 다음과 같다.

- context_length 64와 128 중 어느 쪽이 validation loss에 유리한가?
- n_layers를 늘렸을 때 성능 개선이 학습 시간 증가를 정당화하는가?
- emb_dim을 키우면 모델 성능이 좋아지는가, 아니면 과도하게 느려지는가?

## 2. 공통 환경

| 항목 | 값 |
| --- | --- |
| Colab GPU |  |
| Python version |  |
| PyTorch version |  |
| CUDA version |  |
| git commit |  |
| seed | 42 |
| 데이터 경로 |  |

## 3. 고정 설정

| 항목 | 값 |
| --- | --- |
| batch_size | 8 |
| learning_rate | 3e-4 |
| drop_rate | 0.1 |
| n_heads | 4 |
| optimizer | AdamW |
| num_epochs | 2~3 |
| eval_freq / eval_iter |  |

## 4. 실험 결과

| 실험 ID | 변경 변수 | 값 | best val loss | final train loss | 소요 시간 | 메모리/OOM | 결론 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | context_length | 64 |  |  |  |  |  |
| C1 | context_length | 128 |  |  |  |  |  |
| C2 | n_layers | 1 |  |  |  |  |  |
| C2 | n_layers | 2 |  |  |  |  |  |
| C2 | n_layers | 4 |  |  |  |  |  |
| C3 | emb_dim | 64 |  |  |  |  |  |
| C3 | emb_dim | 128 |  |  |  |  |  |
| C3 | emb_dim | 192 |  |  |  |  |  |

## 5. Best 후보

| 항목 | 선택값 | 선택 이유 |
| --- | --- | --- |
| best context_length |  |  |
| best n_layers |  |  |
| best emb_dim |  |  |

## 6. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 |
| --- | --- | --- |
|  |  |  |

## 7. 최종 결론

```text
모델을 키웠을 때 얻은 성능 이득과 학습 시간 증가를 비교하고,
발표에 사용할 최종 구조 후보를 적는다.
```
