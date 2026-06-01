# B 실험 결과: 학습 하이퍼파라미터 탐색

| 항목 | 내용 |
| --- | --- |
| 담당자 | 영빈 |
| 담당 영역 | 하이퍼파라미터 탐색 1 |
| 주요 실험 | batch_size, learning_rate, drop_rate |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |

## 1. 실험 목표

학습 설정이 사전 학습 성능과 학습 시간에 미치는 영향을 확인한다. 시간이 제한되어 있으므로 learning_rate 비교를 필수로 수행하고, batch_size와 drop_rate는 가능한 범위에서 진행한다.

핵심 질문은 다음과 같다.

- learning_rate 1e-4, 3e-4, 5e-4 중 어떤 값이 가장 안정적으로 수렴하는가?
- batch_size를 키웠을 때 학습 시간과 validation loss가 어떻게 달라지는가?
- drop_rate가 과적합 완화에 도움이 되는가?

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
| context_length | 64 |
| n_layers | 2 |
| emb_dim | 128 |
| n_heads | 4 |
| optimizer | AdamW |
| num_epochs | 2~3 |
| eval_freq / eval_iter |  |

## 4. 실험 결과

| 실험 ID | 변경 변수 | 값 | best val loss | final train loss | 소요 시간 | 결론 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | batch_size | 2 |  |  |  |  |
| B1 | batch_size | 4 |  |  |  |  |
| B1 | batch_size | 8 |  |  |  |  |
| B1 | batch_size | 16 |  |  |  |  |
| B2 | learning_rate | 1e-4 |  |  |  |  |
| B2 | learning_rate | 3e-4 |  |  |  |  |
| B2 | learning_rate | 5e-4 |  |  |  |  |
| B3 | drop_rate | 0.0 |  |  |  |  |
| B3 | drop_rate | 0.1 |  |  |  |  |
| B3 | drop_rate | 0.2 |  |  |  |  |

## 5. Best 후보

| 항목 | 선택값 | 선택 이유 |
| --- | --- | --- |
| best batch_size |  |  |
| best learning_rate |  |  |
| best drop_rate |  |  |

## 6. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 |
| --- | --- | --- |
|  |  |  |

## 7. 최종 결론

```text
가장 좋은 learning_rate, batch_size, drop_rate 후보와
그 선택 근거를 validation loss, 안정성, 소요 시간 기준으로 적는다.
```
