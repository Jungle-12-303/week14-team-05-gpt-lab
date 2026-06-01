# D 실험 결과: 감성 분류 개선 및 리포트 취합

| 항목 | 내용 |
| --- | --- |
| 담당자 | 형민 |
| 담당 영역 | 감성 분류 개선 및 리포트 취합 |
| 주요 실험 | class imbalance, freeze, lr 분리, best checkpoint |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |

## 1. 실험 목표

사전 학습된 GPT backbone을 감성 분류에 더 잘 활용하기 위한 fine-tuning 전략을 비교한다. 최종 발표 전에는 A/B/C의 best pretrain 후보를 받아 sentiment test 결과를 확정한다.

핵심 질문은 다음과 같다.

- train/validation/test label 0/1 비율이 균형적인가?
- backbone 일부 freeze가 validation loss 또는 accuracy를 개선하는가?
- classifier learning rate와 backbone learning rate를 분리하면 성능이 좋아지는가?
- 마지막 epoch가 아니라 best validation checkpoint를 선택했을 때 test accuracy가 개선되는가?

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
| 사용한 pretrain checkpoint |  |

## 3. Class Imbalance 확인

| split | label 0 | label 1 | total | positive ratio |
| --- | --- | --- | --- | --- |
| train |  |  |  |  |
| validation |  |  |  |  |
| test |  |  |  |  |

## 4. Fine-tuning 고정 설정

| 항목 | 값 |
| --- | --- |
| max_length | 128 |
| batch_size |  |
| num_epochs |  |
| baseline lr |  |
| backbone lr |  |
| classifier lr |  |
| freeze 범위 |  |

## 5. 실험 결과

| 실험 ID | 변경점 | best val loss | val acc | test loss | test acc | checkpoint | 결론 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | sentiment baseline |  |  |  |  |  |  |
| D1 | class imbalance 확인 |  |  |  |  |  |  |
| D2 | backbone 일부 freeze |  |  |  |  |  |  |
| D3 | classifier/backbone lr 분리 |  |  |  |  |  |  |
| D4 | best validation checkpoint 선택 |  |  |  |  |  |  |

## 6. 취합용 최종 결과

| 구분 | best 설정 | 핵심 수치 | 발표에 넣을 결론 |
| --- | --- | --- | --- |
| 사전 학습 안정화 |  |  |  |
| 학습 하이퍼파라미터 |  |  |  |
| 모델 구조 |  |  |  |
| 감성 분류 |  |  |  |

## 7. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 |
| --- | --- | --- |
|  |  |  |

## 8. 최종 결론

```text
최종 감성 분류 모델의 선택 근거와 test accuracy를 적고,
A/B/C 결과를 취합해 발표용 핵심 메시지 3개를 정리한다.
```
