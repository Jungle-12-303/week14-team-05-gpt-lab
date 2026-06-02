# D 실험 결과: 감성 분류 개선 및 리포트 취합

| 항목 | 내용 |
| --- | --- |
| 담당자 | 형민 |
| 담당 영역 | 감성 분류 개선 및 리포트 취합 |
| 주요 실험 | class imbalance, freeze, lr 분리, best checkpoint |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |
| 실행 스크립트 | `experiments/scripts/run_d_sentiment.py` |

## 1. 실험 목표

사전 학습된 GPT backbone을 감성 분류에 더 잘 활용하기 위한 fine-tuning 전략을 비교한다. 최종 발표 전에는 A/B/C의 best pretrain 후보를 받아 sentiment test 결과를 확정한다.

핵심 질문은 다음과 같다.

- train/validation/test label 0/1 비율이 균형적인가?
- backbone 일부 freeze가 validation loss 또는 accuracy를 개선하는가?
- classifier learning rate와 backbone learning rate를 분리하면 성능이 좋아지는가?
- D0/D2/D3 중 best validation checkpoint를 선택했을 때 test accuracy가 개선되는가?

## 2. 공통 환경

| 항목 | 값 |
| --- | --- |
| Colab GPU | T4 |
| Colab runtime | GPU, Latest |
| Python version |  |
| PyTorch version |  |
| CUDA version |  |
| git commit |  |
| seed | 42 |
| 데이터 경로 | `/content/week14-team-05-gpt-lab/data/` |
| Google Drive output root | `/content/drive/MyDrive/gpt-lab/experiment_outputs/sentiment/{실험ID}_{YYYYMMDD}_HYEONGMIN/` |
| 공유 tokenizer 경로 | `artifacts/tokenizers/nsmc_bpe_vocab3000_full.json` |
| 사용한 pretrain checkpoint |  |
| raw metric JSONL 경로 | `{output_root}/metrics/D_{YYYYMMDD}_metrics.jsonl` |
| 결과 report 경로 | `{output_root}/logs/D_{YYYYMMDD}_HYEONGMIN.md` |
| best checkpoint 경로 | `{output_root}/checkpoints/{D0\|D2\|D3}_{YYYYMMDD}_step{global_step}_best.pt` |

## 3. Class Imbalance 확인

| split | label 0 | label 1 | total | positive ratio |
| --- | --- | --- | --- | --- |
| train |  |  |  |  |
| validation |  |  |  |  |
| test |  |  |  |  |

## 4. Fine-tuning 고정 설정

| 항목 | 값 |
| --- | --- |
| max_length | 64 |
| batch_size | 8 |
| num_epochs | 2 |
| baseline lr | 3e-4 |
| backbone lr | 1e-4 |
| classifier lr | 3e-4 |
| freeze_blocks | 1 |
| log_every_steps | 20 |
| eval_every_steps | 100 |
| save_every_steps | 100 |
| keep_latest | 2 |
| 저장 방식 | step 단위 metric JSONL/latest checkpoint 저장, validation loss 개선 시 best checkpoint 저장 |
| D4 test 1회 정책 | D0/D2/D3 중 validation loss가 가장 낮은 checkpoint만 test set 1회 평가 |

## 5. 실험 결과

| 실험 ID | 변경점 | best val loss | val acc | best checkpoint | metric JSONL | 결론 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | sentiment baseline |  |  |  |  |  |
| D2 | backbone 일부 freeze |  |  |  |  |  |
| D3 | classifier/backbone lr 분리 |  |  |  |  |  |

D1은 별도 학습 실험이 아니라 class imbalance 확인 기록이다. D0/D2/D3 후보 표에는 test loss/test accuracy를 채우지 않는다.

## 6. D4 최종 선택 기록

| 후보 | source checkpoint | validation loss | validation acc | test 평가 여부 | 선택/탈락 이유 |
| --- | --- | --- | --- | --- | --- |
| D0 |  |  |  | 미평가 |  |
| D2 |  |  |  | 미평가 |  |
| D3 |  |  |  | 미평가 |  |

D4는 독립 학습 실험이 아니라 D0/D2/D3의 후보 checkpoint 중 validation loss가 가장 낮은 모델을 선택하고 test set을 1회 평가하는 단계다. 선택된 후보의 test 결과만 아래 표에 채운다.

| selected candidate | source checkpoint | best val loss | val acc | test loss | test acc |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## 7. 취합용 최종 결과

| 구분 | best 설정 | 핵심 수치 | 발표에 넣을 결론 |
| --- | --- | --- | --- |
| 사전 학습 안정화 |  |  |  |
| 학습 하이퍼파라미터 |  |  |  |
| 모델 구조 |  |  |  |
| 감성 분류 |  |  |  |

## 8. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 | 보존 로그 |
| --- | --- | --- | --- |
|  |  |  |  |

## 9. 최종 결론

```text
최종 감성 분류 모델의 선택 근거와 test accuracy를 적고,
A/B/C 결과를 취합해 발표용 핵심 메시지 3개를 정리한다.
```
