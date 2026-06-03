# D 실험 결과: 감성 분류 개선 및 리포트 취합

| 항목 | 내용 |
| --- | --- |
| 담당자 | 형민 |
| 담당 영역 | 감성 분류 개선 및 리포트 취합 |
| 주요 실험 | class imbalance, freeze, lr 분리, best checkpoint |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |
| 실행 스크립트 | `experiments/scripts/run_d_sentiment.py` |

## 1. 실험 목표

사전 학습된 GPT backbone을 감성 분류에 더 잘 활용하기 위한 fine-tuning 전략을 비교했다. 최종 D 실험은 A 실험의 Basic baseline checkpoint인 `A0_basic`을 입력으로 사용했고, D0/D2/D3 validation 후보 중 validation loss가 가장 낮은 checkpoint를 D4에서 test set에 1회 평가했다.

핵심 질문과 답은 다음과 같다.

- train/validation/test label 0/1 비율이 균형적인가?
  세 split 모두 거의 50:50에 가까워 class imbalance 보정은 필요하지 않았다.
- backbone 일부 freeze가 validation loss 또는 accuracy를 개선하는가?
  개선하지 못했다. D2는 D0보다 validation loss가 높고 accuracy가 낮았다.
- classifier learning rate와 backbone learning rate를 분리하면 성능이 좋아지는가?
  freeze보다 좋았지만 baseline인 D0보다 낮은 성능을 보였다.
- D0/D2/D3 중 best validation checkpoint를 선택했을 때 test accuracy가 개선되는가?
  D4는 validation loss가 가장 낮은 D0 checkpoint를 선택했고, test accuracy는 0.8188을 기록했다.

## 2. 공통 환경

최초 Colab 실행은 장시간 실행 중단과 동기화 불확실성 때문에 공식 결과에서 제외했다. 최종 공식 D 결과는 로컬 GPU에서 재실행한 아래 결과를 기준으로 한다.

| 항목 | 값 |
| --- | --- |
| 실행 환경 | Local Windows GPU |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Python version | 3.11.15 |
| PyTorch version | 2.11.0+cu128 |
| CUDA version | 12.8 |
| device | cuda |
| git commit | 3c4cf51 |
| seed | 42 |
| 데이터 경로 | `data/` |
| output root | `local/experiment_outputs/sentiment/D4_A0_basic_local_20260603_170850_HYEONGMIN/` |
| 공유 tokenizer 경로 | `artifacts/tokenizers/nsmc_bpe_vocab3000_full.json` |
| 사용한 pretrain checkpoint | `local/experiment_outputs/drive_upload/pretrain/A0_basic_20260602_JAEHWAN/checkpoints/A0_basic_20260602_step1574_best.pt` |
| raw metric JSONL 경로 | `local/experiment_outputs/sentiment/D4_A0_basic_local_20260603_170850_HYEONGMIN/metrics/D_20260603_metrics.jsonl` |
| 결과 report 경로 | `local/experiment_outputs/sentiment/D4_A0_basic_local_20260603_170850_HYEONGMIN/logs/D_20260603_HYEONGMIN.md` |
| summary 경로 | `local/experiment_outputs/sentiment/D4_A0_basic_local_20260603_170850_HYEONGMIN/summary.json` |

## 3. Class Imbalance 확인

| split | label 0 | label 1 | total | positive ratio |
| --- | ---: | ---: | ---: | ---: |
| train | 69076 | 68920 | 137996 | 0.4994 |
| validation | 6094 | 5905 | 11999 | 0.4921 |
| test | 24826 | 25171 | 49997 | 0.5035 |

train, validation, test 모두 label 0/1 비율이 거의 균형적이다. 따라서 이번 D 실험에서는 class weight, over/under-sampling, threshold 조정 같은 imbalance 대응을 적용하지 않았다.

## 4. Fine-tuning 고정 설정

| 항목 | 값 |
| --- | --- |
| pretrain source | A0_basic Basic baseline |
| vocab_size | 3000 |
| max_length | 128 |
| emb_dim / n_layers / n_heads | 128 / 2 / 4 |
| batch_size | 8 |
| num_epochs | 2 |
| baseline lr | 3e-4 |
| backbone lr | 1e-4 |
| classifier lr | 3e-4 |
| drop_rate | 0.1 |
| freeze_blocks | 1 |
| log_every_steps | 20 |
| eval_every_steps | 0 |
| save_every_steps | 0 |
| keep_latest | 2 |
| local GPU option | `--local-gpu`, `num_workers=4`, `pin_memory=True`, `enable_tf32=True` |
| 저장 방식 | epoch 종료 시 validation loss를 평가하고, 개선 시 best checkpoint 저장. step 단위 validation/latest checkpoint 저장은 비활성화 |
| D4 test 1회 정책 | D0/D2/D3 중 validation loss가 가장 낮은 checkpoint만 test set 1회 평가 |

`max_length=128`은 A0_basic pretrain checkpoint의 Basic 설정(`context_length=128`)에 맞춘 값이다. D runner는 `--pretrain-run-dir`로 전달한 A0_basic 산출물의 `run_config.json`을 읽어 모델 구조와 tokenizer 경로를 맞췄다.

## 5. 실험 결과

| 실험 ID | 변경점 | train loss | train acc | best val loss | val acc | global step | elapsed min | 결론 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D0 | sentiment baseline | 0.4049 | 0.8151 | 0.3930 | 0.8186 | 34500 | 20.16 | D4 선택 |
| D2 | backbone 일부 freeze | 0.5697 | 0.6964 | 0.5469 | 0.7088 | 34500 | 18.73 | 탈락 |
| D3 | classifier/backbone lr 분리 | 0.4589 | 0.7820 | 0.4342 | 0.7931 | 34500 | 23.16 | 탈락 |

D1은 별도 학습 실험이 아니라 class imbalance 확인 기록이다. D0/D2/D3 후보 표에는 test loss/test accuracy를 채우지 않는다. test set은 D4에서 선택된 checkpoint에 대해서만 1회 평가했다.

Best checkpoint 경로는 다음과 같다.

| 실험 ID | best checkpoint |
| --- | --- |
| D0 | `local/experiment_outputs/sentiment/D4_A0_basic_local_20260603_170850_HYEONGMIN/checkpoints/D0_20260603_step34500_best.pt` |
| D2 | `local/experiment_outputs/sentiment/D4_A0_basic_local_20260603_170850_HYEONGMIN/checkpoints/D2_20260603_step34500_best.pt` |
| D3 | `local/experiment_outputs/sentiment/D4_A0_basic_local_20260603_170850_HYEONGMIN/checkpoints/D3_20260603_step34500_best.pt` |

## 6. D4 최종 선택 기록

| 후보 | source checkpoint | validation loss | validation acc | test 평가 여부 | 선택/탈락 이유 |
| --- | --- | ---: | ---: | --- | --- |
| D0 | `D0_20260603_step34500_best.pt` | 0.3930 | 0.8186 | 평가 | validation loss가 가장 낮아 D4 최종 선택 |
| D2 | `D2_20260603_step34500_best.pt` | 0.5469 | 0.7088 | 미평가 | freeze 적용 후 validation loss/accuracy가 모두 악화 |
| D3 | `D3_20260603_step34500_best.pt` | 0.4342 | 0.7931 | 미평가 | D2보다는 좋지만 D0보다 validation loss가 높음 |

D4는 독립 학습 실험이 아니라 D0/D2/D3의 후보 checkpoint 중 validation loss가 가장 낮은 모델을 선택하고 test set을 1회 평가하는 단계다.

| selected candidate | source checkpoint | best val loss | val acc | test loss | test acc |
| --- | --- | ---: | ---: | ---: | ---: |
| D0 | `D0_20260603_step34500_best.pt` | 0.3930 | 0.8186 | 0.3939 | 0.8188 |

## 7. 취합용 최종 결과

| 구분 | best 설정 | 핵심 수치 | 발표에 넣을 결론 |
| --- | --- | --- | --- |
| 사전 학습 안정화 | A0_basic | best val loss 6.7148 | Basic 기준 2 epoch에서는 안정화 기법 추가보다 baseline이 가장 낮은 validation loss를 기록했다. |
| 학습 하이퍼파라미터 | B1 batch_size=2, B2 lr=5e-4, B3 drop_rate=0.0 | Light screening 기준 각 그룹 best | B 결과는 Light screening 기준이므로 Basic confirmation 없이 최종 구조로 단정하지 않는다. |
| 모델 구조 | C2 n_layers=4, C3 emb_dim=192 후보 | C2 best val loss 7.1540, C3 best val loss 6.8935 | C 결과는 architecture screening 후보이며 Basic 조합 실험은 추가 확인이 필요하다. |
| 감성 분류 | D0 baseline fine-tuning | val loss 0.3930, test acc 0.8188 | A0_basic checkpoint 기반 downstream에서는 freeze와 split LR보다 baseline fine-tuning이 가장 좋았다. |

## 8. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 | 보존 로그 |
| --- | --- | --- | --- |
| Colab D4_A0_basic_20260603_054922_HYEONGMIN | Colab/VS Code 확장 실행 중 장시간 지연 및 중단. D0는 완료됐지만 D2/D3/D4가 완료되지 않음 | 공식 결과에서 제외하고 로컬 GPU에서 전체 D0/D2/D3/D4 재실행 | Google Drive 및 로컬 복사본 `local/D_20260603_metrics.jsonl` |
| D_local_smoke_A0_basic_20260603_HYEONGMIN | 로컬 CUDA smoke test | 공식 수치에서 제외하고 실행 가능성 확인에만 사용 | `local/experiment_outputs/sentiment/D_local_smoke_A0_basic_20260603_HYEONGMIN/` |
| D_local_gpu_verify_smoke_20260603_HYEONGMIN | GPU 사용 여부 확인용 smoke test | 공식 수치에서 제외하고 CUDA 경로 검증에만 사용 | `local/experiment_outputs/sentiment/D_local_gpu_verify_smoke_20260603_HYEONGMIN/` |

## 9. 최종 결론

최종 D 실험에서는 A0_basic pretrain checkpoint를 입력으로 D0/D2/D3 fine-tuning 후보를 비교했다. validation loss 기준으로 D0 baseline이 가장 낮은 0.3930을 기록했고, D4 단계에서 D0 checkpoint만 test set에 1회 평가했다. 최종 test accuracy는 0.8188이다.

이번 조건에서는 class imbalance 문제가 크지 않았고, backbone 일부 freeze나 backbone/classifier learning rate 분리는 baseline보다 성능을 개선하지 못했다. 따라서 최종 감성 분류 checkpoint는 `D0_20260603_step34500_best.pt`로 선택한다.

보고서에는 D 결과를 다음 문장으로 요약할 수 있다.

```text
A0_basic 사전 학습 checkpoint를 사용한 감성 분류 fine-tuning에서는 D0 baseline이
validation loss 0.3930, test accuracy 0.8188로 가장 좋았다.
Freeze와 learning rate 분리는 이번 설정에서 baseline보다 낮은 성능을 보였으므로
최종 sentiment 모델은 D0 best checkpoint를 선택했다.
```
