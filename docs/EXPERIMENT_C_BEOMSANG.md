# C 실험 결과: 모델 구조 하이퍼파라미터 탐색

| 항목 | 내용 |
| --- | --- |
| 담당자 | 범상 |
| 담당 영역 | 하이퍼파라미터 탐색 2 |
| 주요 실험 | context_length, n_layers, emb_dim |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |
| 실행 스크립트 | `experiments/scripts/run_c_architecture.py` |

## 1. 실험 목표

모델 구조를 바꾸었을 때 validation loss, 학습 시간, GPU 메모리 사용량이 어떻게 변하는지 확인한다. 시간이 제한되어 있으므로 context_length 또는 n_layers 비교를 필수로 수행하고, emb_dim 비교는 가능한 범위에서 진행한다.

핵심 질문은 다음과 같다.

- context_length 64와 128 중 어느 쪽이 validation loss에 유리한가?
- n_layers를 늘렸을 때 성능 개선이 학습 시간 증가를 정당화하는가?
- emb_dim을 키우면 모델 성능이 좋아지는가, 아니면 과도하게 느려지는가?

## 2. 공통 환경

| 항목 | 값 |
| --- | --- |
| Colab GPU | T4 |
| Colab runtime | GPU, Latest |
| Python version | 3.12.13 |
| PyTorch version | 2.11.0+cu128 |
| CUDA version | 12.8 |
| git commit | b9a1b64 |
| seed | 42 |
| 데이터 경로 | `/content/week14-team-05-gpt-lab/data/` |
| Google Drive output root | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/{실험ID}_{YYYYMMDD}_BEOMSANG/` |
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
| n_heads | 4 |
| optimizer | AdamW |
| num_epochs | 2 |
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

위 표는 계획서의 실행 규모 기준이다. 실제 C 산출물의 `run_config.json`을 확인한 결과, 이번 C 실험은 Light 또는 Basic 기준을 그대로 따른 실행이 아니라 `vocab_size=3000`, `train_char_limit=500000`으로 통일한 architecture screening으로 수행되었다.

| 항목 | 실제 실행값 |
| --- | ---: |
| vocab_size | 3000 |
| train_char_limit | 500000 |
| num_epochs | 2 |
| batch_size | 8 |
| learning_rate | 3e-4 |
| GPU | Tesla T4 |

따라서 `C1_ctx64`는 계획서의 Light 기준이 아니고, `C1_ctx128`도 계획서의 Basic confirmation이 아니다. 두 결과는 같은 `vocab_size=3000`, `train_char_limit=500000` 조건에서 `context_length`만 바꾼 비교로 해석한다. Basic 기준인 `vocab_size=3000`, `train_char_limit=1500000`, `context_length=128` confirmation은 수행하지 않았다.

## 5. 실험 결과

| 실험 ID | 변경 변수 | 값 | vocab_size | train_char_limit | context_length | n_layers | emb_dim | best global_step | best val loss | final train loss | 소요 시간 | 메모리/OOM | best checkpoint | metric JSONL | 결론 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| C1_ctx64 | context_length | 64 | 3000 | 500000 | 64 | 2 | 128 | 1144 | 7.2248 | 7.2784 | 5.9분 | 없음 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C1_ctx64_20260602_BEOMSANG/checkpoints/C1_ctx64_20260602_step1144_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C1_ctx64_20260602_BEOMSANG/metrics/C1_ctx64_20260602_metrics.jsonl` | keep |
| C1_ctx128 | context_length | 128 | 3000 | 500000 | 128 | 2 | 128 | 572 | 7.2879 | 7.2996 | 5.6분 | 없음 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C1_ctx128_20260602_BEOMSANG/checkpoints/C1_ctx128_20260602_step0572_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C1_ctx128_20260602_BEOMSANG/metrics/C1_ctx128_20260602_metrics.jsonl` | keep |
| C2_layers1 | n_layers | 1 | 3000 | 500000 | 64 | 1 | 128 | 1144 | 7.2354 | 7.2793 | 5.7분 | 없음 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C2_layers1_20260602_BEOMSANG/checkpoints/C2_layers1_20260602_step1144_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C2_layers1_20260602_BEOMSANG/metrics/C2_layers1_20260602_metrics.jsonl` | keep |
| C2_layers2 | n_layers | 2 | 3000 | 500000 | 64 | 2 | 128 | 1144 | 7.2248 | 7.2784 | 5.7분 | 없음 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C2_layers2_20260602_BEOMSANG/checkpoints/C2_layers2_20260602_step1144_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C2_layers2_20260602_BEOMSANG/metrics/C2_layers2_20260602_metrics.jsonl` | keep |
| C2_layers4 | n_layers | 4 | 3000 | 500000 | 64 | 4 | 128 | 1144 | 7.1540 | 7.2627 | 5.8분 | 없음 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C2_layers4_20260602_BEOMSANG/checkpoints/C2_layers4_20260602_step1144_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C2_layers4_20260602_BEOMSANG/metrics/C2_layers4_20260602_metrics.jsonl` | keep |
| C3_dim64 | emb_dim | 64 | 3000 | 500000 | 64 | 2 | 64 | 1144 | 7.2933 | 7.3021 | 5.7분 | 없음 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C3_dim64_20260602_BEOMSANG/checkpoints/C3_dim64_20260602_step1144_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C3_dim64_20260602_BEOMSANG/metrics/C3_dim64_20260602_metrics.jsonl` | keep |
| C3_dim128 | emb_dim | 128 | 3000 | 500000 | 64 | 2 | 128 | 1144 | 7.2248 | 7.2784 | 5.8분 | 없음 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C3_dim128_20260602_BEOMSANG/checkpoints/C3_dim128_20260602_step1144_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C3_dim128_20260602_BEOMSANG/metrics/C3_dim128_20260602_metrics.jsonl` | keep |
| C3_dim192 | emb_dim | 192 | 3000 | 500000 | 64 | 2 | 192 | 1144 | 6.8935 | 7.1465 | 5.9분 | 없음 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C3_dim192_20260602_BEOMSANG/checkpoints/C3_dim192_20260602_step1144_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/C3_dim192_20260602_BEOMSANG/metrics/C3_dim192_20260602_metrics.jsonl` | keep |

## 6. 독립 비교 기준 후보

| 항목 | 선택값 | 선택 이유 | 기준 metric |
| --- | --- | --- | --- |
| best context_length | 64 | 128보다 best validation loss가 낮음 | best val loss 7.2248 |
| best n_layers | 4 | 1, 2 layers보다 best validation loss가 낮음 | best val loss 7.1540 |
| best emb_dim | 192 | 64, 128보다 best validation loss가 낮고 전체 C 실험 중 최저 loss 기록 | best val loss 6.8935 |

위 후보는 각 변수별 독립 비교에서 나온 유망값이다. `context_length=64`, `n_layers=4`, `emb_dim=192`를 동시에 적용한 조합 실험은 수행하지 않았다. 실제 산출물 중 best validation loss가 가장 낮은 checkpoint는 `C3_dim192`이며, 해당 모델 설정은 `context_length=64`, `n_layers=2`, `emb_dim=192`이다.

## 7. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 | 보존 로그 |
| --- | --- | --- | --- |
| 없음 | - | - | - |

## 8. 최종 결론

C 실험에서는 `vocab_size=3000`, `train_char_limit=500000` 조건에서 context_length, n_layers, emb_dim을 각각 변경해 모델 구조가 validation loss에 미치는 영향을 screening했다. context_length 비교에서는 64가 128보다 낮은 best validation loss를 보였고, n_layers 비교에서는 4 layers가 가장 낮은 loss를 기록했다. emb_dim 비교에서는 192가 가장 좋은 결과를 보였으며, 전체 C 실험 중에서도 `C3_dim192`가 best val loss 6.8935로 가장 우수했다.

따라서 독립 비교 기준의 유망 구조 후보는 `context_length=64`, `n_layers=4`, `emb_dim=192`이다. 다만 이 세 값을 동시에 적용한 조합 실험과 계획서의 Basic 기준인 `vocab_size=3000`, `train_char_limit=1500000`, `context_length=128` confirmation은 수행하지 않았다. 최종 보고서에서는 이 결과를 최종 확정 설정이 아니라 C 담당 영역의 architecture screening 결과로 표현한다.
