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
| Python version |  |
| PyTorch version |  |
| CUDA version |  |
| git commit |  |
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

C는 context_length 자체를 비교하므로 `64` 결과는 1차 screening 기준, `128` 결과는 Basic 기준 확인으로 기록한다. 1차 screening은 `--vocab-size 2000 --train-char-limit 500000`으로 실행하고, Basic 확인은 `--vocab-size 3000 --train-char-limit 1500000`으로 실행한다. 최종 구조 후보가 `64`를 선택한다면 validation loss, 학습 시간, GPU 제약을 근거로 설명한다.

## 5. 실험 결과

| 실험 ID | 변경 변수 | 값 | best global_step | best val loss | final train loss | 소요 시간 | 메모리/OOM | best checkpoint | metric JSONL | 결론 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | context_length | 64 |  |  |  |  |  |  |  |  |
| C1 | context_length | 128 |  |  |  |  |  |  |  |  |
| C2 | n_layers | 1 |  |  |  |  |  |  |  |  |
| C2 | n_layers | 2 |  |  |  |  |  |  |  |  |
| C2 | n_layers | 4 |  |  |  |  |  |  |  |  |
| C3 | emb_dim | 64 |  |  |  |  |  |  |  |  |
| C3 | emb_dim | 128 |  |  |  |  |  |  |  |  |
| C3 | emb_dim | 192 |  |  |  |  |  |  |  |  |

## 6. Best 후보

| 항목 | 선택값 | 선택 이유 | 기준 metric |
| --- | --- | --- | --- |
| best context_length |  |  |  |
| best n_layers |  |  |  |
| best emb_dim |  |  |  |

## 7. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 | 보존 로그 |
| --- | --- | --- | --- |
|  |  |  |  |

## 8. 최종 결론

```text
모델을 키웠을 때 얻은 성능 이득과 학습 시간 증가를 비교하고,
발표에 사용할 최종 구조 후보를 적는다.
```
