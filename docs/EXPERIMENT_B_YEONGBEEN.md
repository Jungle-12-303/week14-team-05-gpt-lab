# B 실험 결과: 학습 하이퍼파라미터 탐색

| 항목 | 내용 |
| --- | --- |
| 담당자 | 영빈 |
| 담당 영역 | 하이퍼파라미터 탐색 1 |
| 주요 실험 | batch_size, learning_rate, drop_rate |
| 원본 계획 | [`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md) |
| 실행 스크립트 | `experiments/scripts/run_b_hparams.py` |

## 1. 실험 목표

학습 설정이 사전 학습 성능과 학습 시간에 미치는 영향을 확인한다. 시간이 제한되어 있으므로 learning_rate 비교를 필수로 수행하고, batch_size와 drop_rate는 가능한 범위에서 진행한다.

핵심 질문은 다음과 같다.

- learning_rate 1e-4, 3e-4, 5e-4 중 어떤 값이 가장 안정적으로 수렴하는가?
- batch_size를 키웠을 때 학습 시간과 validation loss가 어떻게 달라지는가?
- drop_rate가 과적합 완화에 도움이 되는가?

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
| Google Drive output root | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/{실험ID}_{YYYYMMDD}_YEONGBEEN/` |
| 공유 tokenizer 경로 | `artifacts/tokenizers/nsmc_bpe_vocab{vocab_size}_full.json` |
| raw metric JSONL 경로 | `{output_root}/metrics/{실험ID}_{YYYYMMDD}_metrics.jsonl` |
| stdout/stderr log 경로 | `{output_root}/logs/{실험ID}_{YYYYMMDD}.out` |
| latest checkpoint 경로 | `{output_root}/checkpoints/{실험ID}_{YYYYMMDD}_step{global_step}_latest.pt` |
| best checkpoint 경로 | `{output_root}/checkpoints/{실험ID}_{YYYYMMDD}_step{global_step}_best.pt` |

## 3. 고정 설정

| 항목 | 값 |
| --- | --- |
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

위 고정 설정의 `context_length=64`는 1차 screening 기준이다. 1차 screening은 `--vocab-size 2000 --train-char-limit 500000`으로 실행하고, 최종 하이퍼파라미터 후보는 가능하면 Basic 기준인 `--vocab-size 3000 --train-char-limit 1500000`으로 재확인한다. Basic 재확인을 하지 못한 경우 그 이유를 결론에 적는다.

## 5. 실험 결과

| 실험 ID | 변경 변수 | 값 | best global_step | best val loss | final train loss | 소요 시간 | best checkpoint | metric JSONL | 결론 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | batch_size | 2 |  |  |  |  |  |  |  |
| B1 | batch_size | 4 |  |  |  |  |  |  |  |
| B1 | batch_size | 8 |  |  |  |  |  |  |  |
| B1 | batch_size | 16 |  |  |  |  |  |  |  |
| B2 | learning_rate | 1e-4 |  |  |  |  |  |  |  |
| B2 | learning_rate | 3e-4 |  |  |  |  |  |  |  |
| B2 | learning_rate | 5e-4 |  |  |  |  |  |  |  |
| B3 | drop_rate | 0.0 |  |  |  |  |  |  |  |
| B3 | drop_rate | 0.1 |  |  |  |  |  |  |  |
| B3 | drop_rate | 0.2 |  |  |  |  |  |  |  |

## 6. Best 후보

| 항목 | 선택값 | 선택 이유 | 기준 metric |
| --- | --- | --- | --- |
| best batch_size |  |  |  |
| best learning_rate |  |  |  |
| best drop_rate |  |  |  |

## 7. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 | 보존 로그 |
| --- | --- | --- | --- |
|  |  |  |  |

## 8. 최종 결론

```text
가장 좋은 learning_rate, batch_size, drop_rate 후보와
그 선택 근거를 validation loss, 안정성, 소요 시간 기준으로 적는다.
```
