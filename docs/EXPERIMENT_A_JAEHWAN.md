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
| Colab GPU | Tesla T4 |
| Colab runtime | GPU, Latest |
| Python version | 3.12.13 |
| PyTorch version | 2.11.0+cu128 |
| CUDA version | 12.8 |
| git commit | 129aae3 |
| seed | 42 |
| 데이터 경로 | `/content/week14-team-05-gpt-lab/data/nsmc_lm_train.txt`, `/content/week14-team-05-gpt-lab/data/nsmc_lm_val.txt` |
| Google Drive output root | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/{실험ID}_{YYYYMMDD}_JAEHWAN/` |
| 공유 tokenizer 경로 | `artifacts/tokenizers/nsmc_bpe_vocab3000_full.json` |
| raw metric JSONL 경로 | `{output_root}/metrics/{실험ID}_{YYYYMMDD}_metrics.jsonl` |
| stdout/stderr log 경로 | `{output_root}/logs/{실험ID}_{YYYYMMDD}.out` |
| latest checkpoint 경로 | `{output_root}/checkpoints/{실험ID}_{YYYYMMDD}_step{global_step}_latest.pt` |
| best checkpoint 경로 | `{output_root}/checkpoints/{실험ID}_{YYYYMMDD}_step{global_step}_best.pt` |
| W&B | `--wandb --wandb-mode offline`, `/content/drive/MyDrive/gpt-lab/wandb/wandb/offline-run-20260602_140926-3xiswjd9` |

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
| A0 | quick smoke baseline, shared BPE vocab3000, W&B offline | 40 | 8.0831 | 8.1491 | 0.07 min | `local/experiment_outputs/pretrain/A0_20260602_JAEHWAN/checkpoints/A0_20260602_step0040_best.pt` | `local/experiment_outputs/pretrain/A0_20260602_JAEHWAN/metrics/A0_20260602_metrics.jsonl` | 실행 확인 완료, 공식 비교 제외 |
| A0_basic | Basic submission baseline, shared BPE vocab3000, W&B offline | 1574 | 6.7148 | 7.0854 | 13.81 min | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/A0_basic_20260602_JAEHWAN/checkpoints/A0_basic_20260602_step1574_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/A0_basic_20260602_JAEHWAN/metrics/A0_basic_20260602_metrics.jsonl` | Basic 기준 baseline 확보 |
| A1 | warmup + cosine decay, Basic context, min lr floor, shared BPE vocab3000, W&B offline | 1574 | 7.2470 | 7.2717 | 13.47 min | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/A1_20260603_JAEHWAN/checkpoints/A1_20260603_step1574_best.pt` | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/A1_20260603_JAEHWAN/metrics/A1_20260603_metrics.jsonl` | A0_basic 대비 val loss 악화, 최종 후보 제외 |
| A2 | gradient clipping, Basic context |  |  |  |  |  |  |  |
| A3 | weight_decay=0.01, Basic context |  |  |  |  |  |  |  |
| A4 | warmup + cosine + clipping + weight_decay, Basic context |  |  |  |  |  |  |  |

## 6. Step metric 기록

| 실험 ID | global_step | epoch | train loss | val loss | checkpoint | 메모 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 20 | 1 | 8.0988 | 8.1016 | `A0_20260602_step0020_best.pt` | quick smoke eval |
| A0 | 40 | 1 | 8.0696 | 8.0831 | `A0_20260602_step0040_best.pt` | best val loss |
| A0 | 46 | 1 | 8.1491 | 8.1125 | `A0_20260602_step0040_best.pt` | epoch end |
| A0_basic | 100 | 1 | 7.3437 | 7.3329 |  | eval |
| A0_basic | 200 | 1 | 7.2860 | 7.3116 |  | eval |
| A0_basic | 300 | 1 | 7.2802 | 7.3046 |  | eval |
| A0_basic | 400 | 1 | 7.2842 | 7.3001 |  | eval |
| A0_basic | 500 | 1 | 7.2793 | 7.2968 |  | eval |
| A0_basic | 600 | 1 | 7.2899 | 7.2944 |  | eval |
| A0_basic | 700 | 1 | 7.2583 | 7.2889 |  | eval |
| A0_basic | 787 | 1 | 7.3562 | 7.2747 |  | epoch 1 end |
| A0_basic | 800 | 2 | 7.2929 | 7.2794 |  | eval |
| A0_basic | 900 | 2 | 7.2298 | 7.2472 |  | eval |
| A0_basic | 1000 | 2 | 7.1773 | 7.1982 |  | eval |
| A0_basic | 1100 | 2 | 7.1039 | 7.1389 |  | eval |
| A0_basic | 1200 | 2 | 7.0280 | 7.0632 |  | eval |
| A0_basic | 1300 | 2 | 6.9579 | 6.9819 |  | eval |
| A0_basic | 1400 | 2 | 6.8452 | 6.8968 | `A0_basic_20260602_step1400_latest.pt` | latest checkpoint |
| A0_basic | 1500 | 2 | 6.7438 | 6.7979 | `A0_basic_20260602_step1500_latest.pt` | latest checkpoint |
| A0_basic | 1574 | 2 | 7.0854 | 6.7148 | `A0_basic_20260602_step1574_best.pt` | best val loss, epoch 2 end |
| A1 | 1500 | 2 | 7.2523 | 7.2516 | `A1_20260603_step1500_best.pt` | eval, learning_rate 3.15e-5 |
| A1 | 1574 | 2 | 7.2717 | 7.2470 | `A1_20260603_step1574_best.pt` | best val loss, epoch 2 end, learning_rate 3.00e-5 |
| A2 |  |  |  |  |  |  |
| A3 |  |  |  |  |  |  |
| A4 |  |  |  |  |  |  |

## 7. 실패 또는 중단 실험

| 실험 ID | 원인 | 조치 | 보존 로그 |
| --- | --- | --- | --- |
| A1_20260602 | `context_length=64` 기준으로 추정되어 A0_basic(`context_length=128`)과 직접 비교 불가. 후반 learning rate도 0.0까지 감소 | A1을 `context_length=128`, `warmup_steps=50`, `min_lr_ratio=0.1`로 수정 후 `A1_20260603_JAEHWAN` 재실행 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/A1_20260602_JAEHWAN/logs/A1_20260602.out` |

## 8. 최종 결론

A0_basic을 Basic 제출 기준 baseline으로 확보했다. `vocab_size=3000`, `context_length=128`, `train_char_limit=1500000`, `num_epochs=2` 설정에서 best validation loss는 6.7148이고, best checkpoint는 final global step 1574에서 저장되었다.

epoch 1 종료 시 validation loss는 7.2747이었고, epoch 2 종료 시 6.7148까지 낮아졌다. 이후 A1~A4 안정화 실험은 이 A0_basic 결과를 기준선으로 삼아 validation loss 개선 여부와 loss spike/NaN 발생 여부를 비교한다.

A1은 기존 실행의 비교 조건 문제를 보정하기 위해 `context_length=128`, `warmup_steps=50`, `min_lr_ratio=0.1`로 재실행했다. 이번 재실행은 A0_basic과 같은 final global step 1574에서 종료되었고, 마지막 learning rate도 3.00e-5로 유지되어 Basic 기준과 scheduler floor가 정상 적용되었다.

그러나 A1 재실행의 best validation loss는 7.2470으로 A0_basic의 6.7148보다 0.5322 높았다. 따라서 현재 warmup + cosine decay 설정은 validation loss 개선에 실패했으므로 최종 안정화 후보에서 제외한다. 이 결과는 warmup + cosine 자체의 일반적 실패라기보다, 현재 2 epoch 학습 길이와 `warmup_steps=50`, `min_lr_ratio=0.1` 조합이 baseline constant learning rate보다 유리하지 않았다는 근거로 해석한다.
