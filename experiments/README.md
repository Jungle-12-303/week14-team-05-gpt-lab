# 실험 실행 스크립트

이 디렉토리는 A/B/C/D 추가 미션을 Colab에서 재현 가능하게 실행하기 위한 runner를 담는다. 핵심 모델, dataset, tokenizer, train/fine-tune 로직은 `src/`를 사용하고, `experiments/scripts/`는 담당자별 실험 ID, 설정값, 출력 디렉토리, checkpoint/log/metric 파일명을 통일한다.

## 디렉토리 역할

```text
experiments/
  README.md
  scripts/
    _pretrain_runner.py           # A/B/C 공통 사전 학습 자동화
    run_a_pretrain_stability.py   # A: warmup, cosine, clipping, weight decay
    run_b_hparams.py              # B: batch_size, learning_rate, drop_rate
    run_c_architecture.py         # C: context_length, n_layers, emb_dim
    run_d_sentiment.py            # D: 감성 분류 fine-tuning
```

## Colab 사전 준비

Colab에서는 repo를 `/content`에 두고, 산출물만 Google Drive에 저장한다.

```python
from google.colab import drive

drive.mount("/content/drive")
```

```bash
cd /content
git clone https://github.com/Jungle-12-303/week14-team-05-gpt-lab.git
cd week14-team-05-gpt-lab
pip install -r requirements.txt
python download_data.py
```

## A/B/C 사전 학습 실행

A/B/C runner는 다음 작업을 자동으로 수행한다.

- 데이터 파일 확인, 필요하면 `download_data.py` 실행
- 공유 BPE tokenizer 로드 또는 전체 `data/nsmc_lm_train.txt` 기준 신규 학습
- train/validation dataloader 생성
- GPTModel, AdamW, 선택 scheduler 구성
- `src.train.train_model()` 실행
- step 단위 metric JSONL 저장
- step 단위 latest checkpoint 저장 및 최근 N개 유지
- validation loss 개선 시 best checkpoint 저장
- `run_config.json`, `summary.json`, stdout/stderr log 저장

Light screening 실행 예시는 다음과 같다. `A0`는 빠른 후보 선별 기준이고, `A0_basic`은 아래 Basic baseline 명령으로 별도 실행한다.

```bash
python experiments/scripts/run_a_pretrain_stability.py \
  --experiment A0 \
  --vocab-size 2000 \
  --train-char-limit 500000

python experiments/scripts/run_b_hparams.py \
  --experiment B2 \
  --vocab-size 2000 \
  --train-char-limit 500000

python experiments/scripts/run_c_architecture.py \
  --experiment C1 \
  --vocab-size 2000 \
  --train-char-limit 500000
```

`--experiment all`을 사용하면 해당 담당자의 전체 후보를 순차 실행한다. B/C는 그룹 ID도 사용할 수 있다.

```bash
python experiments/scripts/run_b_hparams.py --experiment B2
python experiments/scripts/run_c_architecture.py --experiment C2
```

Colab Drive 저장 경로를 명시하려면 `--output-root`를 지정한다.

```bash
python experiments/scripts/run_a_pretrain_stability.py \
  --experiment A1 \
  --output-root /content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain
```

smoke test는 `--quick`을 사용한다. `--quick`은 작은 모델, 짧은 입력 규모, 작은 vocab을 사용하므로 공식 결과가 아니라 실행 확인용이다.

```bash
python experiments/scripts/run_a_pretrain_stability.py --experiment A0 --quick
```

공식 성능 비교 기준은 Smoke/Light/Basic을 구분한다. Smoke는 실행 확인, Light는 1차 screening, Basic은 최종 제출 후보 검증 기준이다.

| 구분 | 모델 학습 입력 규모 | vocab_size | context_length | 용도 |
| --- | ---: | ---: | ---: | --- |
| Smoke | `corpus[:5000]` 수준 | 300 | 32 | 실행 확인 |
| Light | `corpus[:500000]` 수준 | 2000 | 64 | 1차 screening |
| Basic | `corpus[:1500000]` 수준 | 3000 | 128 | 최종 후보 검증 |

Basic baseline은 `A0_basic`으로 실행한다. 현재 NSMC LM train text가 1,500,000자보다 짧으면 아래 `--train-char-limit 1500000`은 전체 train corpus 사용과 같다.

```bash
python experiments/scripts/run_a_pretrain_stability.py \
  --experiment A0_basic \
  --vocab-size 3000 \
  --train-char-limit 1500000 \
  --output-root /content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain
```

1차 screening 이후 좋은 후보만 더 확인할 때는 같은 명령에 `--num-epochs 3` 이상을 추가한다.

## D 감성 분류 실행

D runner는 D0/D2/D3 감성 분류 fine-tuning 후보를 학습하고, D4 단계에서 validation loss가 가장 낮은 checkpoint를 선택해 test set을 1회만 평가한다. A/B/C에서 고른 best pretrain checkpoint가 있으면 `--pretrain-checkpoint`로 전달한다.

```bash
python experiments/scripts/run_d_sentiment.py \
  --pretrain-checkpoint /content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/A1_20260602_JAEHWAN/checkpoints/A1_20260602_step0800_best.pt \
  --output-dir /content/drive/MyDrive/gpt-lab/experiment_outputs/sentiment/D4_20260602_HYEONGMIN
```

## A/B/C 공통 옵션

| 옵션 | 의미 |
| --- | --- |
| `--experiment` | 실행할 실험 ID 또는 그룹 ID. 기본값은 `all` |
| `--quick` | smoke test용 작은 설정 사용 |
| `--output-root` | 산출물 저장 루트 |
| `--vocab-size` | BPE vocabulary 크기 |
| `--num-epochs` | 기본 2 epoch 대신 실행할 epoch 수 |
| `--eval-iter` | validation loss 계산에 사용할 batch 수 |
| `--tokenizer-path` | 기본 공유 tokenizer가 아닌 파일을 직접 지정 |
| `--force-tokenizer-train` | 기존 tokenizer가 있어도 다시 학습 |
| `--train-char-limit` | 모델 학습에 사용할 train corpus 문자 수 제한 |
| `--val-char-limit` | validation corpus 문자 수 제한 |
| `--train-token-limit` | tokenizer 인코딩 후 train token 수 제한 |
| `--val-token-limit` | tokenizer 인코딩 후 validation token 수 제한 |
| `--install` | 실행 전 `pip install -r requirements.txt` 수행 |
| `--skip-download` | 데이터 다운로드를 건너뜀. 데이터가 없으면 실패 |
| `--dry-run` | 학습하지 않고 config와 output 디렉토리만 생성 |
| `--log-every-steps` | train metric을 JSONL에 기록할 step 간격 |
| `--eval-every-steps` | validation 평가 및 best 후보 갱신 step 간격 |
| `--save-every-steps` | latest checkpoint 저장 step 간격 |
| `--keep-latest` | 유지할 latest checkpoint 개수 |
| `--device cpu/cuda` | 계산 장치 강제 지정 |

## D 주요 옵션

| 옵션 | 의미 |
| --- | --- |
| `--quick` | smoke test용 작은 설정 사용 |
| `--output-dir` | D 감성 분류 산출물을 저장할 개별 실행 디렉토리 |
| `--pretrain-checkpoint` | A/B/C에서 선택한 pretrain checkpoint 경로 |
| `--vocab-size` | BPE vocabulary 크기 |
| `--tokenizer-path` | 기본 공유 tokenizer가 아닌 파일을 직접 지정 |
| `--force-tokenizer-train` | 기존 tokenizer가 있어도 다시 학습 |
| `--install` | 실행 전 `pip install -r requirements.txt` 수행 |
| `--skip-download` | 데이터 다운로드를 건너뜀. 데이터가 없으면 실패 |

## 산출물 구조

A/B/C 사전 학습 산출물은 다음 구조로 저장한다.

```text
{output-root}/
  {실험ID}_{YYYYMMDD}_{담당자}/
    run_config.json
    summary.json
    checkpoints/
      {실험ID}_{YYYYMMDD}_step{global_step}_latest.pt
      {실험ID}_{YYYYMMDD}_step{global_step}_best.pt
    logs/
      {실험ID}_{YYYYMMDD}.out
    metrics/
      {실험ID}_{YYYYMMDD}_metrics.jsonl
    plots/
```

D 감성 분류 산출물은 다음 구조로 저장한다.

```text
{output-dir}/
  run_config.json
  summary.json
  checkpoints/
    D0_{YYYYMMDD}_step{global_step}_latest.pt
    D0_{YYYYMMDD}_step{global_step}_best.pt
    D2_{YYYYMMDD}_step{global_step}_latest.pt
    D2_{YYYYMMDD}_step{global_step}_best.pt
    D3_{YYYYMMDD}_step{global_step}_latest.pt
    D3_{YYYYMMDD}_step{global_step}_best.pt
  logs/
    D_{YYYYMMDD}_HYEONGMIN.md
  metrics/
    D_{YYYYMMDD}_metrics.jsonl
  tokenizers/
    D_tokenizer_vocab{vocab_size}_quick.json
```

D0/D2/D3 행은 validation 후보 학습 결과만 담는다. `summary.json`과 `logs/D_{YYYYMMDD}_HYEONGMIN.md`의 D4 항목에 선택된 checkpoint의 test loss/test accuracy가 기록된다.

## 공유 tokenizer 정책

공식 실험에서는 전체 `data/nsmc_lm_train.txt`로 BPE tokenizer를 학습한다. 공유 tokenizer가 이미 있으면 새로 학습하지 않고 재사용한다.

```text
artifacts/tokenizers/nsmc_bpe_vocab{vocab_size}_full.json
```

`--quick` 실행은 공식 결과가 아니므로 output 디렉토리 아래 임시 tokenizer를 사용할 수 있다.

## Git에 올리지 않는 파일

다음 파일은 개인 Google Drive 또는 `local/experiment_outputs/`에만 보관한다.

- `.pt`, `.pth` checkpoint
- raw metric JSONL
- stdout/stderr log
- plot 이미지
- 개인 Drive 경로가 포함된 대용량 산출물
