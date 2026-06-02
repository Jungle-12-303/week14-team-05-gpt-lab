# 추가 미션 실험 계획서

## 1. 목표

이 문서는 4명이 병렬로 추가 미션 실험을 진행하기 위한 실행 계획이다. 목표는 단순히 많은 조합을 돌리는 것이 아니라, 같은 기준선에서 변수를 통제하며 실험하고 재현 가능한 근거로 최종 설정을 선택하는 것이다.

주요 질문은 다음과 같다.

- 사전 학습에서 warmup, cosine decay, gradient clipping, weight decay가 loss 안정성과 validation loss를 개선하는가?
- batch size, learning rate, dropout, context length, layer 수, embedding 차원이 성능과 학습 시간에 어떤 영향을 주는가?
- 감성 분류 fine-tuning에서 freeze, learning rate 분리, class imbalance 확인, best checkpoint 선택이 validation/test 성능을 개선하는가?

## 2. 공통 실험 원칙

모든 실험은 한 번에 하나의 핵심 변수만 바꾼다. 이렇게 해야 성능 차이가 어떤 변경 때문인지 해석할 수 있다.

공통 기준 설정은 다음과 같다.

| 항목 | 기본값 |
| --- | --- |
| 실행 환경 | Colab GPU |
| seed | 42 |
| batch_size | 8 |
| learning_rate | 3e-4 |
| drop_rate | 0.1 |
| context_length | 64 |
| n_layers | 2 |
| emb_dim | 128 |
| n_heads | 4 |
| qkv_bias | False |
| optimizer | AdamW |
| weight_decay | 0.0 또는 baseline 코드 기본값 |
| num_epochs | 2~3 |
| eval 기준 | 동일한 eval_freq, eval_iter 사용 |

공통으로 맞춰야 할 기준 변수는 아래 설정을 사용한다. 각 담당자는 실험에서 바꾸는 변수만 명시적으로 변경하고, 나머지 값은 이 기준을 유지한다.

```python
BASE_CONFIG = {
    "vocab_size": 3000,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TRAIN_CONFIG = {
    "seed": 42,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
    "num_epochs": 2,
    "eval_freq": 100,
    "eval_iter": 10,
    "start_context": "영화",
}

SAVE_CONFIG = {
    "log_every_steps": 20,
    "eval_every_steps": 100,
    "save_every_steps": 100,
    "keep_latest": 2,
}

SENTIMENT_CONFIG = {
    "max_length": 64,
    "freeze_blocks": 1,
    "backbone_lr": 1e-4,
    "classifier_lr": 3e-4,
}

SMOKE_CONFIG = {
    **TRAIN_CONFIG,
    "batch_size": 2,
    "num_epochs": 1,
    "eval_freq": 20,
    "eval_iter": 2,
    "log_every_steps": 10,
    "eval_every_steps": 20,
    "save_every_steps": 20,
}
```

`BASE_CONFIG`와 `TRAIN_CONFIG`는 사전 학습과 공통 비교의 기준이다. `SAVE_CONFIG`는 Colab 중단에 대비한 step 단위 저장 주기를 통제한다. `SENTIMENT_CONFIG`는 D 감성 분류 fine-tuning 전용 설정이며, 사전 학습의 `context_length`와 감성 분류 입력 길이인 `max_length`를 혼동하지 않는다.

### 2.1 실행 규모와 제출 기준

공지의 Basic 제출 기준과 실험 탐색용 screening 기준은 분리해서 기록한다. Smoke/Light는 코드와 저장 정책이 정상 동작하는지 확인하고 후보를 빠르게 거르는 용도이며, Basic은 최종 제출 근거로 사용할 수 있는 최소 기준이다.

| 구분 | 목적 | 모델 학습 입력 규모 | vocab_size | context_length | 판단 |
| --- | --- | ---: | ---: | ---: | --- |
| Smoke | 실행 확인 | `corpus[:5000]` 수준 | 300 | 32 | 공식 성능 비교에 사용하지 않음 |
| Light | 빠른 후보 선별 | `corpus[:500000]` 수준 | 2000 | 64 | 1차 screening 근거 |
| Basic | 제출 최소 기준 | `corpus[:1500000]` 수준 | 3000 | 128 | 최종 후보 검증 기준 |

BPE tokenizer 정책은 위 모델 학습 입력 규모와 별도로 고정한다. 공식 실험에서는 전체 `data/nsmc_lm_train.txt`로 `artifacts/tokenizers/nsmc_bpe_vocab{vocab_size}_full.json`을 만들고, 이미 같은 `vocab_size`의 공유 tokenizer가 있으면 새로 학습하지 않고 해당 파일을 사용한다. `vocab_size`가 다른 실험은 별도 공유 tokenizer 파일을 만든다. 현재 NSMC LM train corpus가 1,500,000자보다 짧으면 Basic의 `corpus[:1500000]`은 전체 train corpus 사용과 같다.

현재 A/B/C 기본 screening config의 `context_length=64`는 후보를 빠르게 비교하기 위한 값이다. A의 `A0`는 `--vocab-size 2000 --train-char-limit 500000`로 실행하는 screening baseline이고, `A0_basic`은 `--vocab-size 3000 --train-char-limit 1500000`로 실행하는 Basic 제출 기준 baseline이다. 최종 후보는 가능하면 Basic 기준인 `vocab_size=3000`, `context_length=128`로 한 번 더 확인한다. 시간이 부족해 최종 선택이 `context_length=64` 결과에 기반한다면, validation loss, 학습 시간, GPU 제약 때문에 Basic 확인을 생략했음을 결과 문서와 `REPORT.md`에 명시한다.

### 2.2 Colab 환경 통일 방법

Colab은 같은 GPU가 항상 배정되지 않을 수 있으므로, 실험 시작 전에 환경을 확인하고 기록한다. 가능한 경우 4명 모두 같은 GPU와 같은 Python major/minor 버전에서 실행한다.

#### Colab 웹 런타임 설정

1. Colab 상단 메뉴에서 `Runtime > Change runtime type`을 연다.
2. `Runtime type`은 `Python 3`으로 설정한다.
3. `Hardware accelerator`는 `GPU`로 설정한다.
4. `Runtime version` 옵션이 보이면 4명 모두 `Latest`로 통일한다.
5. `Runtime shape` 또는 RAM 옵션이 보이면 4명 모두 동일하게 맞춘다. 기본값을 우선 사용하고, High-RAM은 4명 모두 사용할 수 있을 때만 사용한다.
6. GPU는 가능하면 4명 모두 `T4`로 맞춘다. `L4`, `A100` 등 다른 GPU가 섞이면 loss/accuracy 비교는 가능하지만, 학습 시간 비교는 GPU별로 분리한다.

#### VS Code Colab 확장 설정

VS Code에서 Colab 확장을 사용하는 경우에는 Colab 웹의 `Runtime > Change runtime type` 대신 노트북 우측 상단의 커널 선택 메뉴를 사용한다.

1. 실험용 `.ipynb` 파일을 VS Code에서 연다.
2. 노트북 우측 상단의 `Select Kernel`을 클릭한다.
3. 커널 목록에서 `Colab`을 선택한다.
4. 가능한 경우 `Auto Connect`보다 `New Colab Server`를 선택해 새 Colab 서버를 요청한다.
5. 서버 옵션에서 4명 모두 같은 조건을 선택한다.
   - Runtime version: `Latest`
   - Machine type: `GPU`
   - RAM/shape: `Standard` 또는 `High-RAM` 중 하나로 통일
   - Accelerator가 표시되면 가능한 경우 `T4`로 통일
6. 연결 후 실제 배정된 GPU, Python 버전, PyTorch/CUDA 버전은 첫 번째 셀에서 반드시 확인하고 기록한다.

`Latest` 런타임은 Colab 업데이트에 따라 실제 Python 또는 기본 패키지 버전이 바뀔 수 있다. 따라서 같은 실험 묶음에서는 4명 모두 `Latest`로 통일하되, 실험 시작 시 실제 Python, PyTorch, CUDA, GPU 정보를 반드시 기록한다.

VS Code 확장을 사용해도 Colab GPU가 항상 고정 배정되는 것은 아니다. 따라서 설정을 맞춘 뒤에도 실제 배정 GPU가 모두 `T4`이면 학습 시간까지 직접 비교하고, `T4`, `L4`, `A100` 등 GPU가 섞이면 loss/accuracy는 비교하되 학습 시간은 GPU 종류별로 분리해 기록한다.

#### Colab 실행 위치와 Google Drive 산출물 저장소

실험은 Colab GPU T4, Runtime `Latest` 환경을 기본으로 진행한다. 실행용 Git repo는 Colab의 빠른 임시 디스크인 `/content` 아래에 clone하고, checkpoint와 log 같은 산출물은 처음부터 개인 Google Drive 아래에 저장한다.

Google Drive는 `/content`보다 파일 입출력이 느릴 수 있으므로 repo 전체를 Drive에서 실행하지 않는다. 대신 코드 실행과 패키지 import는 `/content/week14-team-05-gpt-lab`에서 수행하고, A/B/C 사전 학습 runner는 `--output-root`를 Drive의 `pretrain/` 루트로 지정한다. D 감성 분류 runner는 `--output-dir`를 Drive의 개별 sentiment 실행 디렉토리로 지정한다. Colab의 `/content`는 런타임이 중단되면 사라지므로, `/content`에만 남은 산출물은 임시 파일로 간주한다.

각 담당자는 Colab 시작 후 Google Drive를 마운트하고 개인 작업 루트를 만든다.

```python
from google.colab import drive

drive.mount("/content/drive")
```

권장 실행 구조는 다음과 같다.

```text
/content/week14-team-05-gpt-lab/          # 실행용 Git repo
  artifacts/tokenizers/                   # Git에 올라가는 공유 기준 tokenizer
/content/drive/MyDrive/gpt-lab/
  data_cache/                            # 선택: 다운로드 데이터 캐시
  experiment_outputs/                    # 필수: checkpoint/log/metric/plot 저장소
```

실행 예시는 다음과 같다.

```bash
cd /content
git clone https://github.com/Jungle-12-303/week14-team-05-gpt-lab.git
cd week14-team-05-gpt-lab

python experiments/scripts/run_a_pretrain_stability.py \
  --experiment A1 \
  --output-root /content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain

python experiments/scripts/run_d_sentiment.py \
  --output-dir /content/drive/MyDrive/gpt-lab/experiment_outputs/sentiment/D4_20260602_HYEONGMIN
```

최종 산출물 저장 루트는 다음과 같다. 팀원별 개인 Drive를 사용하되, 하위 구조와 파일명 규칙은 통일한다.

```text
/content/drive/MyDrive/gpt-lab/
  data_cache/
  experiment_outputs/
    pretrain/
      {실험ID}_{날짜}_{담당자}/
        checkpoints/
        logs/
        metrics/
        plots/
        tokenizers/                       # 실험 중 생성한 tokenizer 후보 또는 사본
    sentiment/
      {실험ID}_{날짜}_{담당자}/
        checkpoints/
        logs/
        metrics/
        plots/
        tokenizers/                       # 실험 중 생성한 tokenizer 후보 또는 사본
```

공유 기준 tokenizer는 Drive 산출물이 아니라 Git repo의 `artifacts/tokenizers/`에 둔다. 공식 실험에서는 전체 training corpus로 BPE tokenizer를 학습한다. 공유 tokenizer가 이미 있으면 새로 학습하지 않고 해당 파일을 사용하며, `vocab_size`가 다른 실험은 별도 공유 tokenizer 파일을 만든다. Drive의 `tokenizers/`는 실험 중 생성한 후보 또는 실행 산출물 사본으로 사용한다.

로컬 VS Code에서 결과를 검토할 때는 필요한 산출물만 `local/experiment_outputs/`로 내려받아 확인한다. `local/`, `data/`, `.pt`, `.pth` 파일은 Git에 commit하지 않고, Git에는 실행 스크립트, 실험 계획, 결과 요약 문서, 공유 기준 tokenizer, 재현 가능한 실행 명령만 올린다.

#### 실험 스크립트 위치와 역할

실험 보조 스크립트는 `experiments/scripts/` 아래에 둔다.

```text
experiments/
  README.md
  scripts/
    run_a_pretrain_stability.py
    run_b_hparams.py
    run_c_architecture.py
    run_d_sentiment.py
```

`experiments/scripts/`는 과제의 핵심 `src/` 코드를 대체하는 위치가 아니다. `src/`는 모델, tokenizer, dataset, train/fine-tune 유틸리티의 기준 구현으로 유지하고, script는 담당자별 실험 ID, 설정값, 출력 디렉토리, checkpoint/log/metric 파일명을 통일하기 위한 실행 보조 계층으로 사용한다.

A/B/C 스크립트는 실제 사전 학습 runner다. 데이터 확인, 공유 tokenizer 재사용 또는 생성, dataloader/model/optimizer 구성, step 단위 metric JSONL 저장, latest/best checkpoint 저장, `run_config.json`과 `summary.json` 기록을 자동으로 수행한다. D 스크립트는 감성 분류 fine-tuning runner이며, 공유 tokenizer 재사용, D0/D2/D3 validation 후보 학습, D4 최종 test 1회 평가를 담당한다. D runner도 step 단위 metric JSONL, latest/best checkpoint, `run_config.json`, `summary.json`, Markdown report를 저장한다.

#### 첫 번째 셀: 환경 확인

모든 팀원은 노트북 첫 번째 셀에서 아래 명령을 실행하고 결과를 실험 로그에 붙인다.

```bash
!python --version
!nvidia-smi
!pip install -r requirements.txt
```

설치 후 다음 셀로 Python, PyTorch, CUDA, 주요 패키지 버전을 기록한다.

```python
import platform
import random
import sys

import matplotlib
import numpy as np
import torch

print("python:", sys.version)
print("python_major_minor:", f"{sys.version_info.major}.{sys.version_info.minor}")
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("numpy:", np.__version__)
print("matplotlib:", matplotlib.__version__)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

#### 환경 통일 판정 기준

| 항목 | 통일 기준 | 다를 때 처리 |
| --- | --- | --- |
| Runtime version | 4명 모두 Latest | 날짜가 바뀌어 재실험하면 실제 Python/PyTorch/CUDA 버전을 다시 기록 |
| Python | major.minor 버전 동일 | 다르면 런타임 재시작 후 재확인, 그래도 다르면 로그에 기록 |
| GPU | 가능하면 4명 모두 T4 | GPU가 섞이면 시간 비교 제외 또는 GPU별 표 분리 |
| PyTorch | `requirements.txt` 설치 후 버전 기록 | 버전이 다르면 재설치 후 런타임 재시작 |
| 데이터 | 같은 데이터 파일, 같은 split, 같은 seed | 다르면 해당 실험 무효 처리 후 재실행 |
| 코드 | 같은 git commit 또는 같은 노트북 사본 | 다르면 commit/hash를 로그에 기록하고 비교 대상에서 분리 |

최종 보고서에는 GPU 이름, Python major/minor 버전, PyTorch 버전을 반드시 적는다. Colab GPU가 통일되지 않은 경우, 결론에서는 loss/accuracy 중심으로 해석하고 학습 시간은 참고 지표로만 사용한다.

실험 결과를 비교할 때는 validation loss를 1순위로 보고, 학습 안정성, 학습 시간, 생성 샘플 품질, 감성 분류 accuracy를 보조 지표로 사용한다.

## 3. 성공 기준

### 3.1 사전 학습

| 우선순위 | 지표 | 설명 |
| --- | --- | --- |
| 1 | best validation loss | 가장 낮은 validation loss |
| 2 | loss 안정성 | loss spike, NaN, 발산 여부 |
| 3 | 학습 시간 | 같은 epoch 기준 소요 시간 |
| 4 | 생성 샘플 | 같은 start_context에서 생성 품질 비교 |

### 3.2 감성 분류

| 우선순위 | 지표 | 설명 |
| --- | --- | --- |
| 1 | best validation loss | checkpoint 선택 기준 |
| 2 | validation accuracy | fine-tuning 중 모델 선택 보조 지표 |
| 3 | test accuracy | 최종 모델 1회 평가 |
| 4 | class별 성능 | label imbalance가 있으면 추가 확인 |

## 4. 팀원별 업무 분배

4명 모두 실험을 진행한다. 단, D는 감성 분류 실험과 함께 최종 리포트 취합 책임을 가진다.

담당자는 랜덤 배정했다.

| 팀원 | 담당자 | 담당 영역 | 실험 내용 | 주요 산출물 | 결과 문서 |
| --- | --- | --- | --- | --- | --- |
| A | 재환 | 사전 학습 안정화 | baseline, warmup, cosine decay, gradient clipping, weight decay | 안정화 기법별 train/val loss 비교 | [`EXPERIMENT_A_JAEHWAN.md`](./EXPERIMENT_A_JAEHWAN.md) |
| B | 영빈 | 하이퍼파라미터 탐색 1 | batch_size, learning_rate, drop_rate | 학습 설정 관련 best 후보 | [`EXPERIMENT_B_YEONGBEEN.md`](./EXPERIMENT_B_YEONGBEEN.md) |
| C | 범상 | 하이퍼파라미터 탐색 2 | context_length, n_layers, emb_dim | 모델 크기와 문맥 길이 관련 best 후보 | [`EXPERIMENT_C_BEOMSANG.md`](./EXPERIMENT_C_BEOMSANG.md) |
| D | 형민 | 감성 분류 개선 및 리포트 취합 | class imbalance, freeze, lr 분리, best checkpoint | fine-tuning 결과, 최종 실험표, 그래프 | [`EXPERIMENT_D_HYEONGMIN.md`](./EXPERIMENT_D_HYEONGMIN.md) |

각 담당자는 자기 결과 문서에 실험 환경, 실행 설정, 결과표, 실패 원인, 최종 결론을 기록한다. `EXPERIMENT_PLAN.md`는 전체 운영 계획으로 유지하고, 실제 수치는 담당자별 결과 문서에 먼저 올린 뒤 최종 발표 전 `REPORT.md`로 취합한다.

## 5. 병렬 진행 방식

처음부터 4명이 동시에 시작한다.

| 단계 | A | B | C | D |
| --- | --- | --- | --- | --- |
| 1단계 | baseline 재현, scheduler/clipping 코드 준비 | baseline 기준으로 batch/lr/drop 실험 시작 | baseline 기준으로 모델 크기 실험 시작 | label 비율 확인, sentiment baseline 실행 |
| 2단계 | warmup/cosine/clipping/weight decay 실험 | 1단계 best 후보 재실험 | 큰 모델 후보 재실험 | fine-tuning 개선 실험 |
| 3단계 | 사전 학습 best 후보 제출 | best hparam 후보 제출 | best architecture 후보 제출 | best checkpoint로 최종 fine-tuning 및 리포트 취합 |

B, C, D는 A의 코드 개선을 기다리지 않고 baseline 실험을 먼저 시작한다. A의 scheduler/clipping 코드가 준비되면, B와 C는 필요할 때 해당 옵션을 추가로 적용해 비교한다.

## 6. 실험 목록

### 6.1 A: 사전 학습 안정화 실험

| 실험 ID | 변경점 | 설정 |
| --- | --- | --- |
| A0 | screening baseline | 공통 기준 설정, `context_length=64` |
| A0_basic | Basic submission baseline | Basic 제출 기준, `vocab_size=3000`, `context_length=128` |
| A1 | warmup + cosine decay | baseline + warmup_steps 적용 + cosine decay |
| A2 | gradient clipping | baseline + max_grad_norm=1.0 |
| A3 | weight decay | baseline + weight_decay=0.01 |
| A4 | combined | warmup + cosine + clipping + best weight_decay |

### 6.2 B: 학습 하이퍼파라미터 실험

| 실험 ID | 변경점 | 후보 |
| --- | --- | --- |
| B1 | batch_size | 2, 4, 8, 16 |
| B2 | learning_rate | 1e-4, 3e-4, 5e-4 |
| B3 | drop_rate | 0.0, 0.1, 0.2 |

B는 각 실험에서 나머지 값은 공통 기준 설정으로 고정한다.

### 6.3 C: 모델 구조 하이퍼파라미터 실험

| 실험 ID | 변경점 | 후보 |
| --- | --- | --- |
| C1 | context_length | 64, 128 |
| C2 | n_layers | 1, 2, 4 |
| C3 | emb_dim | 64, 128, 192 |

C는 모델이 커질수록 학습 시간이 늘어나므로, 1차 실험은 2 epoch로 제한하고 좋은 후보만 3 epoch 이상 재실험한다.

### 6.4 D: 감성 분류 개선 실험

| 실험 ID | 변경점 | 설정 |
| --- | --- | --- |
| D0 | sentiment baseline | 현재 `src/finetune.py` 기준 |
| D1 | class imbalance 확인 | train/val/test label 0/1 개수 기록 |
| D2 | backbone 일부 freeze | embedding + 앞쪽 block freeze |
| D3 | learning rate 분리 | backbone lr < classifier lr |
| D4 | best checkpoint 선택 및 최종 test 평가 | D0/D2/D3에서 validation loss가 가장 낮은 checkpoint를 선택해 test set 1회 평가 |

D4는 별도 학습 실험이 아니라 최종 선택 단계다. D0, D2, D3 각각은 학습 중 best checkpoint를 저장하고, D4에서는 그 후보 중 validation loss가 가장 낮은 checkpoint를 선택해 test set을 1회 평가한다. D는 최종 단계에서 A/B/C가 제출한 best pretrain checkpoint를 사용해 D2/D3 후보를 다시 만들고 D4 기준으로 최종 sentiment checkpoint를 확정한다.

## 7. 권장 실행 순서

### 7.1 빠른 smoke test

각 팀원은 자기 실험을 시작하기 전에 작은 설정으로 1회 실행해 코드가 끝까지 도는지 확인한다.

| 항목 | 값 |
| --- | --- |
| 모델 학습 입력 규모 | `corpus[:5000]` 수준 |
| vocab_size | 300 |
| num_epochs | 1 |
| batch_size | 2 |
| context_length | 32 |
| eval_freq | 20 |
| eval_iter | 2 |

### 7.2 1차 screening

1차 실험은 많은 후보를 빠르게 거르는 단계다.

- 모든 후보는 2 epoch 기준으로 실행한다.
- 1차 screening은 Light 기준인 `vocab_size=2000`, `context_length=64`, `corpus[:500000]` 수준을 우선 사용한다.
- validation loss가 명확히 나쁜 후보는 중단한다.
- NaN, OOM, loss 발산이 발생하면 즉시 기록하고 종료한다.

Light screening 실행 예시는 다음과 같다.

```bash
python experiments/scripts/run_a_pretrain_stability.py \
  --experiment A0 \
  --vocab-size 2000 \
  --train-char-limit 500000 \
  --output-root /content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain
```

### 7.3 2차 confirmation

2차 실험은 1차에서 좋은 후보만 다시 확인하는 단계다.

- 각 담당자는 best 후보 1~2개만 `--num-epochs 3` 이상으로 재실험한다.
- 최종 후보는 가능하면 Basic 기준인 `vocab_size=3000`, `context_length=128`, `corpus[:1500000]` 수준으로 확인한다.
- 제출 기준 baseline은 `A0_basic`으로 실행하고, `A0`는 1차 screening baseline으로만 사용한다.
- A의 안정화 기법과 B/C의 best 후보를 조합해 최종 pretrain 후보를 만든다.
- D는 최종 pretrain checkpoint로 fine-tuning을 실행한다.

Basic baseline 실행 예시는 다음과 같다.

```bash
python experiments/scripts/run_a_pretrain_stability.py \
  --experiment A0_basic \
  --vocab-size 3000 \
  --train-char-limit 1500000 \
  --output-root /content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain
```

## 8. 회의 및 실험 사이클

기준 시간대는 KST이다. 실제 가용 시간은 2026-06-02 화요일 10:00부터 2026-06-03 수요일 15:00까지이다. 발표는 2026-06-04 목요일 10:00이므로, 모든 실험과 결과 정리는 2026-06-03 수요일 15:00까지 완료한다. 2026-06-03 15:00 이후에는 새 실험을 시작하지 않고, 결과 수치도 고정한다.

점심시간은 11:50~13:00, 저녁시간은 17:50~19:00이다. 이 시간에는 사람이 직접 판단해야 하는 실험 전환, 결과 해석, 코드 수정은 하지 않는다. 단, 이미 시작한 Colab 학습은 checkpoint와 로그 저장이 설정되어 있으면 계속 실행할 수 있다.

### 8.1 마감 기준

| 마감 | 기준 |
| --- | --- |
| 실험 시작 | 2026-06-02 화 10:00 |
| baseline 결과 확보 | 2026-06-02 화 15:00 |
| 1차 screening 종료 | 2026-06-02 화 20:30 |
| 최종 후보 확정 | 2026-06-02 화 21:00 |
| 최종 실험 종료 | 2026-06-03 수 13:40 |
| 결과표/그래프/보고서 반영 | 2026-06-03 수 14:40 |
| 실험 결과 freeze | 2026-06-03 수 15:00 |
| 발표 | 2026-06-04 목 10:00 |

### 8.2 압축 운영 원칙

가용 시간이 짧기 때문에 모든 후보를 full grid로 돌리지 않는다. 각 담당자는 담당 영역에서 가장 설명력이 큰 실험만 우선 실행한다.

| 우선순위 | 실행 기준 |
| --- | --- |
| 필수 | A0 screening baseline, A0_basic Basic baseline, D0 sentiment baseline, 각 담당자의 핵심 비교 1개 이상 |
| 필수 | A: warmup+cosine 또는 clipping 중 최소 1개 |
| 필수 | B: learning_rate 3개 후보 비교 |
| 필수 | C: context_length 또는 n_layers 비교 중 최소 1개 |
| 필수 | D: class imbalance, best checkpoint 선택 |
| 선택 | batch_size 전체 비교, drop_rate 전체 비교, emb_dim 전체 비교, weight_decay 추가 비교 |

20분 이상 막힌 실험은 즉시 중단하고 실패 로그를 남긴다. 2026-06-03 오전에는 새 실험을 넓히지 않고, 2026-06-02에 선정한 후보를 확인하는 데 집중한다.

### 8.3 구체 일정

이 일정에서는 실제 이름을 우선 표기하고, 괄호 안에 실험 ID 접두어를 함께 적는다.

- 재환(A): 사전 학습 안정화 실험
- 영빈(B): 학습 하이퍼파라미터 실험
- 범상(C): 모델 구조 하이퍼파라미터 실험
- 형민(D): 감성 분류 개선 실험 및 최종 리포트 취합

| 날짜 | 시간 | 재환(A) | 영빈(B) | 범상(C) | 형민(D) | 산출물/결정 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-02 화 | 10:00~10:30 | A0 screening config와 안정화 실험 순서 확정 | B1/B2/B3 후보와 고정 변수 확정 | C1/C2/C3 후보와 모델 크기 제한 확정 | D0~D4 실행 순서와 리포트 취합 방식 확정 | 공통 config, 실험 ID, 담당자별 첫 실행 목록 |
| 2026-06-02 화 | 10:30~11:50 | A0 quick smoke test, checkpoint 저장 확인 | B 실험용 학습 스크립트 smoke test | C 실험용 모델 설정 smoke test | D0 sentiment smoke test, label 경로 확인 | Colab GPU/데이터 경로, 코드 실행 가능 여부, 실패 환경 기록 |
| 2026-06-02 화 | 11:50~13:00 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 점심시간 |
| 2026-06-02 화 | 13:00~15:00 | A0 Light screening baseline 실행 및 train/val loss 기록 | A0와 같은 Light 기준으로 B2 learning_rate 준비 | A0와 같은 Light 기준으로 C1/C2 준비 | D0 sentiment baseline 실행 및 validation 기준 확인 | baseline 결과 확보 |
| 2026-06-02 화 | 15:00~17:30 | A1 warmup+cosine 또는 A2 clipping 중 1개 이상 실행 | B2 learning_rate 3개 후보 병렬 비교 | C1 context_length 또는 C2 n_layers 비교 | D1 class imbalance 확인, D4 best checkpoint 선택 로직 확인 | 1차 screening 결과표 초안 |
| 2026-06-02 화 | 17:30~17:50 | A0 대비 A1/A2 결과 정리 | B2 결과에서 유지/탈락 후보 표시 | C1/C2 결과에서 유지/탈락 후보 표시 | D0/D2/D3 validation 후보와 D1 class imbalance 로그 정리 | baseline 대비 1차 결과표, OOM/NaN 로그 |
| 2026-06-02 화 | 17:50~19:00 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 저녁시간 |
| 2026-06-02 화 | 19:00~20:30 | 가능하면 A3 weight_decay 또는 A4 combined 초안 실행 | 시간이 되면 B1 batch_size 또는 B3 drop_rate 추가 실행 | 시간이 되면 C3 emb_dim 추가 실행 | D2 freeze 또는 D3 lr 분리 중 가능한 실험 실행 | 선택 실험 결과와 실패 로그 |
| 2026-06-02 화 | 20:30~21:00 | 유지할 안정화 기법 1개 선정 | best learning_rate 후보 선정 | best context/model 후보 선정 | fine-tuning 유지 후보와 최종 평가 방식 선정 | 버릴 후보, 유지할 후보, 6/3 확인 실험 2~3개 |
| 2026-06-02 화 | 21:00~22:30 | 선정한 안정화 설정으로 재실험 시작 | best hparam 후보 재실험 시작 | best architecture 후보 재실험 시작 | 현재 best checkpoint 기준 fine-tuning 재실험 시작 | 최종 후보 재실험 로그 |
| 2026-06-02 화 | 22:30 이후 | checkpoint 저장된 overnight pretrain 유지 | overnight hparam confirmation 유지 | overnight architecture confirmation 유지 | overnight fine-tuning confirmation 유지 | 자동 실행 로그, checkpoint |
| 2026-06-03 수 | 10:00~10:30 | 재환 overnight 결과 성공/실패 판정 | 영빈 overnight 결과 성공/실패 판정 | 범상 overnight 결과 성공/실패 판정 | 형민 overnight 결과 성공/실패 판정 및 누락 표 확인 | 최종 재실행 필요 여부 |
| 2026-06-03 수 | 10:30~11:50 | 최종 pretrain 후보 loss 확인 | best hparam 후보 수치 확정 | best architecture 후보 수치 확정 | best pretrain checkpoint로 sentiment test 평가 | best pretrain 후보, best sentiment 후보 |
| 2026-06-03 수 | 11:50~13:00 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 실행 중인 Colab만 유지 | 점심시간 |
| 2026-06-03 수 | 13:00~13:40 | 재환 결과 문서에 최종 checkpoint와 결론 반영 | 영빈 결과 문서에 최종 hparam 결론 반영 | 범상 결과 문서에 최종 model/context 결론 반영 | 형민 결과 문서와 취합 표에 sentiment 결론 반영 | best checkpoint, 누락 지표 보완 |
| 2026-06-03 수 | 13:40~14:20 | 재환 loss curve 생성 | 영빈 hparam 결과표와 그래프 생성 | 범상 architecture 결과표와 그래프 생성 | 형민 sentiment 결과표와 전체 그래프 취합 | loss curve, hparam 결과표, sentiment 결과표 |
| 2026-06-03 수 | 14:20~14:40 | 재환 최종 수치 `REPORT.md` 반영 | 영빈 최종 수치 `REPORT.md` 반영 | 범상 최종 수치 `REPORT.md` 반영 | 형민이 `REPORT.md`와 발표 자료 전체 취합 | 최종 수치, 실패 실험 요약, 발표 목차 |
| 2026-06-03 수 | 14:40~15:00 | 재환 결론 1개와 예상 질문 작성 | 영빈 결론 1개와 예상 질문 작성 | 범상 결론 1개와 예상 질문 작성 | 형민 최종 결론 3개와 예상 질문 답변 정리 | 최종 결과 freeze, 발표 핵심 메시지 |
| 2026-06-04 목 | 10:00 | 사전 학습 안정화 결과 발표 보조 | 하이퍼파라미터 결과 발표 보조 | 모델 구조 결과 발표 보조 | 최종 리포트와 감성 분류 결과 발표 | 최종 발표 |

회의는 길게 토론하지 않고 다음 3가지만 결정한다.

- 지금까지 가장 좋은 설정은 무엇인가?
- 어떤 실험은 더 이상 돌리지 않을 것인가?
- 다음에 누가 어떤 실험 ID를 실행할 것인가?

## 9. 실험 산출물 저장 및 로그 규칙

모든 실험 산출물은 Google Drive의 `experiment_outputs/` 아래에 저장한다. Git repo는 `/content/week14-team-05-gpt-lab`에서 실행한다. A/B/C 사전 학습 runner는 `--output-root /content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain`을 받아 내부에 `{실험ID}_{날짜}_{담당자}/` 디렉토리를 만든다. D 감성 분류 runner는 `--output-dir /content/drive/MyDrive/gpt-lab/experiment_outputs/sentiment/{실험ID}_{날짜}_{담당자}`를 직접 받는다. Colab `/content`의 산출물 경로는 캐시 또는 임시 작업 경로로만 사용한다.

공유 기준 tokenizer만 예외적으로 Git의 `artifacts/tokenizers/`에 저장한다. raw metric JSONL, stdout/stderr log, checkpoint는 개인 Google Drive에만 남긴다.

A/B/C 사전 학습 runner의 파일명은 다음 규칙을 사용한다.

```text
run_config.json
summary.json
logs/{실험ID}_{날짜}.out
metrics/{실험ID}_{날짜}_metrics.jsonl
checkpoints/{실험ID}_{날짜}_step{global_step}_latest.pt
checkpoints/{실험ID}_{날짜}_step{global_step}_best.pt
plots/{실험ID}_{날짜}_loss.png
tokenizers/{실험ID}_tokenizer_vocab{vocab_size}_full.json
artifacts/tokenizers/nsmc_bpe_vocab{vocab_size}_full.json
```

D 감성 분류 runner의 파일명은 다음 규칙을 사용한다.

```text
run_config.json
summary.json
logs/D_{날짜}_HYEONGMIN.md
metrics/D_{날짜}_metrics.jsonl
checkpoints/D0_{날짜}_step{global_step}_latest.pt
checkpoints/D0_{날짜}_step{global_step}_best.pt
checkpoints/D2_{날짜}_step{global_step}_latest.pt
checkpoints/D2_{날짜}_step{global_step}_best.pt
checkpoints/D3_{날짜}_step{global_step}_latest.pt
checkpoints/D3_{날짜}_step{global_step}_best.pt
```

D0/D2/D3는 validation loss 기준 후보 checkpoint를 만들고, D4는 별도 학습을 하지 않는다. D4는 D0/D2/D3 중 validation loss가 가장 낮은 checkpoint를 선택한 뒤 test set을 1회만 평가하고 그 결과를 `metrics/D_{날짜}_metrics.jsonl`, `summary.json`, `logs/D_{날짜}_HYEONGMIN.md`에 기록한다.

예시:

```text
/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/A1_20260602_JAEHWAN/logs/A1_20260602.out
/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/A1_20260602_JAEHWAN/checkpoints/A1_20260602_step0800_best.pt
/content/drive/MyDrive/gpt-lab/experiment_outputs/sentiment/D4_20260602_HYEONGMIN/checkpoints/D3_20260602_step0400_best.pt
```

### 9.1 저장 단위

checkpoint 저장은 epoch 종료 시점에만 의존하지 않는다. Colab 런타임 중단에 대비해 학습 step 또는 evaluation step 단위로 작은 산출물을 즉시 Google Drive에 남긴다.

아래 저장 단위는 A/B/C 사전 학습 runner와 D 감성 분류 runner에 공통으로 적용한다. D의 Markdown report는 stdout/stderr 원본 로그가 아니라 후보 선택과 D4 test 결과를 요약한 결과 문서다.

| 산출물 | 저장 단위 | 보존 정책 |
| --- | --- | --- |
| stdout/stderr log | 실행 시작부터 파일에 계속 append | 전체 보존 |
| metric JSONL | `log_every_steps` 또는 `eval_every_steps`마다 1줄 append | 전체 보존 |
| latest checkpoint | `save_every_steps`마다 저장 | 최신 1개 또는 최근 2개만 유지 |
| best checkpoint | validation loss 개선 시 저장 | 실험별 1개 보존 |
| tokenizer | vocab 설정별 1회 저장 | 같은 vocab 설정이면 재사용 |
| plot | 실험 종료 또는 중간 요약 시 생성 | 최종본 보존 |

권장 기본값은 다음과 같다.

| 설정 | smoke test | 1차 screening | confirmation |
| --- | ---: | ---: | ---: |
| `log_every_steps` | 10 | 20 | 20 |
| `eval_every_steps` | 20 | 100 | 100 |
| `save_every_steps` | 20 | 100 | 100~200 |

`global_step`은 optimizer update 기준의 누적 step으로 기록한다. epoch, batch index만으로 checkpoint를 식별하지 않는다.

### 9.2 공유 기준 자산과 raw 산출물 구분

공유 기준 자산은 모든 팀원이 같은 기준으로 재사용해야 하는 작은 파일이다. 현재 Git에 올릴 공유 기준 자산은 tokenizer vocabulary로 제한한다.

```text
artifacts/
  tokenizers/
    nsmc_bpe_vocab3000_full.json
    nsmc_bpe_vocab5000_full.json
```

공식 실험에서는 전체 training corpus로 공유 tokenizer를 학습한다. 공유 tokenizer 파일명에는 `vocab_size`와 전체 corpus 기준임을 나타내는 `full`을 포함한다. tokenizer를 다시 학습해 기준을 바꾸는 경우에는 기존 파일을 덮어쓰기보다 새 파일명으로 추가하고, 결과 문서에 어떤 tokenizer를 사용했는지 기록한다.

raw metric JSONL은 실행 중 step마다 계속 쌓이는 개인 실험 로그이므로 Git에 올리지 않는다. 최종 비교에 필요한 수치만 담당자별 결과 문서 또는 `REPORT.md`에 표로 정리한다.

### 9.3 Git에 올리지 않는 파일

다음 파일은 개인 Google Drive 또는 로컬 `local/experiment_outputs/`에만 보관하고 Git에 commit하지 않는다.

- NSMC 원본/가공 데이터
- `.pt`, `.pth` checkpoint
- 실험별 임시 tokenizer vocabulary JSON
- 긴 stdout/stderr 원본 로그
- raw metric JSONL
- 개인 Google Drive 경로가 포함된 대용량 산출물

Git에는 결과 요약 문서, 실행 명령, 주요 수치, best checkpoint의 Drive 경로, 공유 기준 tokenizer만 기록한다.

로그에는 최소한 아래 내용을 남긴다.

| 항목 | 내용 |
| --- | --- |
| experiment_id | 예: B2 |
| owner | A/B/C/D |
| date | YYYY-MM-DD |
| git commit | 가능하면 기록 |
| Colab GPU | T4, L4, A100 등 |
| Python version | 예: 3.x |
| PyTorch version | 예: 2.x |
| CUDA version | `torch.version.cuda` 값 |
| seed | 42 |
| 변경한 변수 | 예: learning_rate=1e-4 |
| 고정한 변수 | baseline config |
| train loss | epoch별 |
| validation loss | epoch별 |
| best validation loss | 숫자 |
| 소요 시간 | 분 단위 |
| checkpoint path | 경로 |
| output root | Google Drive 산출물 루트 |
| global_step | checkpoint/metric 저장 step |
| 특이사항 | OOM, NaN, runtime disconnect 등 |
| 결론 | keep/drop/retry 중 하나 |

## 10. 의사결정 규칙

다음 기준으로 후보를 선택한다.

- validation loss가 가장 낮은 후보를 우선 선택한다.
- validation loss 차이가 1~2% 이내이면 더 작고 빠른 모델을 선택한다.
- loss가 낮아도 학습 시간이 과도하게 길면 최종 후보에서 제외할 수 있다.
- 감성 분류는 마지막 epoch가 아니라 validation loss가 가장 낮은 checkpoint를 사용한다.
- test set은 최종 후보가 정해진 뒤 최소한으로 평가한다.

중단 기준은 다음과 같다.

- GPU OOM이 반복되는 설정
- 2회 이상 NaN 발생
- baseline보다 validation loss가 명확히 나쁜 설정
- 같은 성능인데 학습 시간이 2배 이상 긴 설정

## 11. 최종 보고서에 넣을 표

### 11.1 사전 학습 결과

| 실험 ID | 변경점 | best val loss | train loss | 시간 | 결론 |
| --- | --- | --- | --- | --- | --- |
| A0 | screening baseline |  |  |  |  |
| A0_basic | Basic submission baseline |  |  |  |  |
| A1 | warmup + cosine |  |  |  |  |
| A2 | gradient clipping |  |  |  |  |
| A3 | weight decay |  |  |  |  |

### 11.2 하이퍼파라미터 결과

| 실험 ID | 변경 변수 | 값 | best val loss | 시간 | 결론 |
| --- | --- | --- | --- | --- | --- |
| B1 | batch_size | 2/4/8/16 |  |  |  |
| B2 | learning_rate | 1e-4/3e-4/5e-4 |  |  |  |
| B3 | drop_rate | 0.0/0.1/0.2 |  |  |  |
| C1 | context_length | 64/128 |  |  |  |
| C2 | n_layers | 1/2/4 |  |  |  |
| C3 | emb_dim | 64/128/192 |  |  |  |

### 11.3 감성 분류 결과

| 실험 ID | 변경점 | best val loss | val acc | best checkpoint | test acc | 결론 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | baseline |  |  |  | test 미평가 |  |
| D2 | freeze |  |  |  | test 미평가 |  |
| D3 | lr 분리 |  |  |  | test 미평가 |  |
| D4 | D0/D2/D3 중 best checkpoint 선택 후 test 1회 평가 |  |  |  |  |  |

## 12. 최종 산출물

최종 제출 전까지 다음 항목을 준비한다.

- `REPORT.md`에 최종 실험 결과 반영
- loss curve 이미지
- Google Drive 산출물 루트
- best pretrain checkpoint의 Google Drive 경로
- best sentiment checkpoint의 Google Drive 경로
- 사용한 공유 tokenizer vocabulary 경로
- A/B/C raw metric JSONL의 Google Drive 경로
- D 결과 report Markdown의 Google Drive 경로
- 실험별 로그 파일
- 실패한 실험과 실패 원인
- 최종 선택 설정과 선택 이유

최종 결론은 다음 구조로 작성한다.

```text
우리는 baseline 대비 어떤 설정을 바꿨고,
그 결과 validation loss 또는 accuracy가 얼마나 변했으며,
학습 시간과 안정성까지 고려했을 때 어떤 설정을 최종 선택했다.
```
