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

### 2.1 Colab 환경 통일 방법

Colab은 같은 GPU가 항상 배정되지 않을 수 있으므로, 실험 시작 전에 환경을 확인하고 기록한다. 가능한 경우 4명 모두 같은 GPU와 같은 Python major/minor 버전에서 실행한다.

#### Colab 런타임 설정

1. Colab 상단 메뉴에서 `Runtime > Change runtime type`을 연다.
2. `Runtime type`은 `Python 3`으로 설정한다.
3. `Hardware accelerator`는 `GPU`로 설정한다.
4. `Runtime shape` 또는 RAM 옵션이 보이면 전원 동일하게 맞춘다. 기본값을 우선 사용하고, High-RAM은 전원이 사용할 수 있을 때만 사용한다.
5. GPU는 가능하면 전원 `T4`로 맞춘다. `L4`, `A100` 등 다른 GPU가 섞이면 loss/accuracy 비교는 가능하지만, 학습 시간 비교는 GPU별로 분리한다.

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
| Python | major.minor 버전 동일 | 다르면 런타임 재시작 후 재확인, 그래도 다르면 로그에 기록 |
| GPU | 가능하면 전원 T4 | GPU가 섞이면 시간 비교 제외 또는 GPU별 표 분리 |
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
| A0 | baseline | 공통 기준 설정 |
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
| D4 | best checkpoint 선택 | validation loss 최저 checkpoint로 test 평가 |

D는 최종 단계에서 A/B/C가 제출한 best pretrain checkpoint를 사용해 D2~D4를 한 번 더 실행한다.

## 7. 권장 실행 순서

### 7.1 빠른 smoke test

각 팀원은 자기 실험을 시작하기 전에 작은 설정으로 1회 실행해 코드가 끝까지 도는지 확인한다.

| 항목 | 값 |
| --- | --- |
| num_epochs | 1 |
| batch_size | 2 |
| context_length | 32 또는 64 |
| eval_iter | 작게 설정 |

### 7.2 1차 screening

1차 실험은 많은 후보를 빠르게 거르는 단계다.

- 모든 후보는 2 epoch 기준으로 실행한다.
- validation loss가 명확히 나쁜 후보는 중단한다.
- NaN, OOM, loss 발산이 발생하면 즉시 기록하고 종료한다.

### 7.3 2차 confirmation

2차 실험은 1차에서 좋은 후보만 다시 확인하는 단계다.

- 각 담당자는 best 후보 1~2개만 3 epoch 이상 재실험한다.
- A의 안정화 기법과 B/C의 best 후보를 조합해 최종 pretrain 후보를 만든다.
- D는 최종 pretrain checkpoint로 fine-tuning을 실행한다.

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
| 필수 | A0 baseline, D0 sentiment baseline, 각 담당자의 핵심 비교 1개 이상 |
| 필수 | A: warmup+cosine 또는 clipping 중 최소 1개 |
| 필수 | B: learning_rate 3개 후보 비교 |
| 필수 | C: context_length 또는 n_layers 비교 중 최소 1개 |
| 필수 | D: class imbalance, best checkpoint 선택 |
| 선택 | batch_size 전체 비교, drop_rate 전체 비교, emb_dim 전체 비교, weight_decay 추가 비교 |

20분 이상 막힌 실험은 즉시 중단하고 실패 로그를 남긴다. 2026-06-03 오전에는 새 실험을 넓히지 않고, 2026-06-02에 선정한 후보를 확인하는 데 집중한다.

### 8.3 구체 일정

| 날짜 | 시간 | 목표 | 산출물 |
| --- | --- | --- | --- |
| 2026-06-02 화 | 10:00~10:30 | kickoff, baseline 기준 확정, 실험 로그 양식 확정 | 공통 config, 실험 ID, 담당자별 첫 실행 목록 |
| 2026-06-02 화 | 10:30~11:50 | smoke test 실행, Colab GPU/데이터 경로 확인 | 각자 코드 실행 가능 여부, 실패 환경 기록 |
| 2026-06-02 화 | 11:50~13:00 | 점심시간 | 실행 중인 Colab만 유지 |
| 2026-06-02 화 | 13:00~15:00 | baseline 집중 실행 | A0 pretrain baseline, D0 sentiment baseline |
| 2026-06-02 화 | 15:00~17:30 | 1차 screening 병렬 실행 | A1/A2 중 1개 이상, B2, C1 또는 C2, D1/D4 초안 |
| 2026-06-02 화 | 17:30~17:50 | 중간 결과 저장, 실패 실험 정리 | baseline 대비 1차 결과표, OOM/NaN 로그 |
| 2026-06-02 화 | 17:50~19:00 | 저녁시간 | 실행 중인 Colab만 유지 |
| 2026-06-02 화 | 19:00~20:30 | 남은 1차 screening 실행 | B1/B3/C3/A3 중 가능한 선택 실험 |
| 2026-06-02 화 | 20:30~21:00 | 1차 결과 리뷰, 최종 후보 확정 | 버릴 후보, 유지할 후보, 6/3 확인 실험 2~3개 |
| 2026-06-02 화 | 21:00~22:30 | 최종 후보 재실험 시작 | best hparam 후보, best architecture 후보 로그 |
| 2026-06-02 화 | 22:30 이후 | 긴 confirmation 실험 자동 실행 | checkpoint 저장 설정된 overnight 실험 |
| 2026-06-03 수 | 10:00~10:30 | overnight 결과 확인, 실패 실험 판정 | 성공/실패 로그, 최종 재실행 필요 여부 |
| 2026-06-03 수 | 10:30~11:50 | 최종 confirmation 및 sentiment test 평가 | best pretrain 후보, best sentiment 후보 |
| 2026-06-03 수 | 11:50~13:00 | 점심시간 | 실행 중인 Colab만 유지 |
| 2026-06-03 수 | 13:00~13:40 | 최종 checkpoint 선정, 누락 지표 보완 | best pretrain checkpoint, best sentiment checkpoint |
| 2026-06-03 수 | 13:40~14:20 | 그래프 생성, 결과표 작성 | loss curve, hparam 결과표, sentiment 결과표 |
| 2026-06-03 수 | 14:20~14:40 | `REPORT.md`와 발표 자료에 결과 반영 | 최종 수치, 실패 실험 요약, 발표 목차 |
| 2026-06-03 수 | 14:40~15:00 | 최종 결과 freeze, 발표 핵심 메시지 확정 | 최종 결론 3개, 예상 질문 답변 초안 |
| 2026-06-04 목 | 10:00 | 발표 | 최종 발표 |

회의는 길게 토론하지 않고 다음 3가지만 결정한다.

- 지금까지 가장 좋은 설정은 무엇인가?
- 어떤 실험은 더 이상 돌리지 않을 것인가?
- 다음에 누가 어떤 실험 ID를 실행할 것인가?

## 9. 실험 로그 규칙

모든 실험은 같은 형식으로 기록한다. 파일명은 다음 규칙을 사용한다.

```text
logs/{실험ID}_{날짜}_{담당자}.md
checkpoints/{실험ID}_{날짜}_best.pt
plots/{실험ID}_{날짜}_loss.png
```

예시:

```text
logs/B2_20260602_B.md
checkpoints/D4_20260604_best.pt
plots/A1_20260602_loss.png
```

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
| A0 | baseline |  |  |  |  |
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

| 실험 ID | 변경점 | best val loss | val acc | test acc | 결론 |
| --- | --- | --- | --- | --- | --- |
| D0 | baseline |  |  |  |  |
| D2 | freeze |  |  |  |  |
| D3 | lr 분리 |  |  |  |  |
| D4 | best checkpoint |  |  |  |  |

## 12. 최종 산출물

최종 제출 전까지 다음 항목을 준비한다.

- `REPORT.md`에 최종 실험 결과 반영
- loss curve 이미지
- best pretrain checkpoint 경로
- best sentiment checkpoint 경로
- 실험별 로그 파일
- 실패한 실험과 실패 원인
- 최종 선택 설정과 선택 이유

최종 결론은 다음 구조로 작성한다.

```text
우리는 baseline 대비 어떤 설정을 바꿨고,
그 결과 validation loss 또는 accuracy가 얼마나 변했으며,
학습 시간과 안정성까지 고려했을 때 어떤 설정을 최종 선택했다.
```
