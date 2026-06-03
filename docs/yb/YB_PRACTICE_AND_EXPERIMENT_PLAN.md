# YB 관점 실행 계획: 코드 뼈대 이후 실습·실험 순서

## 1. 이 문서의 목적

이 문서는 과제 공지와 현재 저장소 상태를 함께 보고, "코드 뼈대를 지금처럼 작성한 뒤 다음에 무엇을 어떤 순서로 확인하고 실험할 것인가"를 영빈 기준으로 다시 정리한 문서다.

핵심은 두 가지다.

- 지금은 완전한 빈 템플릿 단계가 아니라, `src/` 핵심 구현이 이미 상당 부분 들어가 있다.
- 따라서 다음 우선순위는 TODO를 더 많이 쓰는 것이 아니라, 구현 검증과 baseline 확보, 그리고 그 다음의 실험 확장이다.

## 2. 현재 진행 상태 파악

### 2.1 과제 요구사항 기준 현재 상태

현재 저장소에는 과제의 핵심 파일이 모두 존재한다.

- `src/bpe.py`
- `src/dataset.py`
- `src/embeddings.py`
- `src/attention.py`
- `src/model.py`
- `src/train.py`
- `src/finetune.py`
- 각 단계별 `tests/test_*.py`

코드를 읽어보면 단순 골격만 있는 상태는 아니다.

- BPE tokenizer의 special token, train/save/load/encode는 구현되어 있다.
- dataset, embedding, attention, GPT model, pretrain utility, sentiment fine-tuning 코드도 상당 부분 채워져 있다.
- 즉 현재 단계는 "구현 시작 전 설계 단계"가 아니라, "구현 후 검증 및 실험 설계 단계"에 가깝다.

### 2.2 실험 문서 기준 현재 상태

`docs/`의 기존 문서를 보면 팀 전체 계획과 일부 개인 결과가 이미 존재한다.

- `EXPERIMENT_PLAN.md`: 팀 공통 실험 프레임
- `EXPERIMENT_B_YEONGBEEN.md`: batch size, learning rate, drop rate 실험 결과 기록
- `EXPERIMENT_B_LR_EXTENSION_YEONGBEEN.md`: learning rate 확장 실험 결과 기록
- `EXPERIMENT_A_JAEHWAN.md`, `EXPERIMENT_C_BEOMSANG.md`, `EXPERIMENT_D_HYEONGMIN.md`: 틀은 있으나 일부는 아직 결과 미기입

즉 영빈 파트는 이미 "실험 0회" 상태가 아니다. 특히 B 영역은 screening 결과가 꽤 나온 상태다.

### 2.3 지금 시점의 현실적 판단

내가 이 저장소를 지금 이어받는다면, 우선순위는 아래처럼 둔다.

1. 구현 정확성 검증
2. 제출용 baseline 확보
3. 기존 B 실험 결과의 해석 보정
4. A/C/D와 연결되는 최종 후보 검증
5. `REPORT.md`에 넣을 수 있는 결론으로 압축

### 2.4 확인된 제약

로컬 환경에서 `pytest` 명령은 바로 실행되지 않았다.

- 확인 결과: `zsh:1: command not found: pytest`

즉 현재 로컬 저장소만 보고는 "테스트가 모두 통과한다"는 사실까지는 확인하지 못했다. 따라서 이후 계획에서는 테스트 통과 여부를 반드시 첫 번째 게이트로 둬야 한다.

## 3. 내가 잡을 전체 전략

### 3.1 기본 원칙

지금 단계에서 가장 위험한 실수는 두 가지다.

- 구현 검증이 덜 된 상태에서 실험을 너무 많이 돌리는 것
- Light screening 결과를 그대로 최종 결론처럼 사용하는 것

그래서 나는 아래 순서를 택한다.

1. 코드 단위 검증
2. end-to-end baseline 1회 확보
3. 실험 변수 탐색
4. promising 후보만 Basic 기준으로 재검증
5. 감성 분류와 보고서 연결

### 3.2 왜 이런 순서가 필요한가

현재 `src/`는 이미 많이 구현되어 있으므로, 남은 리스크는 "코드가 아예 비어 있음"이 아니다. 진짜 리스크는 다음이다.

- shape 또는 loss 계산이 조용히 틀렸을 가능성
- Light 설정에서만 좋아 보이는 hyperparameter일 가능성
- batch size 비교처럼 update 수가 달라져 공정 비교가 아니었을 가능성
- pretrain best와 sentiment best가 다를 가능성

따라서 이 시점의 계획은 "어떤 값을 더 돌릴까"보다 "어떤 결과를 최종 결론으로 인정할까"를 정하는 것이 더 중요하다.

## 4. 내가 세울 실행 단계

## Phase 0. 구현 검증 게이트

목표:

- `src/` 구현이 테스트 기준으로 맞는지 먼저 확인한다.

실행:

1. `tests/test_bpe.py`
2. `tests/test_dataset.py`
3. `tests/test_attention.py`
4. `tests/test_model.py`
5. `tests/test_train.py`
6. `tests/test_finetune.py`
7. 마지막에 전체 `tests/`

체크포인트:

- 개별 테스트가 모두 통과해야 한다.
- 실패 시 실험으로 넘어가지 않고 해당 함수부터 수정한다.
- 특히 `decode`, causal mask, loss flattening, padding 제외 last hidden state는 우선 확인 대상이다.

산출물:

- 통과한 pytest 명령 목록
- 실패 시 에러 로그와 수정 기록

내 판단:

- 지금 저장소 상태에서는 이 단계가 가장 먼저다.
- 이유는 현재 코드가 "뼈대"를 넘어서 "실험 가능한 구현"처럼 보이기 때문이다.
- 구현 검증 없이 실험 수치를 더 쌓는 건 위험하다.

## Phase 1. 제출용 baseline 확보

목표:

- 과제 제출 기준에 맞는 최소한의 end-to-end 결과를 한 번 확보한다.

우선 baseline은 두 층으로 나눈다.

- Screening baseline: 빠른 확인용
- Basic baseline: 제출 근거용

실행 순서:

1. tokenizer 학습/로드 동작 확인
2. small corpus로 encode/decode 복원 확인
3. Light 기준 pretraining 1회
4. Basic 기준 pretraining 1회
5. Basic checkpoint로 sentiment baseline 1회

권장 기준:

| 구분 | corpus | vocab_size | context_length | 목적 |
| --- | --- | ---: | ---: | --- |
| Smoke | `[:5000]` | 300 | 32 | 코드 실행 확인 |
| Light | `[:500000]` | 2000 | 64 | 빠른 후보 선별 |
| Basic | `[:1500000]` | 3000 | 128 | 제출 근거 |

체크포인트:

- tokenizer 저장 경로가 고정돼야 한다.
- checkpoint 저장/복원이 실제로 되는지 확인한다.
- Basic baseline의 train/val loss와 sample generation을 확보한다.
- Basic checkpoint 하나로 sentiment baseline까지 이어지는지 확인한다.

산출물:

- `A0_basic` 또는 그에 준하는 baseline 기록
- best checkpoint 경로
- sample generation
- sentiment baseline val/test 기록

내 판단:

- 지금 시점에서는 B 실험이 이미 진행되어 있어도, Basic baseline이 먼저 명확해야 한다.
- baseline이 불명확하면 B의 좋은 결과도 보고서에서 중심축이 되기 어렵다.

## Phase 2. 실험 우선순위 재정렬

목표:

- 이미 진행된 B 실험을 포함해, 어떤 실험을 추가로 할지 우선순위를 다시 정한다.

내 우선순위는 아래와 같다.

1. A 안정화 기법
2. B learning rate 재검증
3. C 구조 변경 최소 검증
4. D sentiment 개선

이 순서의 이유:

- learning rate가 좋아 보여도, 안정화 기법이 들어가면 best lr가 바뀔 수 있다.
- 구조 실험은 비용이 큰 편이라, optimizer/regularization 쪽을 먼저 정리하는 편이 낫다.
- sentiment는 최종 pretrain 후보가 정리된 뒤 연결하는 것이 효율적이다.

## Phase 3. B 실험 해석 보정

현재 영빈의 기존 결과를 보면:

- batch size는 `2`가 가장 낮은 val loss
- dropout은 `0.0`이 가장 낮은 val loss
- learning rate는 `5e-4`를 넘어 `2e-3`까지 올렸을 때도 Light 기준에서 개선

하지만 이 결과는 그대로 최종 결론으로 쓰면 안 된다.

### 3.1 batch size 비교 보정

기존 B1은 epoch 수를 같게 두고 batch size를 바꿨다. 그러면 batch size가 커질수록 update step 수가 줄어든다.

따라서 B1의 현재 결론은:

- "같은 epoch 기준에서는 `bs=2`가 가장 좋았다"

까지는 말할 수 있다. 하지만

- "batch size 자체가 더 우수하다"

라고 단정하기는 어렵다.

내가 다시 한다면:

- 최종 후보 비교는 epoch 고정이 아니라 update 수 또는 token budget을 맞춘 조건으로 1회 더 확인한다.

### 3.2 learning rate 비교 보정

기존 B2와 확장 실험은 지금까지 가장 강한 결과다.

- `5e-4 -> 7e-4 -> 1e-3 -> 2e-3`로 갈수록 Light 기준 val loss가 계속 좋아졌다.

이 결과는 가치가 높다. 다만 아직 빠진 검증이 있다.

- Basic 기준에서도 같은 추세가 유지되는가
- A의 warmup/cosine/clipping을 넣었을 때도 `2e-3`가 여전히 최선인가

내가 다시 한다면:

- `3e-4`, `1e-3`, `2e-3` 정도만 골라 Basic 기준으로 재검증한다.
- Light에서만 좋고 Basic에서 흔들리는 후보는 탈락시킨다.

### 3.3 dropout 비교 보정

기존 B3에서는 `drop_rate=0.0`이 가장 좋았다.

이 해석은 현재로선 타당하다.

- 작은 모델
- 적은 epoch
- 제한된 데이터

조건에서는 regularization보다 학습 부족이 더 클 수 있기 때문이다.

다만 여기서도 최종 판단은 Basic에서 다시 봐야 한다.

- Basic에서 train loss만 빠르게 내려가고 val loss gap이 커지면 `0.0`은 과적합 후보가 된다.

## Phase 4. 최종 후보 실험만 Basic으로 재검증

목표:

- Light screening에서 살아남은 소수 후보만 Basic 기준으로 재검증한다.

내가 남길 후보 예시는 아래와 같다.

| 영역 | Light 기준 후보 | Basic 재검증 여부 |
| --- | --- | --- |
| batch size | `2`, `4` | 예 |
| learning rate | `1e-3`, `2e-3` | 예 |
| drop rate | `0.0`, `0.1` | 예 |
| 안정화 기법 | baseline, warmup+cosine, clipping, combined | 예 |

실행 원칙:

- 모든 조합을 곱해서 돌리지 않는다.
- 후보 수를 줄여야 한다.
- Basic은 "최종 확인" 용도다.

내가 고를 최소 조합:

1. Basic baseline
2. Basic + best lr
3. Basic + best lr + best 안정화
4. 필요하면 Basic + best lr + drop 0.0

이렇게 하면 실험 수를 통제하면서도 최종 결론을 낼 수 있다.

## Phase 5. 감성 분류 연결

목표:

- pretrain 후보가 실제 downstream 성능에도 이어지는지 확인한다.

실행 순서:

1. baseline pretrain checkpoint로 D0
2. best pretrain checkpoint로 D0 재실행
3. 그 다음 D2 freeze
4. 그 다음 D3 backbone/classifier lr 분리
5. validation best 1개만 D4 test

내 판단:

- pretrain loss가 가장 낮은 checkpoint가 sentiment에서도 가장 좋다는 보장은 없다.
- 따라서 sentiment는 별도 검증 단계로 반드시 남겨야 한다.
- 특히 발표/보고서에서는 "LM loss 개선이 downstream으로 연결됐는가"가 중요한 메시지가 된다.

## 5. 실제로 내가 지금 당장 할 일

우선순위를 아주 현실적으로 줄이면 아래 다섯 가지다.

1. 테스트 환경부터 복구해서 `pytest`를 실제로 돌린다.
2. Basic baseline 1개를 확실히 만든다.
3. 기존 B 결과에서 `learning_rate` 후보를 2개만 남긴다.
4. 그 2개를 Basic에서 다시 본다.
5. best pretrain 후보로 sentiment baseline을 연결한다.

즉 지금 시점에서 가장 중요한 건 새로운 변수를 더 넓히는 것이 아니라, 이미 나온 후보를 Basic과 downstream으로 검증하는 것이다.

## 6. 기존 실험 계획과의 비교

## 6.1 공통점

기존 `EXPERIMENT_PLAN.md`와 내 계획은 아래 점에서 같다.

- Smoke / Light / Basic 단계를 분리한다.
- 한 번에 하나의 핵심 변수만 바꾼다.
- checkpoint, metric JSONL, log를 남긴다.
- 최종 후보는 Basic 기준으로 다시 확인해야 한다.
- D 단계에서는 validation 기준으로 checkpoint를 고르고 test는 1회만 본다.

즉 큰 틀의 실험 철학은 동일하다.

## 6.2 차이점

차이는 초점에 있다.

### 기존 계획의 초점

기존 `EXPERIMENT_PLAN.md`는 팀 전체 추가 미션을 병렬로 분배하기 위한 운영 문서다.

- A/B/C/D 담당 분리
- 공통 환경 통일
- Colab/Drive 산출물 구조 통일
- runner/script 중심 실험 수행

즉 "여러 명이 동시에 어떻게 돌릴 것인가"에 강하다.

### 내 계획의 초점

이 문서는 영빈 관점에서, 현재 구현이 이미 어느 정도 들어간 상태를 전제로 다시 세운 실행 계획이다.

- 구현 검증을 첫 게이트로 둔다.
- baseline 확보를 추가 미션보다 앞에 둔다.
- 이미 수행된 B 결과의 해석 오류 가능성을 먼저 정리한다.
- Light 결과를 Basic과 downstream으로 축소 검증한다.

즉 "지금 이 저장소 상태에서 무엇부터 해야 최종 제출과 발표에 유리한가"에 더 초점이 있다.

## 6.3 가장 큰 차이 한 줄 요약

기존 계획이 "병렬 실험 운영 계획"이라면, 내 계획은 "현재 코드 상태를 반영한 단계형 의사결정 계획"이다.

## 6.4 왜 내가 이쪽을 더 추천하는가

지금 저장소는 이미 B 결과가 나온 상태다. 이런 시점에는 실험 공간을 더 넓히는 것보다, 아래 질문에 답하는 편이 더 중요하다.

- 이 코드가 정말 맞는가
- Basic에서도 유지되는가
- sentiment까지 이어지는가
- 보고서에 한 문장으로 설명 가능한가

내 계획은 이 질문들에 답하도록 설계돼 있다.

## 7. 최종 제안

내가 영빈이라면 지금은 이렇게 정리한다.

1. 구현 검증을 먼저 끝낸다.
2. Basic baseline을 확보한다.
3. B에서 이미 좋아 보인 `learning_rate` 중심 후보만 좁힌다.
4. 그 후보를 Basic에서 재검증한다.
5. best pretrain을 sentiment와 `REPORT.md`로 연결한다.

즉 다음 단계의 핵심은 "새 실험 많이 돌리기"가 아니라 "현재 결과를 최종 결론으로 승격시킬 증거를 만들기"다.
