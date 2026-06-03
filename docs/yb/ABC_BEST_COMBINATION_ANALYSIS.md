# A~C 결과 기반 예측 조합 제안

## 결론

현재 A, B, C 실험 결과를 함께 보면, 다음 실험은 "지금까지 가장 좋았던 값의 단순 조합"보다 "관측된 방향성을 한 단계 확장한 예측 조합"으로 가는 편이 더 낫다.

내가 가장 먼저 제안하는 조합은 아래다.

```text
vocab_size=2000
context_length=64
emb_dim=192
n_layers=4
n_heads=4
batch_size=6
learning_rate=7e-4
drop_rate=0.0
weight_decay=0.0
warmup/cosine decay/clipping 미사용
num_epochs=3
```

이 조합은 "현재 최저값"이 아니라, A~C에서 읽히는 경향을 바탕으로 다음 개선 가능성이 가장 높은 방향을 보수적으로 예측한 값이다.

## 근거

### A에서 읽히는 결론

A에서는 baseline이 가장 좋았고, 아래 기법들은 모두 성능을 개선하지 못했다.

- warmup + cosine decay
- gradient clipping
- weight decay
- 위 조합의 결합

따라서 다음 실험의 기본 전제는 명확하다.

- constant learning rate 유지
- weight decay 제거
- scheduler 제거
- clipping 제거

즉 다음 실험은 optimizer trick을 더 붙이기보다, 구조와 학습률 쪽을 보는 것이 맞다.

### B에서 읽히는 결론

#### learning rate

B2는 비교 조건이 가장 공정하다. 같은 조건에서:

- `1e-4 < 3e-4 < 5e-4`

즉 현재 범위에서는 learning rate를 올릴수록 성능이 좋아졌다. 이 경우 가장 자연스러운 추론은 하나다.

- 최적점이 아직 `5e-4`보다 오른쪽에 있을 가능성이 높다.

그래서 다음 값으로는 `7e-4`가 가장 타당하다. `1e-3`도 후보지만, 첫 확장 실험으로는 `7e-4`가 더 보수적이고 해석이 쉽다.

#### dropout

B3에서는:

- `0.0`이 `0.1`, `0.2`보다 좋았다.

즉 현재 모델 크기와 학습 길이에서는 regularization보다 학습 부족 영향이 더 크다. 따라서 다음 실험도 `drop_rate=0.0`부터 시작하는 것이 맞다.

#### batch size

B1은 `batch_size=2`가 가장 좋았지만, 그 비교는 그대로 믿으면 안 된다. 같은 epoch 기준에서 batch size가 작을수록 update step 수가 더 많았기 때문이다.

그래도 한 가지 방향성은 읽힌다.

- 더 작은 batch가 유리할 가능성은 있다.

하지만 `2`는 너무 공격적이고, `8`은 너무 보수적이다. 그래서 다음 조합에서는 둘 사이의 절충값인 `6`이 가장 설득력 있다.

`6`을 고른 이유는:

- `8`보다 update 수를 더 확보할 수 있다.
- `2`처럼 비교 왜곡을 크게 만들지 않는다.
- `n_layers=4`, `emb_dim=192` 구조에서도 실행 가능성을 유지할 확률이 높다.

### C에서 읽히는 결론

C의 구조 신호는 비교적 분명하다.

- `context_length=64`가 `128`보다 좋았다.
- `n_layers=4`가 `1`, `2`보다 좋았다.
- `emb_dim=192`가 `64`, `128`보다 좋았다.

즉 현재 실험 범위에서는 모델을 더 깊고 더 넓게 만드는 쪽이 성능에 유리했다. 그래서 구조는 굳이 보수적으로 되돌릴 이유가 없다.

따라서 구조는 그대로 아래 값을 채택한다.

- `context_length=64`
- `n_layers=4`
- `emb_dim=192`

## 왜 이 조합을 추천하는가

이 조합은 "최적 조합"이라고 주장하는 문서가 아니다. 내가 지금 실험을 하나 새로 설계한다면, 가장 먼저 던져볼 "예측 조합"이다.

핵심 논리는 단순하다.

- A는 불필요한 안정화 기법을 빼라고 말한다.
- B는 learning rate를 더 올려볼 여지가 있다고 말한다.
- B는 batch size를 약간 줄이면 더 좋아질 가능성이 있다고 말한다.
- C는 구조를 더 크게 유지하라고 말한다.

그래서 결과적으로:

- optimizer trick은 빼고
- 구조는 강하게 유지하고
- learning rate는 한 단계 더 올리고
- batch size는 한 단계 더 줄이는

방향이 가장 논리적이다.

## 대안 후보

메인 후보 외에 같이 볼 만한 값은 아래 두 개다.

### 보수적 대안

```text
context_length=64
emb_dim=192
n_layers=4
n_heads=4
batch_size=8
learning_rate=7e-4
drop_rate=0.0
weight_decay=0.0
num_epochs=3
```

이 조합은 learning rate만 확장하고 나머지는 거의 그대로 두는 안이다. 메모리나 실행 안정성이 더 중요하면 이쪽이 안전하다.

### 공격적 대안

```text
context_length=64
emb_dim=192
n_layers=4
n_heads=4
batch_size=6
learning_rate=1e-3
drop_rate=0.05
weight_decay=0.0
num_epochs=3
```

이 조합은 learning rate 상승 추세를 더 강하게 반영한 안이다. 다만 첫 실험으로는 메인 후보보다 실패 확률이 높다.

## epoch 제안

현재 문서들을 보면 대부분 핵심 비교가 `num_epochs=2`에서 이뤄졌다. 이 길이는 screening에는 충분하지만, "이 조합이 정말 낫다"를 보기에는 조금 짧다.

내 의견은 아래와 같다.

### 1차 검증: `3 epoch`

가장 먼저는 `3 epoch`를 추천한다.

이유:

- 기존 실험보다 1 epoch 더 길어, 추세가 이어지는지 확인할 수 있다.
- 계산 비용이 크게 늘지 않는다.
- `2 epoch`에서 좋아 보인 값이 3 epoch부터 무너지거나 과적합되는지 빠르게 볼 수 있다.

즉 `3 epoch`는 "경향 확인"에 가장 적절한 최소 확장이다.

### 2차 확인: `4~5 epoch`

`3 epoch`에서 아래 둘 중 하나가 보이면 `4~5 epoch`까지 늘릴 가치가 있다.

- validation loss가 아직 계속 내려간다.
- train loss와 val loss가 함께 안정적으로 내려간다.

반대로 `3 epoch` 시점에서:

- val loss가 정체되거나
- train loss만 내려가고 val loss가 반등하면

그 조합은 더 길게 돌려도 의미가 작다.

### 실전 판단 기준

정리하면 이렇게 본다.

- `2 epoch`: screening 용도
- `3 epoch`: 유의미한 비교를 시작할 최소 길이
- `4~5 epoch`: 3 epoch에서 유망한 후보만 추가 확인

따라서 지금 요청한 "예측 조합 실험"은 `num_epochs=3`으로 먼저 돌리는 것이 가장 합리적이다.

## 최종 제안

지금 한 번만 새 조합을 돌린다면 나는 아래 설정을 쓴다.

```text
vocab_size=2000
context_length=64
emb_dim=192
n_layers=4
n_heads=4
batch_size=6
learning_rate=7e-4
drop_rate=0.0
weight_decay=0.0
warmup/cosine decay/clipping 미사용
num_epochs=3
```

그리고 이 조합이 유망해 보이면, 다음 단계는 아래 순서로 간다.

1. 같은 조합으로 `4 epoch`
2. 같은 조합에서 `learning_rate=1e-3`
3. 같은 조합에서 `batch_size=8`

즉 지금은 "최적값 확정"보다 "경향을 가장 잘 따라가는 예측 조합 검증"이 우선이다.

## 실행 결과 반영

예측 조합을 실제로 `2000`과 `3000`에서 각각 실행한 결과는 아래와 같다.

### 실행 조건

공통 조합:

```text
context_length=64
emb_dim=192
n_layers=4
n_heads=4
batch_size=6
learning_rate=7e-4
drop_rate=0.0
weight_decay=0.0
warmup/cosine decay/clipping 미사용
num_epochs=3
```

비교 변수:

- run 1: `vocab_size=2000`
- run 2: `vocab_size=3000`

### 결과 요약

| run | vocab_size | tokenizer | best val loss | final global step | elapsed_min | 해석 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| predicted combo 2000 | 2000 | 새로 생성 후 저장 | 5.2275 | 2559 | 25.15 | 현재 확보된 결과 중 가장 강한 성능 신호 |
| predicted combo 3000 | 3000 | 기존 `nsmc_bpe_vocab3000_full.json` 재사용 | 5.7709 | 2286 | 5.75 | 3000에서도 경향은 유지되지만 2000보다는 불리 |

### 2000 실행 해석

`vocab_size=2000` 실행은 매우 강한 결과다.

- epoch 1: `best val loss 5.5593`
- epoch 2: `best val loss 5.2701`
- epoch 3: `best val loss 5.2275`

즉 3 epoch 내내 validation loss가 꾸준히 하락했다. 이건 두 가지를 의미한다.

- 예측 조합 자체는 유효했다.
- 아직 완전히 수렴한 상태는 아닐 수 있다.

특히 기존 B 실험의 대표 결과였던 `B2_lr5e-4 = 5.8555`, `B1_bs2 = 5.7060`보다 낮았기 때문에, 이 조합은 단순 추측이 아니라 실제 성능 개선 후보로 볼 수 있다.

### 3000 실행 해석

`vocab_size=3000` 실행도 실패는 아니다. 오히려 중요한 검증을 통과했다.

- 로그에 `loaded tokenizer: ... nsmc_bpe_vocab3000_full.json`가 찍혀 기존 tokenizer 재사용이 확인됐다.
- epoch 1: `best val loss 6.4067`
- epoch 2: `best val loss 5.8388`
- epoch 3: `best val loss 5.7709`

즉 3000에서도 epoch가 진행될수록 loss는 계속 감소했다. 다시 말해:

- 조합의 방향성은 3000에서도 유지된다.
- 다만 현재 3 epoch 기준에서는 2000보다 절대 수치가 높다.

이 결과는 "예측 조합이 2000 전용 꼼수는 아니었다"는 점에서는 긍정적이다. 하지만 같은 3 epoch 기준 성능만 보면 현재는 2000 쪽이 더 강하다.

### 2000과 3000 비교 결론

현재 데이터만 보면 결론은 명확하다.

1. 경향 검증 목적에서는 `2000` 결과가 더 강하다.
2. 제출 기준 정합성과 tokenizer 재사용 검증 측면에서는 `3000`도 충분히 의미 있다.
3. 하지만 지금 즉시 다음 실험 축을 하나만 고르라면, `2000` 결과를 더 연장 검증하는 쪽이 우선이다.

즉 지금 시점의 우선순위는:

1. `2000` 조합을 `4 epoch`까지 연장
2. 그 다음 `2000`에서 `learning_rate=1e-3` 확인
3. 이후 `3000`은 제출용 승격 검증으로 별도 유지

### 지금 기준 최종 판단

현재 실행 결과를 반영하면, 이 문서의 추천은 아래처럼 수정된다.

- 가장 강한 실험 결과: `vocab_size=2000`, `best val loss=5.2275`
- 가장 안전한 제출 연계 검증: `vocab_size=3000`, `best val loss=5.7709`, 기존 tokenizer 재사용 확인

따라서 "무엇을 다음 기준점으로 삼을 것인가"라는 질문에는 `2000`을, "무엇을 보고서 연결용 승격 검증으로 둘 것인가"라는 질문에는 `3000`을 답으로 두는 것이 가장 자연스럽다.
