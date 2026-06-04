# NSMC 리뷰 길이 분포 분석

이 문서는 `REPORT.md`와 발표 질의응답에서 `max_length=128`, `context_length=64/128`, truncation 영향을 설명하기 위한 보조 분석 자료다.

## 분석 기준

- 원본 데이터: `data/ratings_train.txt`, `data/ratings_test.txt`
- Fine-tuning 데이터: `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl`
- Tokenizer: `artifacts/tokenizers/nsmc_bpe_vocab3000_full.json`
- Token 길이 계산: 현재 fine-tuning Dataset과 동일하게 BOS/EOS를 추가하지 않은 `encode(text)` 기준
- 별도 영구 분석 도구는 만들지 않았고, 일회성 Python 명령으로 계산했다.

## 원본 TSV 정제 결과

원본 NSMC는 JSON이 아니라 TSV 형식이며, 컬럼은 `id`, `document`, `label`이다. 빈 리뷰와 잘못된 label을 제거한 뒤 fine-tuning용 `text`, `label` 구조로 변환한다.

| 원본 파일 | raw rows | 사용 rows | label 0 | label 1 | positive ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ratings_train.txt` | 150,000 | 149,995 | 75,170 | 74,825 | 0.4988 |
| `ratings_test.txt` | 50,000 | 49,997 | 24,826 | 25,171 | 0.5035 |

`ratings_train.txt`의 사용 가능 row 149,995개는 seed 고정 shuffle 뒤 train/validation으로 나뉘었다. `ratings_test.txt`는 최종 test set으로 유지했다.

## Fine-tuning Split

| split | rows | label 0 | label 1 | positive ratio |
| --- | ---: | ---: | ---: | ---: |
| train | 137,996 | 69,076 | 68,920 | 0.4994 |
| validation | 11,999 | 6,094 | 5,905 | 0.4921 |
| test | 49,997 | 24,826 | 25,171 | 0.5035 |

세 split 모두 label 0/1 비율이 거의 50:50이다. 따라서 D 실험에서 class weight나 sampling 기반 imbalance 보정은 적용하지 않았다.

## 문자 길이 분포

문자 길이는 Python `len(text)` 기준이다.

| split | mean | median | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 35.19 | 27 | 75 | 107 | 139 | 146 |
| validation | 35.40 | 27 | 76 | 108 | 139 | 141 |
| test | 35.32 | 27 | 76 | 108 | 139 | 144 |

NSMC 리뷰는 대부분 짧다. 문자 기준 중앙값은 27자이고, 95% 지점도 약 107~108자 수준이다.

## BPE Token 길이 분포

Token 길이는 `vocab_size=3000` 공유 BPE tokenizer 기준이다.

| split | mean | median | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 20.86 | 16 | 44 | 63 | 86 | 413 |
| validation | 21.02 | 15 | 45 | 64 | 88 | 135 |
| test | 20.94 | 16 | 44 | 63 | 87 | 389 |

Token 기준으로도 대부분의 리뷰는 짧다. 중앙값은 15~16 token, p95는 63~64 token, p99는 86~88 token 수준이다.

일부 max 값이 큰 이유는 반복 특수문자, 비한국어 문자열, BPE merge가 잘 적용되지 않는 긴 byte sequence 같은 outlier가 있기 때문이다. 문자 수는 140자 안팎이어도 byte-level BPE token 수는 더 커질 수 있다.

## max_length Coverage

### `max_length=64`

| split | <=64 tokens | >64 tokens | coverage | truncation ratio |
| --- | ---: | ---: | ---: | ---: |
| train | 131,488 | 6,508 | 95.28% | 4.72% |
| validation | 11,415 | 584 | 95.13% | 4.87% |
| test | 47,633 | 2,364 | 95.27% | 4.73% |

`max_length=64`는 약 95%의 리뷰를 온전히 담는다. 다만 약 4.7~4.9%의 리뷰는 truncation된다.

### `max_length=128`

| split | <=128 tokens | >128 tokens | coverage | truncation ratio |
| --- | ---: | ---: | ---: | ---: |
| train | 137,983 | 13 | 99.9906% | 0.0094% |
| validation | 11,997 | 2 | 99.9833% | 0.0167% |
| test | 49,991 | 6 | 99.9880% | 0.0120% |

`max_length=128`은 거의 모든 리뷰를 커버한다. Train/validation/test 전체 199,992개 중 128 token을 초과하는 리뷰는 21개뿐이다.

| 전체 fine-tuning split | rows |
| --- | ---: |
| total rows | 199,992 |
| <=64 tokens | 190,536 |
| >64 tokens | 9,456 |
| <=128 tokens | 199,971 |
| >128 tokens | 21 |

## 해석

`max_length=128`은 D 감성 분류 실험에서 타당한 선택이다. Fine-tuning 전체 데이터의 99.99% 이상을 truncation 없이 처리할 수 있기 때문이다.

`max_length=64`도 약 95%의 리뷰를 커버하므로 빠른 screening에는 사용할 수 있다. 다만 약 4.7%의 리뷰가 잘리므로, 최종 감성 분류 평가에서는 128이 더 안전하다.

C 실험에서 `context_length=64`가 유망하게 나온 것은 pretraining screening 관점의 결과다. 그러나 감성 분류 fine-tuning에서는 64와 128의 의미가 다르다. 리뷰 길이 분포 기준으로 보면 128은 거의 모든 리뷰를 보존하고, 64는 일부 긴 리뷰를 잘라낸다.

## 질의응답용 요약

질문: 왜 D fine-tuning에서 `max_length=128`을 사용했나요?

답변: NSMC fine-tuning 데이터의 BPE token 길이 분포를 보면 128 token 이하가 전체의 99.99% 이상입니다. 따라서 `max_length=128`은 거의 모든 리뷰를 자르지 않고 처리할 수 있는 안전한 기준입니다.

질문: `context_length=64`도 충분한 것 아닌가요?

답변: 64 token도 약 95%의 리뷰를 커버하므로 screening에는 충분히 쓸 수 있습니다. 하지만 약 4.7%의 리뷰는 잘리기 때문에 최종 감성 분류 평가에서는 128이 더 안전합니다.

질문: 긴 리뷰 truncation이 결과에 큰 영향을 줬나요?

답변: `max_length=128` 기준으로는 전체 199,992개 중 21개만 128 token을 초과했습니다. 따라서 이번 D 실험에서 truncation 영향은 매우 제한적이었다고 볼 수 있습니다.
