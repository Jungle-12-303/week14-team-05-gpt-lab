# B2 Learning Rate 추가 탐색 실험

## 1. 실험 목적

기존 B2 learning_rate 실험에서는 `1e-4`, `3e-4`, `5e-4` 순서로 learning_rate를 높일수록 validation loss가 계속 낮아졌다. 따라서 `5e-4`보다 높은 구간에서 loss가 더 낮아지는지, 또는 발산/불안정 구간이 나타나는지 확인한다.

이 문서는 기존 `EXPERIMENT_B_YEONGBEEN.md`와 분리해 learning_rate 확장 실험 결과만 기록한다.

## 2. 비교 기준

| 항목 | 값 |
| --- | --- |
| 기준 실험 | B2 learning_rate 확장 |
| 주요 변경 변수 | learning_rate |
| 비교 후보 | `7e-4`, `1e-3`, 선택 시 `2e-3` |
| baseline | `B2_lr5e-4` |
| baseline best val loss | 5.855498008430004 |
| 판단 기준 | best validation loss, 학습 안정성, 생성 샘플 품질 |

## 3. 고정 설정

learning_rate의 영향만 비교하기 위해 기존 B2 조건을 유지한다.

| 항목 | 값 |
| --- | --- |
| vocab_size | 2000 |
| tokenizer | `artifacts/tokenizers/nsmc_bpe_vocab2000_full.json` |
| train_char_limit / val_char_limit | 500000 / 50000 |
| batch_size | 8 |
| drop_rate | 0.1 |
| context_length | 64 |
| n_layers / emb_dim / n_heads | 2 / 128 / 4 |
| num_epochs | 2 |
| seed | 42 |

## 4. 공식 비교 결과

| experiment_id | learning_rate | best global_step | best val loss | final train loss | elapsed_min | best checkpoint | 결론 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `B2_lr5e-4` | 0.0005 | 1280 | 5.855498008430004 | 6.2834 | 4.039404197533925 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr5e-4_20260603_YEONGBEEN/checkpoints/B2_lr5e-4_20260603_step1280_best.pt` | 기존 B2 최고 성능 baseline |
| `B2_lr7e-4` | 0.0007 | 1280 | 5.613503843545914 | 5.9309 | 4.046044325828552 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr7e-4_20260603_YEONGBEEN/checkpoints/B2_lr7e-4_20260603_step1280_best.pt` | baseline보다 best val loss 개선 |
| `B2_lr1e-3` | 0.001 | 1280 | 5.468767359852791 | 5.6630 | 4.179062175750732 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr1e-3_20260603_YEONGBEEN/checkpoints/B2_lr1e-3_20260603_step1280_best.pt` | 7e-4보다 추가 개선 |
| `B2_lr2e-3` | 0.002 | 1280 | 5.341542527079582 | 5.4075 | 4.102034298578898 | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr2e-3_20260603_YEONGBEEN/checkpoints/B2_lr2e-3_20260603_step1280_best.pt` | 1e-3보다 추가 개선, 현재 확장 실험 중 best |

## 5. 결과 요약

기존 B2 실험에서 가장 좋았던 `5e-4`보다 높은 learning_rate를 추가로 확인한 결과, `7e-4`, `1e-3`, `2e-3` 모두 baseline보다 낮은 validation loss를 기록했다. 특히 `2e-3`은 best validation loss 5.341542527079582로 이번 확장 실험에서 가장 낮은 값을 보였다.

| 비교 | best val loss 차이 | 해석 |
| --- | ---: | --- |
| `7e-4` vs `5e-4` | -0.241994164884090 | baseline보다 개선 |
| `1e-3` vs `7e-4` | -0.144736483693123 | 7e-4보다 추가 개선 |
| `2e-3` vs `1e-3` | -0.127224832773209 | 1e-3보다 추가 개선 |

## 6. 결과 분석

### 6.1 현재 관찰

`B2_lr7e-4` 실행 결과 best validation loss는 5.613503843545914로, baseline인 `B2_lr5e-4`의 5.855498008430004보다 낮아졌다. 이어서 `B2_lr1e-3`은 best validation loss 5.468767359852791을 기록했고, `B2_lr2e-3`은 5.341542527079582까지 낮아졌다.

`5e-4 -> 7e-4 -> 1e-3 -> 2e-3` 순서로 learning_rate를 높였을 때 best validation loss는 5.855498008430004, 5.613503843545914, 5.468767359852791, 5.341542527079582 순서로 계속 감소했다. 네 실험 모두 `final_global_step=1280`으로 동일하므로, 이번 확장 실험에서도 learning_rate를 높인 효과가 유지되었다고 볼 수 있다.

학습 중 loss가 NaN으로 튀거나 급격히 불안정해지는 징후는 보이지 않았다. `B2_lr2e-3`은 epoch 1의 val loss 5.6024에서 epoch 2의 val loss 5.3415까지 감소했고, train loss도 6.3893에서 5.4075로 낮아졌다. 따라서 현재까지는 `2e-3`이 확장 실험의 best 후보이다.

현재 탐색 범위에서는 learning_rate를 높일수록 loss가 계속 낮아졌지만, GPU 사용 제한으로 추가 실험은 진행하지 않았다. 따라서 이번 추가 레포트에서는 `2e-3`을 현재 확인된 범위 안의 best 후보로 정리한다.

### 6.2 판단 기준

추가 실험의 best validation loss가 5.855498008430004보다 낮고, loss가 NaN이 되거나 급격히 튀지 않으면 기존 후보인 `5e-4`보다 우선 후보로 본다.

반대로 validation loss가 증가하거나, train loss는 낮아지는데 val loss가 크게 악화되거나, 생성 샘플이 심하게 깨지면 learning_rate가 너무 큰 것으로 판단한다. 이 경우 기존 후보인 `5e-4`를 유지한다.

## 7. 세부 기록

### 7.1 B2_lr7e-4

| 항목 | 값 |
| --- | --- |
| 실행 규모 | Light screening |
| experiment_id | `B2_lr7e-4` |
| 변경 변수 | learning_rate |
| 변경값 | 0.0007 |
| vocab_size | 2000 |
| batch_size | 8 |
| drop_rate | 0.1 |
| num_epochs | 2 |
| final_global_step | 1280 |
| elapsed_min | 4.046044325828552 |
| best checkpoint | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr7e-4_20260603_YEONGBEEN/checkpoints/B2_lr7e-4_20260603_step1280_best.pt` |
| stdout/stderr log | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr7e-4_20260603_YEONGBEEN/logs/B2_lr7e-4_20260603.out` |

Epoch별 결과:

| epoch | train loss | val loss | best val loss |
| ---: | ---: | ---: | ---: |
| 1 | 6.8723 | 6.3611 | 6.3611 |
| 2 | 5.9309 | 5.6135 | 5.613503843545914 |

생성 샘플:

```text
영화.
평점이 잘 보는데.
이 나구., 연기가 돋은영화로, 취향에 10점..정말 이시의 배우들의 자니다. 시대박..감동을 가니
그냥..평점이 너무 하는 영화
```

### 7.2 B2_lr1e-3

| 항목 | 값 |
| --- | --- |
| 실행 규모 | Light screening |
| experiment_id | `B2_lr1e-3` |
| 변경 변수 | learning_rate |
| 변경값 | 0.001 |
| vocab_size | 2000 |
| batch_size | 8 |
| drop_rate | 0.1 |
| num_epochs | 2 |
| final_global_step | 1280 |
| elapsed_min | 4.179062175750732 |
| best checkpoint | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr1e-3_20260603_YEONGBEEN/checkpoints/B2_lr1e-3_20260603_step1280_best.pt` |
| stdout/stderr log | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr1e-3_20260603_YEONGBEEN/logs/B2_lr1e-3_20260603.out` |

Epoch별 결과:

| epoch | train loss | val loss | best val loss |
| ---: | ---: | ---: | ---: |
| 1 | 6.7152 | 5.9710 | 5.9710 |
| 2 | 5.6630 | 5.4688 | 5.468767359852791 |

생성 샘플:

```text
영화중간는 소재를.
재미없는데..진짜...이 영화입니다
너무 재미있게봤어요......
정말 왜..정말 이름의 배우들의 자식의 매력을텐데.....정말 재밌어서 나는 개연했다.
```

### 7.3 B2_lr2e-3

`B2_lr1e-3`에서도 개선되고 발산 징후가 없어 추가로 진행한 선택 실험이다.

| 항목 | 값 |
| --- | --- |
| 실행 규모 | Light screening |
| experiment_id | `B2_lr2e-3` |
| 변경 변수 | learning_rate |
| 변경값 | 0.002 |
| vocab_size | 2000 |
| batch_size | 8 |
| drop_rate | 0.1 |
| num_epochs | 2 |
| final_global_step | 1280 |
| elapsed_min | 4.102034298578898 |
| best checkpoint | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr2e-3_20260603_YEONGBEEN/checkpoints/B2_lr2e-3_20260603_step1280_best.pt` |
| stdout/stderr log | `/content/drive/MyDrive/gpt-lab/experiment_outputs/pretrain/B2_lr2e-3_20260603_YEONGBEEN/logs/B2_lr2e-3_20260603.out` |

Epoch별 결과:

| epoch | train loss | val loss | best val loss |
| ---: | ---: | ---: | ---: |
| 1 | 6.3893 | 5.6024 | 5.6024 |
| 2 | 5.4075 | 5.3415 | 5.341542527079582 |

생성 샘플:

```text
영화의 모습을 잘 보는데.
별로
우., 연기가 돋보며..이 많았다는 영화로 바라
시의 배우들의 자를 보는내내 오늘은 영화보고싶다.
내가 좋았는데.. 이건 뭐한
```

## 8. 최종 결론

`B2_lr2e-3`은 baseline `B2_lr5e-4`와 추가 실험 `B2_lr7e-4`, `B2_lr1e-3`보다 best validation loss가 낮아졌고, 학습 불안정 징후도 보이지 않았다. 현재까지의 learning_rate 후보는 `2e-3`으로 갱신한다.

이번 추가 실험은 `5e-4` 이후 learning_rate 구간에서 성능 개선이 이어지는지 확인하는 데 목적이 있었다. `7e-4`, `1e-3`, `2e-3` 모두 같은 `final_global_step=1280` 조건에서 baseline보다 낮은 validation loss를 보였으므로, 기존 B2 결론의 learning_rate 후보는 `5e-4`에서 `2e-3`으로 갱신하는 것이 타당하다.

다만 이번 실험은 2 epoch 기준의 Light screening이므로, `2e-3`이 전체 학습 과정에서 최종 최적값이라고 단정할 수는 없다. 현재 확보한 결과만 기준으로는 `2e-3`이 가장 좋은 후보이며, 보고서에는 “확인된 learning_rate 범위 내 best”로 표현한다.
