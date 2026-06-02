# REPORT.md 추가 개선 계획

이 문서는 `REPORT.md` 원본 양식을 크게 바꾸지 않고, 과제 공지의 7.1 이후 추가 과제를 수행한 뒤 리포트를 어떻게 보완할지 정리한 문서다. 기본 제출 기준은 `REPORT.md`의 6.2 `Basic baseline 1회 실행 기준`에 먼저 기록하고, 아래 내용은 그 이후의 개선 방향으로 사용한다.

## 1. 기본 제출 기준과 추가 과제 구분

| 구분 | REPORT 반영 방식 | 우선순위 |
| --- | --- | --- |
| Basic baseline | `A0_basic` 실행 결과를 `REPORT.md` 6.2에 기록 | 필수 |
| 7.1 사전 학습 성능 향상 | 6.3 결과 또는 9. 고찰에 baseline 대비 개선 여부 기록 | 추가 |
| 7.2 하이퍼파라미터 탐색 | 9. 고찰에 주요 비교표와 선택 근거 기록 | 추가 |
| 7.3 감성 분류 개선 | 7. 미세 조정에 D4 최종 test 결과 기록 | 추가 |

Basic 기준은 `corpus[:1_500_000]`, `vocab_size=3000`, `context_length=128`이다. 현재 LM train text가 1,500,000자보다 짧으면 `corpus[:1_500_000]`은 전체 train corpus 사용과 같다. `A0`는 screening baseline이고, `A0_basic`은 `--vocab-size 3000 --train-char-limit 1500000`로 실행하는 Basic 제출 기준 baseline이다.

## 2. 7.1 사전 학습 성능 향상 기록

`A0_basic`을 먼저 확보한 뒤, 아래 항목은 Basic baseline 대비 비교로만 적는다.

| 실험 ID | 변경점 | best val loss | baseline 대비 | checkpoint | 결론 |
| --- | --- | ---: | --- | --- | --- |
| A1 | warmup + cosine decay |  |  |  |  |
| A2 | gradient clipping |  |  |  |  |
| A3 | weight decay |  |  |  |  |
| A4 | combined |  |  |  |  |

보고서에는 모든 raw metric을 붙이지 않는다. 최종 판단에 필요한 best validation loss, train loss 요약, checkpoint 경로, 실패 원인만 남긴다.

## 3. 7.2 하이퍼파라미터 탐색 기록

하이퍼파라미터 실험은 전체 grid를 다 돌렸는지보다, baseline 대비 어떤 설정을 선택했는지가 중요하다.

| 실험 ID | 변경 변수 | 후보 | 선택값 | 선택 근거 |
| --- | --- | --- | --- | --- |
| B1 | batch_size | 2, 4, 8, 16 |  |  |
| B2 | learning_rate | 1e-4, 3e-4, 5e-4 |  |  |
| B3 | drop_rate | 0.0, 0.1, 0.2 |  |  |
| C1 | context_length | 64, 128 |  |  |
| C2 | n_layers | 1, 2, 4 |  |  |
| C3 | emb_dim | 64, 128, 192 |  |  |

`context_length=64` 결과만 있는 경우에는 Light screening 결과임을 표시한다. 최종 성능 근거로 쓰려면 가능하면 Basic 기준의 `context_length=128` 재실행 결과를 함께 적는다.

## 4. 7.3 감성 분류 개선 기록

D0/D2/D3는 validation 후보 비교로만 기록하고, test set은 D4에서 선택된 checkpoint에 대해 1회만 평가한다.

| 실험 ID | 변경점 | best val loss | val acc | test 평가 | 결론 |
| --- | --- | ---: | ---: | --- | --- |
| D0 | baseline |  |  | 미평가 |  |
| D2 | freeze |  |  | 미평가 |  |
| D3 | lr 분리 |  |  | 미평가 |  |
| D4 | selected checkpoint test |  |  | 1회 평가 |  |

`REPORT.md` 7장에는 최종 D4의 test loss/test accuracy를 적고, D0/D2/D3 각각의 test 결과는 만들지 않는다.

## 5. 리포트 보완 체크리스트

- `A0_basic` 결과가 `REPORT.md` 6.2에 있는지 확인한다.
- BPE vocabulary 경로와 학습 corpus 크기를 적는다.
- best pretrain checkpoint와 best sentiment checkpoint의 Google Drive 경로를 적는다.
- raw metric JSONL, stdout/stderr log, `.pt` 파일은 Git에 올리지 않는다.
- 실패한 실험은 OOM, NaN, 시간 부족, validation loss 악화 중 하나로 원인을 짧게 적는다.
- 최종 고찰은 “baseline 대비 무엇을 바꿨고, 어떤 수치가 변했으며, 왜 그 설정을 선택했는가”로 마무리한다.
