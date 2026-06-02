# 영빈 하이퍼파라미터 개인 실행 런북

| 항목 | 내용 |
| --- | --- |
| 목적 | 혼자서 사전학습 하이퍼파라미터를 우선순위대로 비교하고, 가장 효율적으로 best 후보를 고른다. |
| 적용 범위 | pretraining 기준 하이퍼파라미터 탐색 |
| 관련 문서 | [`../EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md), [`../EXPERIMENT_B_YEONGBEEN.md`](../EXPERIMENT_B_YEONGBEEN.md), [`../EXPERIMENT_C_BEOMSANG.md`](../EXPERIMENT_C_BEOMSANG.md) |

## 1. 혼자 할 때 무엇부터 볼까

혼자 진행할 때도 가장 먼저 봐야 할 값은 `learning_rate`다.

이유:
- 같은 epoch 수에서 수렴 안정성 차이가 가장 빨리 드러난다.
- 잘못된 learning rate는 batch size, dropout, context length 비교까지 왜곡한다.
- 비교 비용 대비 얻는 정보량이 가장 크다.

그래서 혼자 할 때는 아래 순서로 간다.

1. `learning_rate`
2. `batch_size`
3. `drop_rate`
4. `context_length`

## 2. 혼자 진행할 때 권장 범위

4개 축을 다 보되, 한 번에 모든 조합을 돌리지 말고 축 하나씩 줄여 가는 방식이 좋다.

| 순서 | 실험 ID | 변수 | 후보 | 비고 |
| --- | --- | --- | --- | --- |
| 1 | B2 | learning_rate | `1e-4`, `3e-4`, `5e-4` | 가장 먼저 best lr 압축 |
| 2 | B1 | batch_size | `4`, `8`, `16` | 혼자 할 때는 `2`는 후순위 |
| 3 | B3 | drop_rate | `0.0`, `0.1`, `0.2` | best lr, best bs 고정 후 비교 |
| 4 | C1 | context_length | `64`, `128` | 정말 시간 남을 때만 수행 |

## 3. 추천 실행 순서

### 3.1 Step 0: smoke test

목표:
- 코드가 끝까지 도는지
- 토크나이저 재사용이 되는지
- 체크포인트가 저장되는지

설정:
- `USE_SMOKE = True`
- `EXPERIMENT_ID = "A0_smoke"`

### 3.2 Step 1: learning_rate 비교

먼저 아래 3개만 순서대로 돌린다.

| EXPERIMENT_ID | TRAIN_OVERRIDES |
| --- | --- |
| `B2_lr1e-4` | `{"learning_rate": 1e-4}` |
| `B2_lr3e-4` | `{"learning_rate": 3e-4}` |
| `B2_lr5e-4` | `{"learning_rate": 5e-4}` |

판단 기준:
- `best val loss`
- loss 발산 여부
- epoch당 체감 속도

이 단계에서 목표는 `best_lr` 하나를 고르는 것이다.

### 3.3 Step 2: batch_size 비교

`best_lr`를 고른 뒤 아래를 비교한다.

| EXPERIMENT_ID | TRAIN_OVERRIDES |
| --- | --- |
| `B1_bs4` | `{"batch_size": 4, "learning_rate": best_lr}` |
| `B1_bs8` | `{"batch_size": 8, "learning_rate": best_lr}` |
| `B1_bs16` | `{"batch_size": 16, "learning_rate": best_lr}` |

판단 기준:
- `best val loss`
- OOM 여부
- 학습 시간

이 단계에서 목표는 `best_bs` 하나를 고르는 것이다.

### 3.4 Step 3: drop_rate 비교

`best_lr`, `best_bs`를 고정하고 아래를 비교한다.

| EXPERIMENT_ID | BASE_OVERRIDES | TRAIN_OVERRIDES |
| --- | --- | --- |
| `B3_drop0.0` | `{"drop_rate": 0.0}` | `{"learning_rate": best_lr, "batch_size": best_bs}` |
| `B3_drop0.1` | `{"drop_rate": 0.1}` | `{"learning_rate": best_lr, "batch_size": best_bs}` |
| `B3_drop0.2` | `{"drop_rate": 0.2}` | `{"learning_rate": best_lr, "batch_size": best_bs}` |

이 단계에서 목표는 `best_drop` 하나를 고르는 것이다.

### 3.5 Step 4: context_length 비교

시간이 남으면 마지막에만 비교한다.

| EXPERIMENT_ID | BASE_OVERRIDES | TRAIN_OVERRIDES |
| --- | --- | --- |
| `C1_ctx64` | `{"context_length": 64}` | `{"learning_rate": best_lr, "batch_size": best_bs}` |
| `C1_ctx128` | `{"context_length": 128}` | `{"learning_rate": best_lr, "batch_size": best_bs}` |

이 단계는 필수가 아니다.

## 4. 중단 기준

아래 경우는 바로 중단하거나 후순위로 내린다.

| 상황 | 처리 |
| --- | --- |
| loss가 NaN 또는 급격히 발산 | 즉시 중단 |
| GPU 메모리 부족 | 해당 batch size 또는 context length 제외 |
| val loss 차이가 거의 없고 더 느림 | 느린 후보 제외 |
| drop_rate 차이가 미미 | `0.1` 유지 |

## 5. 공통 기준 설정

아래 기본값을 유지하고, 실험 변수만 바꾼다.

```python
from src.experiment_config import BASE_CONFIG, TRAIN_CONFIG, SMOKE_CONFIG
```

실제 값:

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

SMOKE_CONFIG = {
    **TRAIN_CONFIG,
    "batch_size": 2,
    "num_epochs": 1,
    "eval_freq": 20,
    "eval_iter": 2,
}
```

## 6. Colab 실행 셀 예시

### 6.1 공통 import / 설정 셀

```python
from copy import deepcopy
import random

import numpy as np
import torch

from src.dataset import create_dataloader
from src.experiment_config import (
    BASE_CONFIG,
    DEFAULT_TOKENIZER_NAME,
    TRAIN_CONFIG,
    SMOKE_CONFIG,
)
from src.experiment_utils import (
    load_or_encode_pretrain_corpus,
    load_or_train_tokenizer,
    load_pretrain_texts,
)
from src.model import GPTModel
from src.train import train_model


def merge_config(base, overrides=None):
    config = deepcopy(base)
    if overrides:
        config.update(overrides)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


USE_SMOKE = True
TOKENIZER_NAME = DEFAULT_TOKENIZER_NAME
EXPERIMENT_ID = "B2_lr3e-4"
RUN_DATE = "20260602"

BASE_OVERRIDES = {}
TRAIN_OVERRIDES = {
    "learning_rate": 3e-4,
}

base_config = merge_config(BASE_CONFIG, BASE_OVERRIDES)
train_config = merge_config(
    SMOKE_CONFIG if USE_SMOKE else TRAIN_CONFIG,
    TRAIN_OVERRIDES,
)

set_seed(train_config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("base_config:", base_config)
print("train_config:", train_config)
```

### 6.2 토크나이저 재사용 셀

```python
train_text, val_text = load_pretrain_texts(
    "data/nsmc_lm_train.txt",
    "data/nsmc_lm_val.txt",
)

tokenizer, tokenizer_path, was_trained = load_or_train_tokenizer(
    train_text=train_text,
    vocab_size=base_config["vocab_size"],
    tokenizer_name=TOKENIZER_NAME,
    output_dir="tokenizers",
)

print("tokenizer_path:", tokenizer_path)
print("trained_now:", was_trained)
print("vocab_size:", tokenizer.vocab_size)
print("num_merges:", len(tokenizer.merges))

(
    train_token_ids,
    val_token_ids,
    train_ids_path,
    val_ids_path,
    ids_were_encoded,
) = load_or_encode_pretrain_corpus(
    tokenizer,
    train_text,
    val_text,
    vocab_size=base_config["vocab_size"],
    tokenizer_name=TOKENIZER_NAME,
    output_dir="tokenizers",
)

print("train_ids_path:", train_ids_path)
print("val_ids_path:", val_ids_path)
print("token_ids_encoded_now:", ids_were_encoded)
```

### 6.3 dataloader / model / optimizer 셀

```python
train_loader = create_dataloader(
    train_token_ids,
    context_length=base_config["context_length"],
    batch_size=train_config["batch_size"],
    stride=base_config["context_length"],
    shuffle=True,
)

val_loader = create_dataloader(
    val_token_ids,
    context_length=base_config["context_length"],
    batch_size=train_config["batch_size"],
    stride=base_config["context_length"],
    shuffle=False,
)

model = GPTModel(base_config)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config["learning_rate"],
    weight_decay=train_config["weight_decay"],
)

history = {}
```

### 6.4 학습 실행 셀

```python
train_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=train_config["num_epochs"],
    eval_freq=train_config["eval_freq"],
    eval_iter=train_config["eval_iter"],
    start_context=train_config["start_context"],
    tokenizer=tokenizer,
    ckpt_dir="checkpoints",
    experiment_id=EXPERIMENT_ID,
    run_date=RUN_DATE,
    history=history,
)

print("train_losses:", train_losses)
print("val_losses:", history.get("val_losses"))
print("best_val_loss:", history.get("best_val_loss"))
print("best_checkpoint_path:", history.get("best_checkpoint_path"))
```

## 7. 최종 목표

혼자 할 때의 최종 목표는 모든 조합을 다 도는 게 아니다.

목표:
- `best_lr` 1개
- `best_bs` 1개
- `best_drop` 1개
- 시간이 되면 `best_ctx` 1개

이 3~4개만 정리해도 발표와 보고서용 근거로 충분하다.
