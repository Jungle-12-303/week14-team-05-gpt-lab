# mini GPT 과제 정리

## 1. 과제 목표

- PyTorch만 사용해서 교육용 `mini GPT`를 직접 구현한다.
- 거대한 LLM을 만드는 것이 아니라, GPT 계열 언어 모델의 핵심 구성 요소를 이해하는 것이 목적이다.
- 제출 기준은 동작하는 소스코드, 실행 가능한 노트북, `REPORT.md` 작성까지 포함한다.

## 2. 개발 및 협업 규칙

### Git 규칙

- 팀 저장소에서 작업한다.
- 개인 작업은 각자 브랜치에서 진행한다.
- `main` 또는 `master`에 직접 push하지 않는다.
- PR 리뷰 후 병합한다.
- 병합 전 관련 테스트와 전체 테스트를 통과시킨다.
- 데이터 파일, checkpoint, token, 비밀번호는 commit하지 않는다.

### 개발 환경

- Python `3.11`
- 허용 라이브러리:
  - `torch`
  - `torch.nn`
  - `torch.utils.data`
  - `numpy`
  - `matplotlib`
  - `pytest`
- 금지 항목:
  - Hugging Face `transformers`, `datasets`, `tokenizers`
  - `sentencepiece`
  - `spacy`
  - `nltk`
  - `lightning`
  - `accelerate`
  - 외부 pretrained model
  - 외부 tokenizer vocabulary
  - `tiktoken`

## 3. 프로젝트 구조

```text
gpt-lab/
├── README.md
├── REPORT.md
├── requirements.txt
├── download_data.py
├── gpt-lab.ipynb
├── data/
├── src/
│   ├── __init__.py
│   ├── bpe.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── model.py
│   ├── train.py
│   └── finetune.py
└── tests/
    ├── test_bpe.py
    ├── test_dataset.py
    ├── test_attention.py
    ├── test_model.py
    ├── test_train.py
    └── test_finetune.py
```

### 파일 역할

- `download_data.py`: NSMC 원본 다운로드 및 과제용 데이터 생성
- `gpt-lab.ipynb`: Colab/로컬 실행 순서 안내
- `src/bpe.py`: UTF-8 byte-level BPE tokenizer
- `src/dataset.py`: GPT pretraining용 dataset/dataloader
- `src/embeddings.py`: token embedding + position embedding
- `src/attention.py`: causal multi-head self-attention
- `src/model.py`: GPT 구성 요소와 모델 본체
- `src/train.py`: pretraining, loss 계산, checkpoint, generation
- `src/finetune.py`: NSMC 감성 분류 fine-tuning

## 4. 데이터

### 기본 데이터셋

- NAVER Sentiment Movie Corpus(NSMC)
- 원본 저장소: <https://github.com/e9t/nsmc>
- 라이선스: `CC0 1.0`

### 데이터 준비

```bash
python download_data.py
```

### 생성 파일

- `data/nsmc_lm_train.txt`
- `data/nsmc_lm_val.txt`
- `data/nsmc_sentiment_train.jsonl`
- `data/nsmc_sentiment_val.jsonl`
- `data/nsmc_sentiment_test.jsonl`

### 권장 데이터 규모

| 단계 | 권장 설정 | 목적 |
| --- | --- | --- |
| Smoke | `corpus[:5000]`, `vocab_size=300`, `context_length=32` | 동작 확인 |
| Light | `corpus[:500000]`, `vocab_size=2000`, `context_length=64` | 빠른 실험 |
| Basic | `corpus[:1500000]`, `vocab_size=3000`, `context_length=128` | 기본 제출 |

## 5. 구현 원칙

- 큰 기능을 한 번에 만들지 말고 TODO 단위로 구현한다.
- 작은 데이터로 먼저 실행해서 동작을 확인한다.
- 현재 단계 테스트를 먼저 통과시킨 뒤 다음 단계로 넘어간다.
- 마지막에 전체 테스트 `pytest tests/ -v`를 실행한다.
- Colab 런타임 종료에 대비해 오래 걸리는 학습 결과와 checkpoint는 저장한다.

## 6. 권장 구현 순서

| 순서 | 구현 대상 | 파일 | 테스트 |
| --- | --- | --- | --- |
| 1 | BPE tokenizer | `src/bpe.py` | `pytest tests/test_bpe.py -v` |
| 2 | Dataset / InputEmbedding | `src/dataset.py`, `src/embeddings.py` | `pytest tests/test_dataset.py -v` |
| 3 | MultiHeadAttention | `src/attention.py` | `pytest tests/test_attention.py -v` |
| 4 | GPT model | `src/model.py` | `pytest tests/test_model.py -v` |
| 5 | Pretraining utility | `src/train.py` | `pytest tests/test_train.py -v` |
| 6 | Sentiment fine-tuning | `src/finetune.py` | `pytest tests/test_finetune.py -v` |
| 7 | 전체 테스트 | 전체 | `pytest tests/ -v` |

## 7. 단계별 구현 내용

### 7.1 BPE tokenizer

구현 파일: `src/bpe.py`

구현 대상:

- `BPETokenizer._init_special_tokens`
- `BPETokenizer.train`
- `BPETokenizer.save`
- `BPETokenizer.load`
- `BPETokenizer.encode`
- `BPETokenizer.decode`

핵심 규칙:

- 특수 토큰 ID:
  - `0`: `<pad>`
  - `1`: `<unk>`
  - `2`: `<bos>`
  - `3`: `<eos>`
- `4~259`는 byte `0~255`
- `260` 이상은 merge token
- 입력 텍스트는 `text.encode("utf-8")`로 byte 목록으로 변환한다.
- 가장 자주 등장하는 pair를 반복적으로 merge한다.
- `decode()`에서는 merge token을 최종 byte 배열로 모두 펼친 뒤 마지막에 한 번만 UTF-8 decode 해야 한다.

한글 관련 주의:

- 한글 한 글자는 UTF-8에서 보통 3 byte다.
- byte-level BPE를 사용하면 한국어, 영어, 숫자, 문장부호를 동일한 방식으로 처리할 수 있다.
- byte를 중간에 문자로 바꾸면 한글이 깨질 수 있다.

### 7.2 GPTDataset / InputEmbedding

구현 파일:

- `src/dataset.py`
- `src/embeddings.py`

구현 대상:

- `GPTDataset.__len__`
- `GPTDataset.__getitem__`
- `create_dataloader`
- `InputEmbedding.__init__`
- `InputEmbedding.forward`

핵심 개념:

```text
token_ids = [10, 11, 12, 13]
context_length = 3

input  = [10, 11, 12]
target = [11, 12, 13]
```

- target은 input보다 한 칸 뒤여야 한다.
- 샘플 1개를 만들려면 `context_length + 1`개 token이 필요하다.
- `stride`는 다음 샘플 시작 위치 이동 칸 수다.
- embedding 출력 shape는 `(batch_size, seq_len, emb_dim)`이다.

### 7.3 MultiHeadAttention

구현 파일: `src/attention.py`

구현 대상:

- `MultiHeadAttention.__init__`
- `MultiHeadAttention.forward`

핵심 흐름:

1. 입력에서 `Q`, `K`, `V`를 만든다.
2. `n_heads` 기준으로 차원을 분리한다.
3. `Q @ K.T / sqrt(head_dim)`으로 attention score를 계산한다.
4. causal mask로 미래 token을 가린다.
5. softmax로 attention weight를 만든다.
6. weight와 `V`를 곱한다.
7. head를 다시 합치고 output projection을 적용한다.

주요 조건:

- `d_model % n_heads == 0`
- attention weight shape와 causal mask 동작이 테스트 대상이다.

### 7.4 GPT 모델

구현 파일: `src/model.py`

구현 대상:

- `LayerNorm`
- `GELU`
- `FeedForward`
- `TransformerBlock`
- `GPTModel`
- `generate_text_simple`

모델 구조:

```text
token IDs
-> InputEmbedding
-> TransformerBlock x n_layers
-> LayerNorm
-> Linear lm_head
-> vocab logits
```

주의 사항:

- `FeedForward`는 보통 `d_model -> 4*d_model -> d_model`
- residual connection을 정확히 넣어야 한다.
- `targets`가 없으면 logits만, 있으면 `(loss, logits)`을 반환한다.
- loss 계산 시 logits와 targets를 펼쳐서 `F.cross_entropy`에 넣는다.

### 7.5 사전 학습

구현 파일: `src/train.py`

구현 대상:

- `calc_loss_batch`
- `calc_loss_loader`
- `save_checkpoint`
- `load_checkpoint`
- `generate`
- `generate_and_print_sample`
- `train_model`

주의 사항:

- batch는 반드시 `device`로 이동한다.
- 평가 시 `model.eval()`과 `torch.no_grad()`를 사용한다.
- 평가 후 `model.train()` 상태로 복귀한다.
- checkpoint에는 model/optimizer state, epoch, global step을 함께 저장한다.
- `temperature=0`이면 greedy, `temperature>0`이면 sampling이다.
- `top_k`, `eos_id` 처리 가능해야 한다.

### 7.6 감성 분류 fine-tuning

구현 파일: `src/finetune.py`

구현 대상:

- `make_sentiment_dataset`
- `ReviewSentimentDataset`
- `GPTForSequenceClassification`
- `train_epoch_sentiment`
- `evaluate_sentiment`

핵심 개념:

- LM head는 다음 token 예측용이다.
- 감성 분류는 `classifier = nn.Linear(emb_dim, 2)` 형태의 별도 classification head가 필요하다.
- padding이 아닌 마지막 유효 token의 hidden state를 문장 대표 벡터로 사용한다.
- `pad_id` 기본값은 `0`

## 8. 실험 및 보고서에 남길 내용

### BPE 관련 기록

- 사용한 corpus 크기
- `vocab_size`
- 학습 시간
- Colab/로컬 환경
- vocabulary 저장 경로

### REPORT.md 필수 항목

| 섹션 | 포함 내용 |
| --- | --- |
| 0 | 반, 팀명, 팀원 이름 |
| 1 | 구현한 TODO 목록, 담당자 |
| 2 | 실행한 pytest 명령과 결과 |
| 3 | NSMC 사용량, train/val/test 파일 |
| 4 | `vocab_size`, corpus 크기, 학습 시간, 저장 경로 |
| 5 | `context_length`, `emb_dim`, `n_heads`, `n_layers`, 파라미터 수 |
| 6 | pretraining loss, checkpoint, 샘플 생성 결과 |
| 7 | validation/test accuracy, loss |
| 8 | colab/local, cpu/gpu |
| 9 | 어려웠던 점, 개선 시도, 한계 |

## 9. 추가 미션

### 9.1 사전 학습 성능 향상

- learning rate warmup
- cosine decay
- gradient clipping
- weight decay 실험

### 9.2 하이퍼파라미터 탐색

- `batch_size`: 2, 4, 8, 16
- `drop_rate`: 0.0, 0.1, 0.2
- `learning_rate`: `1e-4`, `3e-4`, `5e-4`
- `context_length`: 64, 128
- `n_layers`: 1, 2, 4
- `emb_dim`: 64, 128, 192

### 9.3 감성 분류 개선

- backbone 일부 freeze
- classifier/backbone learning rate 분리
- class imbalance 확인
- validation loss가 가장 낮은 checkpoint 선택

## 10. AI 활용 시 주의

- 전체 코드를 한 번에 요청하지 말고, 실패한 테스트 1개와 현재 함수 1개 기준으로 묻는 편이 좋다.
- AI 답을 붙이기 전에 아래 항목을 반드시 확인한다:
  - 입력 shape
  - 출력 shape
  - device 이동 여부
  - train/eval 모드
  - 실제 테스트 통과 여부

## 11. 최종 제출물

- 동작하는 `src/` 소스 코드
- 실행 가능한 `gpt-lab.ipynb`
- `REPORT.md`
