# Shared Tokenizers

이 디렉토리는 팀 공통 기준으로 사용하는 BPE tokenizer vocabulary만 저장한다.

Colab 실험 중 생성되는 임시 tokenizer 파일은 개인 Google Drive의
`experiment_outputs/.../tokenizers/`에 저장한다. 그중 팀 전체가 같은 기준으로
재사용하기로 확정한 작은 JSON 파일만 이 디렉토리에 commit한다.

## 파일명 규칙

```text
nsmc_bpe_vocab{vocab_size}_full.json
```

예시:

```text
nsmc_bpe_vocab3000_full.json
nsmc_bpe_vocab5000_full.json
```

## 운영 규칙

- 공식 실험용 공유 tokenizer는 전체 training corpus로 학습한다.
- 공유 tokenizer 파일명에는 `vocab_size`와 전체 corpus 기준임을 나타내는 `full`을 포함한다.
- 같은 파일명을 덮어쓰기보다 새 설정은 새 파일명으로 추가한다.
- 실험 로그에는 사용한 tokenizer 경로를 기록한다.
- checkpoint, raw metric JSONL, stdout/stderr log는 이 디렉토리에 저장하지 않는다.
