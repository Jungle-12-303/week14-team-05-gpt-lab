#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${1:-main}"
TITLE="${2:-}"
HEAD_BRANCH="$(git branch --show-current)"
REPO="$(git remote get-url origin | sed -E 's#^git@github.com:##; s#^https://github.com/##; s#\.git$##')"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI(gh)가 설치되어 있지 않습니다." >&2
  echo "설치 후 다시 실행하세요: sudo apt-get install gh" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI 인증이 필요합니다." >&2
  echo "먼저 실행하세요: gh auth login" >&2
  exit 1
fi

if [[ -z "$TITLE" ]]; then
  TITLE="$(git log -1 --pretty=%s)"
fi

BODY_FILE="$(mktemp)"
trap 'rm -f "$BODY_FILE"' EXIT

cat >"$BODY_FILE" <<'MARKDOWN'
## 변경 내용
- [ ] 핵심 변경사항 1
- [ ] 핵심 변경사항 2
- [ ] 필요 시 핵심 변경사항 3

## 확인 사항
- [ ] 로컬 테스트 또는 수동 확인 완료
- [ ] 영향 범위 확인 완료
- [ ] 배포 전 추가 확인이 필요한 내용 없음

## 리뷰 포인트
- 리뷰어가 집중해서 보면 좋은 부분
- 설계상 선택 이유 또는 트레이드오프
MARKDOWN

git push -u origin "$HEAD_BRANCH"
gh pr create \
  --repo "$REPO" \
  --base "$BASE_BRANCH" \
  --head "$HEAD_BRANCH" \
  --title "$TITLE" \
  --body-file "$BODY_FILE"
