# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

import json
from pathlib import Path
from collections import defaultdict

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        self.id_to_token = {}
        self.token_to_id = {}

        # 1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다
        for token, token_id in SPECIAL_IDS.items():
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        # 2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        for index in range(NUM_BYTES):
            self.id_to_token[index + BYTE_OFFSET] = bytes([index])
            self.token_to_id[bytes([index])] = index + BYTE_OFFSET

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self._init_special_tokens()
        self.merges = []

        ids = [b + BYTE_OFFSET for b in corpus.encode("utf-8")]
        while len(self.id_to_token) < self.vocab_size:
            pair_cnt = defaultdict(int)

            for i in range(len(ids) - 1):
                key = (ids[i], ids[i + 1])
                pair_cnt[key] += 1

            if not pair_cnt:
                break
            max_pair = max(pair_cnt, key=lambda pair: pair_cnt[pair])
            new_id = len(self.id_to_token)

            self.merges.append(max_pair)
            self.id_to_token[new_id] = max_pair
            self.token_to_id[max_pair] = new_id

            ids = self._replace_pair(ids, max_pair, new_id)

    def save(self, path: str | Path):
        data = {}
        data["vocab_size"] = self.vocab_size
        data["id_to_token"] = []
        data["merges"] = []

        for token_id, token in self.id_to_token.items():
            if isinstance(token, bytes):
                data["id_to_token"].append({
                    "id": token_id,
                    "type": "bytes",
                    "value": list(token),
                })
            elif isinstance(token, tuple):
                data["id_to_token"].append({
                    "id": token_id,
                    "type": "tuple",
                    "value": list(token),
                })
            else:
                data["id_to_token"].append({
                    "id": token_id,
                    "type": "str",
                    "value": token,
                })

        for pair in self.merges:
            data["merges"].append(list(pair))

        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str | Path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        for d in data["id_to_token"]:
            token_id = d["id"]

            if d["type"] == "bytes":
                token = bytes(d["value"])
            elif d["type"] == "tuple":
                token = tuple(d["value"])
            else:
                token = d["value"]

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        for pair in data["merges"]:
            self.merges.append(tuple(pair))

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        # UTF-8 byte ID 리스트
        token_id_list = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]

        for pair in self.merges:
            new_id = self.token_to_id[pair]
            token_id_list = self._replace_pair(token_id_list, pair, new_id)

        if add_bos_eos:
            token_id_list = [self.get_bos_id()
                            ] + token_id_list + [self.get_eos_id()]

        return token_id_list

    # token_id_list에서 지정한 pair가 연속으로 등장하면 하나의 new_id로 치환한다.
    # 예: token_id_list = [69, 70, 69], pair = (69, 70), new_id = 260
    # 결과: [260, 69]
    def _replace_pair(self, token_id_list: list[int], pair: tuple[int, int],
                      new_id: int) -> list[int]:
        temp_list = []
        i = 0
        while i < len(token_id_list):
            if i < len(token_id_list) - 1 and (token_id_list[i],
                                               token_id_list[i + 1]) == pair:
                temp_list.append(new_id)
                i += 2
            else:
                temp_list.append(token_id_list[i])
                i += 1
        return temp_list

    # token_id가 가리키는 토큰을 최종 byte 값 리스트로 변환한다.
    # byte 토큰이면 그대로 byte 값을 반환하고,
    # merge 토큰(tuple)이면 내부 토큰들을 재귀적으로 끝까지 풀어서 byte 값으로 변환한다.
    # 예: 260 -> (69, 70) -> [65, 66]
    def _token_id_to_bytes(self, token_id: int) -> list[int]:
        token = self.id_to_token[token_id]

        if isinstance(token, bytes):
            return list(token)

        if isinstance(token, tuple):
            byte_values = []

            for child_id in token:
                byte_values.extend(self._token_id_to_bytes(child_id))

            return byte_values

        if isinstance(token, str):
            return list(token.encode("utf-8"))

        raise ValueError(f"지원하지 않는 토큰 타입입니다: {type(token)}")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        byte_values = []

        for token_id in ids:
            if skip_special and token_id in SPECIAL_IDS.values():
                continue

            byte_values.extend(self._token_id_to_bytes(token_id))

        return bytes(byte_values).decode("utf-8")
