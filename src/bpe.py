# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

import json
from pathlib import Path

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
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

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

        for pair, new_id in self.merges:
            data["merges"].append({"pair": list(pair), "new_id": new_id})

        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        # UTF-8 byte ID 리스트
        token_id_list = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]

        for pair, new_id in self.merges:
            token_id_list = self._replace_pair(token_id_list, pair, new_id)

        if add_bos_eos:
            token_id_list = [self.get_bos_id()
                            ] + token_id_list + [self.get_eos_id()]

        return token_id_list

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

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        result_bytes = bytearray()

        special_ids = set(SPECIAL_IDS.values())

        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue

            token = self.id_to_token[token_id]

            if isinstance(token, bytes):
                result_bytes.extend(token)
            else:
                result_bytes.extend(token.encode("utf-8"))

        return bytes(result_bytes).decode("utf-8")
