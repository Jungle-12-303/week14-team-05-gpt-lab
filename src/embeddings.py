# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length

        # 토큰 ID를 emb_dim 차원의 벡터로 변환
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        # 위치 ID를 emb_dim 차원의 벡터로 변환
        self.position_embedding = nn.Embedding(context_length, emb_dim)

        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """

        # position embedding 범위를 넘는 입력 길이를 검사하는 방어 코드
        seq_len = x.shape[1]
        if seq_len > self.context_length:
            raise ValueError(f"seq_len ({seq_len}) must be <= context_length ({self.context_length})")

        # 토큰 임베딩 -> 각 token ID를 emb_dim 차원의 벡터로 변환
        token_emb = self.token_embedding(x)
        
        # position_ids를 x와 같은 장치(CPU 또는 GPU)에 만든다. (device mismatch 예방 목적)
        position_ids = torch.arange(seq_len, device=x.device)

        # 위치 임베딩 -> position_ids(0, 1, ..., seq_len-1)를 위치별 emb_dim 차원 벡터로 변환
        pos_emb = self.position_embedding(position_ids)

        # 토큰 임베딩과 위치 임베딩을 더한다. (pos_emb는 batch 차원으로 broadcasting)
        input_emb = token_emb + pos_emb

        # 드롭아웃을 적용하여 반환 
        # tensor의 일부 값이 0으로 바뀐다. 드롭아웃 확률은 원소별로 독립적으로 적용된다.
        return self.dropout(input_emb)
        