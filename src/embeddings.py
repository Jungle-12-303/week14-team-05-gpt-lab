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
        # token_embedding, position_embedding, dropout을 정의하세요.
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
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
        # 토큰 임베딩
        embedded_token = self.token_embedding(x)

        # 위치 인덱스
        position_ids = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        # 위치 임베딩
        embedded_position = self.position_embedding(position_ids)

        # 토큰 임베딩, 위치 임베딩 더한 뒤 dropout 적용
        return self.dropout(embedded_token + embedded_position)