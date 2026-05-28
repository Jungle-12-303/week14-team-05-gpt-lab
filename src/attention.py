# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기

1
입력 x에서 Q, K, V를 만듭니다.
2
head 개수만큼 차원을 나눕니다.
3
Q @ K.T / sqrt(head_dim)으로 attention score를 계산합니다.
4
causal mask로 미래 token 위치를 inf로 가립니다.
5
softmax로 attention weight를 만듭니다.
6
attention weight와 V를 곱합니다.
7
head를 다시 합치고 output projection을 적용합니다.

shape 흐름:

Plain text
x: (B, T, C)
q, k, v: (B, T, C)
q, k, v after split: (B, n_heads, T, head_dim)
attention weights: (B, n_heads, T, T)
output: (B, T, C)
신경 쓸 점:

•
d_model % n_heads == 0이어야 합니다.

•
mask는 현재 위치가 미래 위치를 보지 못하게 만드는 장치입니다.

•
return_attention_weights=True일 때 테스트가 attention weight shape와 causal mask를 확인합니다.
    """

    def __init__(
        self,
        d_model: int,       # 입력/출력 벡터 크기 (예: 512)
        n_heads: int,       # attention head 개수, d_model을 나눠 병렬 처리
        drop_rate: float = 0.1,  # dropout 확률 (과적합 방지)
        qkv_bias: bool = False,  # Q/K/V projection에 bias 항 추가 여부
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # qkv projection, output projection, dropout을 정의하세요.
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        raise NotImplementedError("MultiHeadAttention.forward를 구현하세요.")
