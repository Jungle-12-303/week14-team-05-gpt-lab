# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import math
import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        # GELU의 tanh 근사식:
        # GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        return 0.5 * x * (
            1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
        )


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의
        self.layers = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FeedForward 네트워크를 통과시킵니다."""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # attention, ffn, layernorm, dropout을 정의
        self.attention = MultiHeadAttention(
            d_model=d_model,  # 입력/출력 벡터 크기 (예: 512)
            n_heads=n_heads,  # attention head 개수, d_model을 나눠 병렬 처리
            drop_rate=drop_rate,  # dropout 확률 (과적합 방지)
            qkv_bias=qkv_bias,
        )
        self.ffn = FeedForward(d_model, dropout=drop_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop_shortcut = nn.Dropout(drop_rate)


    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """attention과 ffn을 residual connection으로 연결합니다."""
        shortcut = x  # 원본 x (어텐션 블록을 위한 숏컷 연결)
        x = self.norm1(x)
        x = self.attention(x, causal_mask=causal_mask)  # attention 적용
        x = self.drop_shortcut(x)  # attention을 거친 출력에 dropout을 적용
        x = x + shortcut

        shortcut = x  # attention 결과가 반영된 x를 FFN 블록의 shortcut으로 보존
        x = self.norm2(x)
        x = self.ffn(x)  # ffn 적용
        x = self.drop_shortcut(x)  # ffn을 거친 출력에 dropout을 적용
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config  # config는 모델 설정값을 담은 딕셔너리

        d_model = config["emb_dim"]
        vocab_size = config["vocab_size"]

        # embedding, blocks, final layernorm, lm_head를 정의

        # token id를 Transformer가 처리할 수 있는 벡터로 바꾸는 모듈 만들기
        # 토큰 임베딩과 위치 임베딩을 포함하고 있음.
        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            drop_rate=config["drop_rate"],
        )

        # TransformerBlock 여러 개를 순서대로 쌓기
        # (참고) 하나의 블록 안에는 LayerNorm, Causal Self-Attention, Residual connection, LayerNorm, FeedForward, Residual connection 이 들어있다.
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=d_model,
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],  # Q, K, V projection에서 bias 사용 여부
                ) 
                for _ in range(config["n_layers"])  # 블록을 n_layers개 만들겠다는 뜻
            ]
        )

        # 모든 TransformerBlock을 통과한 뒤 마지막으로 정규화를 한 번 적용
        self.final_layernorm = LayerNorm(d_model)

        # hidden vector를 vocab_size 크기의 logits로 변환
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
 
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """

        x = self.embedding(idx)
        x = self.transformer_blocks(x)
        x = self.final_layernorm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits
        
        #  PyTorch가 제공하는 표준 cross entropy 함수를 사용해서 loss 구하기
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        return (loss, logits)


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,  # 현재까지의 토큰 ID - (batch_size, current_seq_len)
    max_new_tokens: int,  # 앞으로 몇 개의 새 토큰을 생성할지
    context_size: int,
) -> torch.Tensor:
    """greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 모델이 처리할 수 있는 최대 context 길이만 남김
        with torch.no_grad():
            logits = model(idx_cond)  # 현재 context에 대한 logits 계산

        logits = logits[:, -1, :]  # 마지막 위치의 다음 토큰 예측만 사용
        probas = torch.softmax(logits, dim=-1)  # 점수를 확률로 변환
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # 가장 확률이 높은 토큰 선택
        idx = torch.cat((idx,idx_next), dim=1)  # 선택한 토큰을 기존 sequence 뒤에 붙임

    return idx