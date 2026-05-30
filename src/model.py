# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # 학습 가능한 scale/shift 파라미터를 둔다.
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        # 각 토큰의 feature 축 기준 평균과 분산을 구한다.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        # 분산이 0에 가까워질 때를 대비해 eps를 더한다.
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta

        return out


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """torch 내장 GELU를 적용합니다."""
        return torch.nn.functional.gelu(x)


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # 각 위치별 표현을 넓혔다가 다시 d_model로 줄인다.
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
        # 토큰 간 문맥 혼합은 attention, 위치별 비선형 변환은 ffn이 담당한다.
        self.attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias
        )

        self.ffn = FeedForward(d_model=d_model, dropout=drop_rate)

        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """attention과 ffn을 residual connection으로 연결합니다."""
        # Pre-LN 구조로 attention 입력을 먼저 정규화한다.
        normed_x1 = self.layernorm1(x)
        attn_out = self.attention(normed_x1, causal_mask=causal_mask)
        attn_out = self.dropout(attn_out)
        # 원본 입력을 더해 정보 손실을 줄이고 학습을 안정화한다.
        x = x + attn_out

        # attention 결과를 다시 정규화한 뒤 FFN에 넣는다.
        normed_x2 = self.layernorm2(x)
        ffn_out = self.ffn(normed_x2)
        ffn_out = self.dropout(ffn_out)
        # 두 번째 residual 연결로 블록 출력을 만든다.
        x = x + ffn_out

        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    """
    GPT_CONFIG_SMALL = {
        "vocab_size": 1000,
        "context_length": 64,
        "emb_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # 토큰 임베딩과 위치 임베딩을 합쳐 입력 표현을 만든다.
        self.embedding = InputEmbedding(
            config["vocab_size"],
            config["emb_dim"],
            config["context_length"],
            config["drop_rate"],
        )
        # 여러 TransformerBlock을 순차적으로 통과시키며 문맥 표현을 깊게 만든다.
        self.blocks = [
            TransformerBlock(
                config["emb_dim"],
                config["n_heads"],
                config["drop_rate"],
                config["qkv_bias"],
            )
            for _ in range(config["n_layers"])
        ]
        # 마지막 정규화 뒤 vocabulary 크기로 사상해 다음 토큰 점수를 만든다.
        self.final_layernorm = LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

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
        # 입력 토큰 ID를 연속 벡터 표현으로 바꾼다.
        x = self.embedding(idx)

        # 각 블록이 문맥 정보를 한 단계씩 더 정교하게 만든다.
        for block in self.blocks:
            x = block(x)

        x = self.final_layernorm(x)
        logits = self.lm_head(x)

        if targets is not None:
            # 배치와 시퀀스 축을 펼쳐 전체 위치에 대해 CE loss를 계산한다.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss, logits

        return logits


def generate_text_simple(
    model: GPTModel,  # 다음 토큰을 예측할 GPT 모델
    idx: torch.Tensor,  # 현재까지의 토큰 ID 시퀀스
    max_new_tokens: int,  # 새로 생성할 토큰 수
    context_size: int,  # 모델이 볼 최대 문맥 길이
) -> torch.Tensor:
    """greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        # 모델 입력 길이를 넘지 않도록 최근 문맥만 남긴다.
        idx_cond = idx[:, -context_size:]

        # 생성 단계이므로 gradient 추적 없이 forward만 수행한다.
        with torch.no_grad():
            # 각 위치의 다음 토큰에 대한 점수 logits를 계산한다.
            logits = model(idx_cond)

        # 마지막 위치의 예측만 다음 토큰 선택에 사용한다.
        logits = logits[:, -1, :]

        # logits를 확률 분포처럼 해석할 수 있게 바꾼다.
        probas = torch.softmax(logits, dim=-1)

        # greedy decoding으로 가장 확률이 큰 토큰을 고른다.
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # 선택한 토큰을 뒤에 붙여 다음 반복의 입력을 갱신한다.
        idx = torch.cat((idx, idx_next), dim=1)

    # 원본 시퀀스와 새로 생성한 토큰이 합쳐진 결과를 반환한다.
    return idx
