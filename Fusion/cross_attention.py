"""
cross_attention.py

Cross-attention for PAER-LLM.

THE MOST IMPORTANT BUG IN THE PROJECT
=====================================
The original module was mathematically correct as an attention implementation,
but it was called with sequences of length 1:

    psycho_embedding = psycho_embedding.unsqueeze(1)   # (B, 1, H)
    hubert_embedding = hubert_embedding.unsqueeze(1)   # (B, 1, H)
    fused, attn = self.cross_attention(psycho, hubert, hubert)

With one query token and one key token the score matrix is (B, 1, 1), and
``softmax`` over a single element is **always exactly 1.0**. So:

    output = attention @ V = 1.0 * V = value_proj(hubert_embedding)

The query - the entire psychoacoustic branch - is multiplied by a constant and
then discarded. It contributes nothing to the fused representation and receives
no gradient through the fusion path at all. The published architecture diagram
says "cross-attention fusion"; the code computes
``classifier(value_proj(hubert_encoder(hubert)))`` - a three-layer *linear*
network (no activations anywhere) on HuBERT features only.

That also explains the attention maps: they are all 1.0 by construction, so any
"attention weight" figure derived from them is uninformative.

Two fixes are provided:

* :class:`MultiHeadCrossAttention` - a correct, general implementation with
  multiple heads, an output projection, dropout, key padding masks and
  residual-friendly output. Use it when you have genuine sequences (e.g.
  frame-level HuBERT states rather than a mean-pooled vector).

* :class:`TokenCrossAttention` - projects each modality's single vector into K
  learned tokens *before* attending, so the softmax is over K > 1 entries and
  the mechanism is no longer degenerate. This is what lets you keep using the
  existing mean-pooled ``hubert_embeddings.csv`` without re-extracting
  frame-level features.

The original ``CrossAttention`` name is kept as a thin alias so old notebooks
still import, but it now raises if you hand it length-1 sequences.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """Standard scaled dot-product multi-head cross-attention.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming query/key/value vectors.
    hidden_dim : int
        Internal (and output) dimensionality. Must be divisible by ``n_heads``.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout applied to the attention weights and to the output projection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        # Missing in the original: without an output projection the heads are
        # never mixed, so multi-head attention degenerates into independent
        # per-head subspaces.
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : (B, Tq, D)
        key, value : (B, Tk, D)
        key_padding_mask : (B, Tk) bool, optional
            ``True`` marks positions to ignore.

        Returns
        -------
        output : (B, Tq, hidden_dim)
        attention : (B, n_heads, Tq, Tk)
        """
        if query.dim() != 3 or key.dim() != 3:
            raise ValueError(
                "expected 3-D (batch, time, dim) tensors; got "
                f"query {tuple(query.shape)}, key {tuple(key.shape)}"
            )

        Q = self._split(self.query(query))
        K = self._split(self.key(key))
        V = self._split(self.value(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attention = F.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)

        out = torch.matmul(attention, V)
        b, h, t, d = out.shape
        out = out.transpose(1, 2).contiguous().view(b, t, h * d)

        out = self.out_dropout(self.out_proj(out))
        return out, attention


class TokenCrossAttention(nn.Module):
    """Cross-attention between two *single-vector* modalities.

    Each modality is first expanded into ``n_tokens`` learned tokens, so the
    attention softmax runs over more than one key and the mechanism actually
    selects information instead of collapsing to the identity.

    Parameters
    ----------
    dim : int
        Shared embedding dimensionality of both modalities.
    n_tokens : int
        Number of tokens each modality is expanded into. Must be >= 2.
    """

    def __init__(
        self,
        dim: int,
        n_tokens: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if n_tokens < 2:
            raise ValueError(
                "n_tokens must be >= 2; with 1 token the softmax is constant "
                "and the attention is a no-op (this was the original bug)."
            )

        self.n_tokens = n_tokens
        self.q_tokens = nn.Linear(dim, dim * n_tokens)
        self.kv_tokens = nn.Linear(dim, dim * n_tokens)
        self.attn = MultiHeadCrossAttention(dim, dim, n_heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, query_vec: torch.Tensor, context_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query_vec : (B, D)
        context_vec : (B, D)

        Returns
        -------
        fused : (B, D)
        attention : (B, n_heads, n_tokens, n_tokens)
        """
        b, d = query_vec.shape

        q = self.q_tokens(query_vec).view(b, self.n_tokens, d)
        kv = self.kv_tokens(context_vec).view(b, self.n_tokens, d)

        out, attention = self.attn(q, kv, kv)
        # Residual keeps the query modality alive even if attention learns to
        # ignore the context - the original had no residual, so a degenerate
        # attention silently deleted a whole modality.
        fused = self.norm(out + q).mean(dim=1)
        return fused, attention


class CrossAttention(MultiHeadCrossAttention):
    """Backwards-compatible alias for old notebooks.

    Defaults to a single head to match the original maths, but refuses
    length-1 sequences, which is the configuration that made the module a
    no-op.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__(input_dim, hidden_dim, n_heads=n_heads, dropout=dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        if key.shape[1] < 2:
            raise ValueError(
                "cross-attention over a single key token is a no-op: "
                "softmax over one element is always 1.0, so the query "
                "modality is discarded. Use TokenCrossAttention for pooled "
                "vectors, or pass frame-level sequences."
            )
        return super().forward(query, key, value, key_padding_mask)


__all__ = ["MultiHeadCrossAttention", "TokenCrossAttention", "CrossAttention"]
