"""
multimodal_model.py

PAER-LLM multimodal emotion recognition network.

Problems in the original architecture
-------------------------------------
1. **No non-linearity anywhere.** ``psycho_encoder``, ``hubert_encoder``, the
   attention projections and ``classifier`` were all bare ``nn.Linear``. A
   composition of linear maps is a single linear map, so the whole "deep
   multimodal network" had exactly the expressive power of multinomial logistic
   regression on the input features. Activations, normalisation and dropout are
   now present.

2. **The psychoacoustic branch was dead.** See the long note in
   ``Fusion/cross_attention.py``: attention over one key token is identically
   1.0, so the fused embedding equalled ``value_proj(hubert_embedding)`` and the
   34 psychoacoustic features never reached the classifier. The entire
   contribution claim of the thesis rested on a branch that was disconnected.
   Fusion is now genuinely bidirectional.

3. **No regularisation.** No dropout, no weight decay, no normalisation, on a
   1440-utterance dataset with a 768-dim input. Dropout and LayerNorm added.

4. **Nothing recorded which feature order the checkpoint expects.** The model
   now stores its configuration, and ``save``/``load`` round-trip it, so a
   checkpoint can never be silently loaded into a mismatched architecture.

Fusion modes
------------
``fusion="cross_attention"`` (default)
    Bidirectional token cross-attention - psycho attends to HuBERT and HuBERT
    attends to psycho. This is the honest version of the intended design.
``fusion="gated"``
    Learned per-dimension gate between the two modalities. Cheap, strong, and
    a good ablation baseline.
``fusion="concat"``
    Plain concatenation + MLP. Use as the "does fusion help at all?" control.

Report all three in the thesis: an ablation table is what makes the
cross-attention claim credible.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from Fusion.cross_attention import TokenCrossAttention
from Utils.constants import HUBERT_DIM, NUM_CLASSES, PSYCHO_DIM


class _Encoder(nn.Module):
    """Two-layer projection with normalisation, activation and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalEmotionModel(nn.Module):
    """Psychoacoustic + HuBERT emotion classifier.

    Parameters
    ----------
    psycho_dim : int
        Number of psychoacoustic features (5 pitch + 5 loudness + 24 bark = 34).
    hubert_dim : int
        HuBERT embedding size (768 for hubert-base).
    hidden_dim : int
        Shared latent width.
    num_classes : int
        Number of emotion classes.
    fusion : {"cross_attention", "gated", "concat"}
    n_tokens, n_heads : int
        Cross-attention configuration (ignored for other fusion modes).
    dropout : float
    """

    def __init__(
        self,
        psycho_dim: int = PSYCHO_DIM,
        hubert_dim: int = HUBERT_DIM,
        hidden_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        *,
        fusion: str = "cross_attention",
        n_tokens: int = 4,
        n_heads: int = 4,
        dropout: float = 0.3,
        modality: str = "both",
    ) -> None:
        super().__init__()

        if fusion not in {"cross_attention", "gated", "concat"}:
            raise ValueError(f"unknown fusion mode: {fusion!r}")
        if modality not in {"both", "psycho", "hubert"}:
            raise ValueError(f"unknown modality: {modality!r}")

        self.config = {
            "psycho_dim": psycho_dim,
            "hubert_dim": hubert_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "fusion": fusion,
            "n_tokens": n_tokens,
            "n_heads": n_heads,
            "dropout": dropout,
            "modality": modality,
        }
        self.fusion_mode = fusion
        self.modality = modality

        # Unimodal ablations keep the SAME encoder depth and the SAME
        # classifier head as the multimodal model, so any difference is
        # attributable to the missing modality rather than to capacity.
        self.psycho_encoder = (
            _Encoder(psycho_dim, hidden_dim, dropout)
            if modality in {"both", "psycho"} else None
        )
        self.hubert_encoder = (
            _Encoder(hubert_dim, hidden_dim, dropout)
            if modality in {"both", "hubert"} else None
        )

        # All fusion operators emit exactly hidden_dim, so the classification
        # head has identical capacity in every condition. Without this, the
        # concat/cross-attention heads were twice the width of the gated head
        # and Table 2 compared head capacity as well as fusion operator.
        fused_dim = hidden_dim

        if modality != "both":
            pass
        elif fusion == "cross_attention":
            # Bidirectional: each modality queries the other. The original was
            # one-directional *and* degenerate.
            self.psycho_to_hubert = TokenCrossAttention(
                hidden_dim, n_tokens=n_tokens, n_heads=n_heads, dropout=dropout
            )
            self.hubert_to_psycho = TokenCrossAttention(
                hidden_dim, n_tokens=n_tokens, n_heads=n_heads, dropout=dropout
            )
            self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        else:  # concat
            self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fusion_norm = nn.LayerNorm(fused_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.embedding_dim = fused_dim

    def n_trainable(self) -> int:
        """Trainable parameter count (excludes the frozen HuBERT encoder)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        psycho_features: torch.Tensor,
        hubert_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        psycho_features : (B, psycho_dim)
        hubert_features : (B, hubert_dim)

        Returns
        -------
        logits : (B, num_classes)
        embedding : (B, embedding_dim)
            The fused representation, handed to the LLM prompt builder.
        attention : (B, n_heads, n_tokens, n_tokens) or None
        """
        attention = None

        if self.modality == "psycho":
            fused = self.fusion_norm(self.psycho_encoder(psycho_features))
            return self.classifier(fused), fused, None

        if self.modality == "hubert":
            fused = self.fusion_norm(self.hubert_encoder(hubert_features))
            return self.classifier(fused), fused, None

        p = self.psycho_encoder(psycho_features)
        h = self.hubert_encoder(hubert_features)

        if self.fusion_mode == "cross_attention":
            p_att, attention = self.psycho_to_hubert(p, h)
            h_att, _ = self.hubert_to_psycho(h, p)
            fused = self.fuse_proj(torch.cat([p_att, h_att], dim=-1))
        elif self.fusion_mode == "gated":
            g = self.gate(torch.cat([p, h], dim=-1))
            fused = g * p + (1.0 - g) * h
        else:
            fused = self.fuse_proj(torch.cat([p, h], dim=-1))

        fused = self.fusion_norm(fused)
        logits = self.classifier(fused)

        return logits, fused, attention

    # ------------------------------------------------------------------
    # Checkpointing that cannot silently mismatch
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()},
            path,
        )
        path.with_suffix(".config.json").write_text(
            json.dumps(self.config, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "MultimodalEmotionModel":
        """Load a checkpoint saved by :meth:`save`.

        Raises a clear error on the old-format checkpoints (a bare
        ``state_dict``), which belong to the previous architecture and cannot
        be loaded into this one.
        """
        blob = torch.load(path, map_location=map_location, weights_only=False)

        if not isinstance(blob, dict) or "state_dict" not in blob:
            raise ValueError(
                f"{path} is an old-format checkpoint (bare state_dict) from the "
                "previous architecture. It was trained on duplicated data with "
                "a degenerate fusion layer - retrain with Training/train.py "
                "rather than trying to load it."
            )

        model = cls(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


__all__ = ["MultimodalEmotionModel"]
