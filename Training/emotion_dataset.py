"""
emotion_dataset.py

Torch ``Dataset`` for the fused psychoacoustic + HuBERT features.

Changes from the original
-------------------------
* ``torch.tensor(x)`` on an existing tensor raises a UserWarning and copies;
  ``torch.as_tensor`` is used instead and numpy arrays are made contiguous
  float32 up front.
* Added length and dimension validation. Previously a psycho/hubert/label
  length mismatch surfaced much later as an opaque indexing error deep in the
  training loop.
* Added ``class_weights()``, needed because RAVDESS "neutral" has half as many
  utterances as every other class (96 vs 192). Training with an unweighted
  cross-entropy and then reporting plain accuracy makes neutral look easy and
  inflates the headline number.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """Paired psychoacoustic / HuBERT feature dataset.

    Parameters
    ----------
    psycho_features : array-like, shape (N, psycho_dim)
    hubert_features : array-like, shape (N, hubert_dim)
    labels : array-like, shape (N,)
        Integer class indices.
    """

    def __init__(
        self,
        psycho_features,
        hubert_features,
        labels,
    ) -> None:
        psycho = np.ascontiguousarray(psycho_features, dtype=np.float32)
        hubert = np.ascontiguousarray(hubert_features, dtype=np.float32)
        y = np.ascontiguousarray(labels, dtype=np.int64)

        if not (len(psycho) == len(hubert) == len(y)):
            raise ValueError(
                "length mismatch: psycho "
                f"{len(psycho)}, hubert {len(hubert)}, labels {len(y)}"
            )
        if psycho.ndim != 2 or hubert.ndim != 2:
            raise ValueError("features must be 2-D (n_samples, n_features)")

        self.psycho_features = torch.as_tensor(psycho)
        self.hubert_features = torch.as_tensor(hubert)
        self.labels = torch.as_tensor(y)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        return (
            self.psycho_features[index],
            self.hubert_features[index],
            self.labels[index],
        )

    @property
    def psycho_dim(self) -> int:
        return int(self.psycho_features.shape[1])

    @property
    def hubert_dim(self) -> int:
        return int(self.hubert_features.shape[1])

    def class_weights(self, num_classes: int | None = None) -> torch.Tensor:
        """Inverse-frequency weights for ``nn.CrossEntropyLoss(weight=...)``."""
        n_classes = num_classes or int(self.labels.max().item()) + 1
        counts = torch.bincount(self.labels, minlength=n_classes).float()
        counts = counts.clamp(min=1.0)
        weights = counts.sum() / (n_classes * counts)
        return weights


__all__ = ["EmotionDataset"]
