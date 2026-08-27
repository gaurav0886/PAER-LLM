"""
splits.py

Speaker-independent (actor-disjoint) train / validation / test splits.

Why this replaces ``train_test_split(..., stratify=labels)``
------------------------------------------------------------
RAVDESS has 24 actors, each producing the *same two* sentences across all eight
emotions. A random utterance-level split leaves the same speaker - and often
the same sentence at the same emotional intensity - in both train and test. The
classifier can then win by recognising the speaker rather than the emotion, and
the resulting accuracy does not transfer to an unseen talker.

Speaker-independent evaluation is the accepted protocol for RAVDESS in the
speech-emotion literature, and it is the number a thesis committee or reviewer
will ask for. Expect it to be substantially lower than the utterance-level
number - that is the point.

This module gives you:

* ``actor_split``          - a single fixed actor-disjoint 3-way split
* ``actor_kfold``          - grouped k-fold for cross-validated results
* ``leave_one_speaker_out``- LOSO, the strictest common protocol
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.constants import RANDOM_STATE, TEST_SIZE, VAL_SIZE


def actor_split(
    df: pd.DataFrame,
    *,
    actor_column: str = "actor",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    balance_gender: bool = True,
) -> dict[str, np.ndarray]:
    """Split rows into train / val / test with no actor appearing twice.

    Parameters
    ----------
    df : DataFrame
        Must contain ``actor_column`` (and ``gender`` if ``balance_gender``).
    test_size, val_size : float
        Fractions of *actors*, not of rows. ``val_size`` is taken from the
        actors remaining after the test actors are removed.
    balance_gender : bool
        Sample actors separately within each gender so every fold keeps a
        roughly even male/female mix. RAVDESS actor ids are odd=male,
        even=female, so an unbalanced draw is easy to make by accident.

    Returns
    -------
    dict
        ``{"train": idx, "val": idx, "test": idx, "actors": {...}}`` where the
        index arrays are positional indices into ``df``.
    """
    rng = np.random.default_rng(random_state)

    if balance_gender and "gender" in df.columns:
        groups = {
            g: np.array(sorted(sub[actor_column].unique()))
            for g, sub in df.groupby("gender")
        }
    else:
        groups = {"all": np.array(sorted(df[actor_column].unique()))}

    test_actors: list[int] = []
    val_actors: list[int] = []
    train_actors: list[int] = []

    for actors in groups.values():
        actors = actors.copy()
        rng.shuffle(actors)

        n = len(actors)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round((n - n_test) * val_size)))

        test_actors.extend(actors[:n_test].tolist())
        val_actors.extend(actors[n_test:n_test + n_val].tolist())
        train_actors.extend(actors[n_test + n_val:].tolist())

    assert not (set(train_actors) & set(val_actors)), "train/val actor overlap"
    assert not (set(train_actors) & set(test_actors)), "train/test actor overlap"
    assert not (set(val_actors) & set(test_actors)), "val/test actor overlap"

    actor_values = df[actor_column].to_numpy()
    idx = np.arange(len(df))

    return {
        "train": idx[np.isin(actor_values, train_actors)],
        "val": idx[np.isin(actor_values, val_actors)],
        "test": idx[np.isin(actor_values, test_actors)],
        "actors": {
            "train": sorted(train_actors),
            "val": sorted(val_actors),
            "test": sorted(test_actors),
        },
    }


def actor_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    actor_column: str = "actor",
    random_state: int = RANDOM_STATE,
):
    """Yield ``(train_idx, test_idx)`` for actor-disjoint k-fold CV.

    A single split of 24 actors is noisy - one unusual test speaker moves the
    headline number by several points. Reporting the mean and standard
    deviation over folds is far more defensible in a thesis.
    """
    from sklearn.model_selection import GroupKFold

    groups = df[actor_column].to_numpy()
    # Shuffle the actor -> group mapping so folds are not simply actors 1-5,
    # 6-10, ... which correlates with recording order.
    unique = np.array(sorted(np.unique(groups)))
    rng = np.random.default_rng(random_state)
    permuted = unique.copy()
    rng.shuffle(permuted)
    remap = {old: new for old, new in zip(unique, permuted)}
    shuffled_groups = np.array([remap[g] for g in groups])

    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(np.zeros(len(df)), groups=shuffled_groups)


def leave_one_speaker_out(df: pd.DataFrame, *, actor_column: str = "actor"):
    """Yield ``(train_idx, test_idx)`` holding out one actor at a time."""
    actor_values = df[actor_column].to_numpy()
    idx = np.arange(len(df))
    for actor in sorted(np.unique(actor_values)):
        mask = actor_values == actor
        yield idx[~mask], idx[mask]


def save_split(split: dict, path: str | Path) -> None:
    """Persist the actor assignment so evaluation reuses the exact split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split["actors"], indent=2), encoding="utf-8")


def load_split(path: str | Path) -> dict[str, list[int]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "actor_split",
    "actor_kfold",
    "leave_one_speaker_out",
    "save_split",
    "load_split",
]
