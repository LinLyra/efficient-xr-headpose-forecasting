"""Privacy-preserving, deterministic user-level data splits."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Iterable


@dataclass(frozen=True)
class SplitAssignment:
    user_hash: str
    split: str


def hash_identifier(identifier: str, salt: str) -> str:
    if not identifier or not salt:
        raise ValueError("identifier and salt must be non-empty")
    return hmac.new(salt.encode(), identifier.encode(), hashlib.sha256).hexdigest()


def assign_user_split(
    user_identifier: str,
    salt: str,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> SplitAssignment:
    """Assign every session from a user to one stable split."""
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("fractions must leave a non-empty test interval")
    digest = hash_identifier(user_identifier, salt)
    bucket = int(digest[:16], 16) / float(16**16)
    if bucket < train_fraction:
        split = "train"
    elif bucket < train_fraction + validation_fraction:
        split = "validation"
    else:
        split = "test"
    return SplitAssignment(user_hash=digest, split=split)


def partition_user_hashes(
    user_hashes: Iterable[str],
    seed: str,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, str]:
    """Return an exact-size, deterministic user-disjoint partition.

    Hash-bucket assignment can accidentally produce an empty validation or test
    set on a modest cohort. Sorting by a keyed digest preserves privacy and
    determinism while allocating exact cohort counts.
    """
    users = sorted(set(user_hashes))
    if len(users) < 3:
        raise ValueError("at least three unique users are required")
    if not seed:
        raise ValueError("seed must be non-empty")
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("fractions must leave a non-empty test interval")
    ordered = sorted(
        users,
        key=lambda user: hmac.new(
            seed.encode(), user.encode(), hashlib.sha256
        ).digest(),
    )
    train_count = max(1, int(len(ordered) * train_fraction))
    validation_count = max(1, int(len(ordered) * validation_fraction))
    if train_count + validation_count >= len(ordered):
        train_count = len(ordered) - validation_count - 1
    assignments: dict[str, str] = {}
    for index, user in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        assignments[user] = split
    return assignments


def partition_user_hashes_exact(
    user_hashes: Iterable[str],
    seed: str,
    counts: Mapping[str, int],
) -> dict[str, str]:
    """Return a deterministic keyed partition with exact named split counts."""
    users = sorted(set(user_hashes))
    required = {"train", "validation", "test"}
    if set(counts) != required:
        raise ValueError("counts must contain train, validation and test")
    if any(int(value) < 1 for value in counts.values()):
        raise ValueError("every split count must be positive")
    if sum(int(value) for value in counts.values()) != len(users):
        raise ValueError("split counts must equal the number of unique users")
    if not seed:
        raise ValueError("seed must be non-empty")
    ordered = sorted(
        users,
        key=lambda user: hmac.new(
            seed.encode(), user.encode(), hashlib.sha256
        ).digest(),
    )
    train_stop = int(counts["train"])
    validation_stop = train_stop + int(counts["validation"])
    assignments: dict[str, str] = {}
    for index, user in enumerate(ordered):
        if index < train_stop:
            split = "train"
        elif index < validation_stop:
            split = "validation"
        else:
            split = "test"
        assignments[user] = split
    return assignments
