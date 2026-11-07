import numpy as np
import pytest

from headpose.evaluation import (
    clustered_bootstrap_mean,
    paired_clustered_bootstrap_difference,
)
from headpose.splits import assign_user_split, hash_identifier, partition_user_hashes
from headpose.windows import make_window_indices, materialize_window


def test_user_split_is_stable_and_private() -> None:
    first = assign_user_split("raw-user-123", "research-only-salt")
    second = assign_user_split("raw-user-123", "research-only-salt")
    assert first == second
    assert first.user_hash != "raw-user-123"
    assert len(first.user_hash) == 64


def test_hash_changes_with_salt() -> None:
    assert hash_identifier("user", "salt-a") != hash_identifier("user", "salt-b")


def test_exact_partition_is_deterministic_disjoint_and_nonempty() -> None:
    users = [f"anonymous-{index}" for index in range(50)]
    first = partition_user_hashes(users, "fixed-study-seed")
    second = partition_user_hashes(reversed(users), "fixed-study-seed")
    assert first == second
    assert {split for split in first.values()} == {"train", "validation", "test"}
    assert list(first.values()).count("train") == 40
    assert list(first.values()).count("validation") == 5
    assert list(first.values()).count("test") == 5


def test_windows_never_cross_session_boundary() -> None:
    indices = make_window_indices("session-hash", 125, 100, 20)
    assert len(indices) == 6
    assert indices[-1].prediction_stop == 125
    poses = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (125, 1))
    context, target = materialize_window(poses, indices[-1])
    assert context.shape == (100, 4)
    assert target.shape == (20, 4)


def test_short_session_produces_no_windows() -> None:
    assert make_window_indices("session-hash", 119, 100, 20) == []


def test_clustered_bootstrap_counts_users_not_windows() -> None:
    errors = np.array([1.0, 1.0, 3.0, 3.0])
    users = np.array(["a", "a", "b", "b"])
    interval = clustered_bootstrap_mean(errors, users, resamples=500)
    assert interval.estimate == pytest.approx(2.0)
    assert interval.clusters == 2
    assert interval.lower <= interval.estimate <= interval.upper


def test_paired_bootstrap_negative_favors_candidate() -> None:
    candidate = np.array([1.0, 2.0, 1.0, 2.0])
    reference = candidate + 0.5
    users = np.array(["a", "a", "b", "b"])
    interval = paired_clustered_bootstrap_difference(
        candidate, reference, users, resamples=500
    )
    assert interval.estimate == pytest.approx(-0.5)
