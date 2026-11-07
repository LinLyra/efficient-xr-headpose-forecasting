import numpy as np
import pytest

from headpose.boxrr import extract_head_trajectory, pseudonym


def test_extract_uses_declared_axes_and_scalar_first_output():
    data = {
        "info": {
            "hardware": {
                "devices": [
                    {"type": "OTHER", "axes": ["x"]},
                    {
                        "type": "HMD",
                        "joint": "HEAD",
                        "axes": ["x", "y", "z", "i", "j", "k", "1"],
                    },
                ]
            },
            "software": {"app": {"name": "Beat Saber"}},
        },
        "frames": [
            [-0.01, 99.0, 1, 2, 3, 0, 0, 0, 0],
            [0.0, 99.0, 1, 2, 3, 0, 0, 0, 1],
            [0.01, 99.0, 1, 2, 3, 0, 0, 0, -1],
            [0.02, 99.0, 1, 2, 3, 0, 0, np.nan, 1],
        ],
    }
    trajectory = extract_head_trajectory(data)
    assert trajectory.app_name == "Beat Saber"
    np.testing.assert_allclose(trajectory.timestamps, [0.0, 0.01])
    np.testing.assert_allclose(trajectory.quaternions, [[1, 0, 0, 0], [1, 0, 0, 0]])


def test_extract_rejects_recording_without_hmd():
    with pytest.raises(ValueError, match="HEAD/HMD"):
        extract_head_trajectory(
            {"info": {"hardware": {"devices": []}}, "frames": [[0.0], [0.1]]}
        )


def test_pseudonym_is_salted_and_stable():
    assert pseudonym("user-a", "salt") == pseudonym("user-a", "salt")
    assert pseudonym("user-a", "salt") != pseudonym("user-a", "other")
    with pytest.raises(ValueError, match="salt"):
        pseudonym("user-a", "")
