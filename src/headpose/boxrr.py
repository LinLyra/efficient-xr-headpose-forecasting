"""BOXRR-23 XROR extraction with explicit device-axis accounting."""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .quaternion import enforce_sign_continuity, normalize


@dataclass(frozen=True)
class HeadTrajectory:
    """One recording reduced to timestamps and scalar-first head quaternions."""

    timestamps: np.ndarray
    quaternions: np.ndarray
    app_name: str


def pseudonym(value: str, salt: str) -> str:
    """Return a non-reversible manifest identifier for a governed user id."""
    if not salt:
        raise ValueError("a non-empty manifest salt is required")
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:20]


def _device_offsets(devices: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    offsets: list[tuple[dict[str, Any], int]] = []
    cursor = 1
    for device in devices:
        offsets.append((device, cursor))
        cursor += len(device.get("axes", []))
    return offsets


def extract_head_trajectory(data: dict[str, Any]) -> HeadTrajectory:
    """Extract an HMD/HEAD rotation using the XROR device-axis declaration."""
    info = data.get("info", {})
    devices = info.get("hardware", {}).get("devices", [])
    candidates = [
        item
        for item in _device_offsets(devices)
        if str(item[0].get("joint", "")).upper() == "HEAD"
        or str(item[0].get("type", "")).upper() in {"HMD", "HEADSET"}
    ]
    if not candidates:
        raise ValueError("recording has no HEAD/HMD device")
    device, offset = candidates[0]
    axes = [str(axis).lower() for axis in device.get("axes", [])]
    aliases = {
        "x": ("i", "qx", "x_rot", "rotation_x"),
        "y": ("j", "qy", "y_rot", "rotation_y"),
        "z": ("k", "qz", "z_rot", "rotation_z"),
        "w": ("1", "l", "qw", "w_rot", "rotation_w"),
    }

    def axis_index(component: str) -> int:
        for name in aliases[component]:
            if name in axes:
                return offset + axes.index(name)
        raise ValueError(f"HEAD/HMD device lacks quaternion {component}-axis")

    frames = np.asarray(data.get("frames", []), dtype=np.float64)
    if frames.ndim != 2 or len(frames) < 2:
        raise ValueError("recording has fewer than two motion frames")
    columns = [axis_index(component) for component in ("w", "x", "y", "z")]
    if frames.shape[1] <= max(columns):
        raise ValueError("frame width is inconsistent with device axis declarations")
    timestamps = frames[:, 0]
    quaternions = frames[:, columns]
    norms = np.linalg.norm(quaternions, axis=1)
    valid = (
        np.isfinite(timestamps)
        & np.all(np.isfinite(quaternions), axis=1)
        & (norms > 1e-8)
    )
    timestamps, quaternions = timestamps[valid], quaternions[valid]
    increasing = np.concatenate(([True], np.diff(timestamps) > 0))
    timestamps, quaternions = timestamps[increasing], quaternions[increasing]
    if len(timestamps) < 2:
        raise ValueError("recording has fewer than two valid increasing frames")
    quaternions = enforce_sign_continuity(normalize(quaternions))
    app = info.get("software", {}).get("app", {})
    app_name = str(app.get("name", app.get("id", "unknown")))
    return HeadTrajectory(timestamps, quaternions, app_name)


def unpack_xror(payload: bytes) -> dict[str, Any]:
    """Decode XROR bytes using the official optional ``xror`` dependency."""
    try:
        from xror import XROR
    except ImportError as error:  # pragma: no cover - cloud integration path
        raise RuntimeError(
            "install the official parser with: pip install "
            "git+https://github.com/MetaGuard/xror.git"
        ) from error
    return XROR.unpack(payload).data


def trajectories_from_user_tar(path: str | Path) -> list[tuple[str, HeadTrajectory]]:
    """Stream XROR members from one user TAR without extracting raw files."""
    output: list[tuple[str, HeadTrajectory]] = []
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".xror"):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            try:
                trajectory = extract_head_trajectory(unpack_xror(source.read()))
            except (ValueError, RuntimeError):
                continue
            output.append((Path(member.name).name, trajectory))
    return output
