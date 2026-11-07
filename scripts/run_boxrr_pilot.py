"""Bounded, cloud-only BOXRR pilot download and head-rotation extraction."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import requests

from headpose.boxrr import pseudonym, trajectories_from_user_tar

REPO = "cschell/boxrr-23"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/users"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def list_candidates(prefix: str, maximum_archive_bytes: int) -> list[dict[str, object]]:
    response = requests.get(
        f"{API}/{prefix}",
        params={"recursive": "false", "expand": "true", "limit": 100},
        timeout=60,
    )
    response.raise_for_status()
    return list(
        (
            item
            for item in response.json()
            if item.get("type") == "file"
            and str(item.get("path", "")).endswith(".tar")
            and int(item.get("size", 0)) <= maximum_archive_bytes
        ),
    )


def download(url: str, destination: Path, expected_bytes: int) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    if destination.stat().st_size != expected_bytes:
        raise RuntimeError("downloaded archive size differs from repository metadata")


def run(arguments: argparse.Namespace) -> dict[str, object]:
    salt = os.environ.get("BOXRR_MANIFEST_SALT")
    if not salt:
        raise RuntimeError("BOXRR_MANIFEST_SALT must be set and must not be committed")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefixes = list(dict.fromkeys(arguments.prefixes.lower()))
    if not prefixes or any(prefix not in "0123456789abcdef" for prefix in prefixes):
        raise ValueError("prefixes must contain hexadecimal characters")
    by_prefix = {
        prefix: sorted(
            list_candidates(prefix, arguments.max_archive_mb * 1024**2),
            key=lambda item: abs(
                int(item["size"]) - arguments.target_archive_mb * 1024**2
            ),
        )
        for prefix in prefixes
    }
    selected: list[dict[str, object]] = []
    total = 0
    while len(selected) < arguments.users:
        progress = False
        for prefix in prefixes:
            while by_prefix[prefix]:
                item = by_prefix[prefix].pop(0)
                size = int(item["size"])
                if size < arguments.min_archive_mb * 1024**2:
                    continue
                if total + size > arguments.max_total_mb * 1024**2:
                    continue
                selected.append(item)
                total += size
                progress = True
                break
            if len(selected) >= arguments.users:
                break
        if not progress:
            break
    if len(selected) < arguments.users:
        raise RuntimeError("bounded selection did not contain the requested number of users")

    report: dict[str, object] = {
        "source": REPO,
        "prefixes": prefixes,
        "download_bytes": total,
        "users": [],
    }
    manifest_path = output / "pilot-manifest.json"
    with tempfile.TemporaryDirectory(dir=arguments.scratch) as scratch:
        for item in selected:
            remote_path = str(item["path"])
            raw_user = Path(remote_path).stem
            user_hash = pseudonym(raw_user, salt)
            archive = Path(scratch) / f"{user_hash}.tar"
            download(f"{RESOLVE}/{remote_path}?download=true", archive, int(item["size"]))
            recordings = trajectories_from_user_tar(archive)
            user_dir = output / user_hash
            user_dir.mkdir(exist_ok=True)
            app_counts: dict[str, int] = {}
            for index, (_, trajectory) in enumerate(recordings):
                app_counts[trajectory.app_name] = app_counts.get(trajectory.app_name, 0) + 1
                np.savez_compressed(
                    user_dir / f"recording-{index:05d}.npz",
                    timestamps=trajectory.timestamps,
                    quaternions=trajectory.quaternions,
                )
            report["users"].append(
                {
                    "user_hash": user_hash,
                    "archive_bytes": int(item["size"]),
                    "recordings": len(recordings),
                    "applications": app_counts,
                }
            )
            manifest_path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"processed {len(report['users'])}/{len(selected)} users; "
                f"recordings={len(recordings)}",
                flush=True,
            )
            archive.unlink()
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    parser.add_argument("--prefixes", default="0")
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--max-archive-mb", type=int, default=50)
    parser.add_argument("--min-archive-mb", type=int, default=1)
    parser.add_argument("--target-archive-mb", type=int, default=20)
    parser.add_argument("--max-total-mb", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
