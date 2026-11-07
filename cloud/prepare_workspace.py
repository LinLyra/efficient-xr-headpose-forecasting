"""Create the persistent cloud layout without accessing governed data."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    data_root = Path(os.environ.get("HEADPOSE_DATA_ROOT", "/workspace/headpose-data"))
    artifact_root = Path(
        os.environ.get("HEADPOSE_ARTIFACT_ROOT", "/workspace/headpose-artifacts")
    )
    directories = [
        data_root / "manifests",
        data_root / "processed" / "pilot",
        artifact_root / "checkpoints",
        artifact_root / "reports",
        artifact_root / "logs",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema_version": 1,
        "raw_data_persistent": False,
        "governed_data_accessed": False,
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
    }
    marker_path = data_root / "manifests" / "workspace.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(marker_path)


if __name__ == "__main__":
    main()
