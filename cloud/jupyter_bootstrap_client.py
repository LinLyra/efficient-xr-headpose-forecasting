"""Upload and verify this project through an authenticated Jupyter terminal.

The password is read only from ``JUPYTER_PASSWORD`` and is never printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import websocket


def authenticate(base_url: str, password: str) -> requests.Session:
    session = requests.Session()
    login_url = f"{base_url.rstrip('/')}/login"
    response = session.get(login_url, timeout=30)
    response.raise_for_status()
    xsrf = session.cookies.get("_xsrf")
    if not xsrf:
        raise RuntimeError("Jupyter login did not issue an XSRF cookie")
    response = session.post(
        login_url,
        data={"_xsrf": xsrf, "password": password},
        headers={"X-XSRFToken": xsrf},
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()
    if "/login" in response.url:
        raise RuntimeError("Jupyter authentication failed")
    return session


def upload_archive(
    session: requests.Session, base_url: str, archive: Path, remote_name: str
) -> None:
    encoded = base64.b64encode(archive.read_bytes()).decode("ascii")
    response = session.put(
        f"{base_url.rstrip('/')}/api/contents/{remote_name}",
        json={"type": "file", "format": "base64", "content": encoded},
        headers={"X-XSRFToken": session.cookies["_xsrf"]},
        timeout=120,
    )
    response.raise_for_status()


def terminal_command(
    session: requests.Session, base_url: str, command: str, marker: str
) -> str:
    headers = {"X-XSRFToken": session.cookies["_xsrf"]}
    response = session.post(
        f"{base_url.rstrip('/')}/api/terminals", headers=headers, timeout=30
    )
    response.raise_for_status()
    terminal_name = response.json()["name"]
    parsed = urlparse(base_url)
    websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
    socket_url = (
        f"{websocket_scheme}://{parsed.netloc}/terminals/websocket/{terminal_name}"
    )
    cookie = "; ".join(f"{key}={value}" for key, value in session.cookies.items())
    output: list[str] = []
    marker_occurrences = 0
    connection = websocket.create_connection(
        socket_url,
        cookie=cookie,
        origin=base_url,
        timeout=30,
    )
    try:
        # A newly-created terminal may accept websocket frames before its shell
        # is ready. Wait for the initial prompt; otherwise the first command is
        # intermittently echoed but never executed.
        ready_deadline = time.monotonic() + 30
        while time.monotonic() < ready_deadline:
            try:
                initial = json.loads(connection.recv())
            except (TimeoutError, websocket.WebSocketTimeoutException):
                continue
            if len(initial) >= 2 and initial[0] == "stdout":
                output.append(initial[1])
                if initial[1].rstrip().endswith(("#", "$")):
                    break
        else:
            raise TimeoutError("cloud terminal did not become ready")
        connection.send(json.dumps(["stdin", command + "\r"]))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                message = json.loads(connection.recv())
            except (TimeoutError, websocket.WebSocketTimeoutException):
                continue
            if len(message) >= 2 and message[0] == "stdout":
                output.append(message[1])
                marker_occurrences += message[1].count(marker)
                # The terminal echoes the submitted command once. Completion is
                # proven only when the command's final echo emits marker #2.
                if marker_occurrences >= 2:
                    return "".join(output)
        raise TimeoutError("cloud bootstrap did not emit its completion marker")
    finally:
        connection.close()
        session.delete(
            f"{base_url.rstrip('/')}/api/terminals/{terminal_name}",
            headers=headers,
            timeout=30,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    arguments = parser.parse_args()
    password = os.environ.get("JUPYTER_PASSWORD")
    if not password:
        raise RuntimeError("JUPYTER_PASSWORD is required")
    base_url = arguments.url.rstrip("/")
    session = authenticate(base_url, password)
    upload_archive(
        session, base_url, arguments.archive, "workspace/headpose-project.tar.gz"
    )
    marker = "__HEADPOSE_BOOTSTRAP_DONE__"
    steps = " && ".join(
        [
            "rm -rf /workspace/xr-headpose-forecasting",
            "mkdir -p /workspace/xr-headpose-forecasting",
            "tar --no-same-owner -xzf /workspace/headpose-project.tar.gz -C /workspace/xr-headpose-forecasting",
            "cd /workspace/xr-headpose-forecasting",
            "python -m pip install -q -e '.[dev]'",
            "python cloud/prepare_workspace.py",
            "PYTHONPATH=src pytest -q",
            "ruff check .",
            "python -c \"import json,torch,pathlib; result={'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}; pathlib.Path('/workspace/headpose-artifacts/reports/cloud-smoke.json').write_text(json.dumps(result,indent=2)+'\\n'); print(result)\"",
        ]
    )
    command = f"{{ {steps}; }}; status=$?; echo __HEADPOSE_STATUS__=$status; echo {marker}"
    output = terminal_command(session, base_url, command, marker)
    print(output)
    if "__HEADPOSE_STATUS__=0" not in output:
        raise RuntimeError("cloud bootstrap failed; see terminal output above")


if __name__ == "__main__":
    main()
