"""Run one command through an authenticated Jupyter terminal."""

from __future__ import annotations

import argparse
import os
import secrets

from jupyter_bootstrap_client import authenticate, terminal_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()
    password = os.environ.get("JUPYTER_PASSWORD")
    if not password:
        raise RuntimeError("JUPYTER_PASSWORD is required")
    marker = f"__COMMAND_DONE_{secrets.token_hex(8)}__"
    command = (
        f"{{ {args.command}; }}; status=$?; "
        f"echo __COMMAND_STATUS__=$status; echo {marker}"
    )
    output = terminal_command(
        authenticate(args.url.rstrip("/"), password),
        args.url.rstrip("/"),
        command,
        marker,
    )
    print(output)
    if "__COMMAND_STATUS__=0" not in output:
        raise RuntimeError("remote command failed")


if __name__ == "__main__":
    main()
