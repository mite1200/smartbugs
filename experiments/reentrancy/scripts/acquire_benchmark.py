#!/usr/bin/env python3
"""Acquire the pinned Ca' Foscari reentrancy benchmark checkout."""

import argparse
import subprocess
from pathlib import Path

REPOSITORY = (
    "https://github.com/ca-foscari-reentrancy-research-group/" "reentrancy-detection-llms.git"
)
COMMIT = "d7c9496121931cff7c55b2047d18fb048fefe2cb"


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def acquire(destination: Path) -> None:
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise RuntimeError(f"destination exists but is not a Git checkout: {destination}")
        run(["git", "fetch", "origin", COMMIT], cwd=destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", REPOSITORY, str(destination)])
    run(["git", "checkout", "--detach", COMMIT], cwd=destination)
    actual = run(["git", "rev-parse", "HEAD"], cwd=destination)
    if actual != COMMIT:
        raise RuntimeError(f"expected benchmark commit {COMMIT}, found {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    acquire(args.destination.resolve())


if __name__ == "__main__":
    main()
