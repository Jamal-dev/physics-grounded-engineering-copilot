"""Fail when publication wording contains retired project labels."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
RETIRED_PATTERNS = (
    re.compile(r"\b" + "A" + r"14\b", re.IGNORECASE),
    re.compile(r"\b" + "Mis" + r"tral\b", re.IGNORECASE),
    re.compile(r"\b" + "de" + r"mo\b", re.IGNORECASE),
    re.compile(r"\b" + "show" + r"case\b", re.IGNORECASE),
    re.compile(r"\b" + "proof" + r"[ -]of[ -]concept\b", re.IGNORECASE),
    re.compile(r"\b" + "toy" + r" example\b", re.IGNORECASE),
)


def tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def violations() -> list[str]:
    failures: list[str] = []
    for path in tracked_text_files():
        if path.resolve() == THIS_FILE or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in RETIRED_PATTERNS):
                failures.append(f"{path.relative_to(ROOT)}:{line_number}")
    return failures


def main() -> None:
    failures = violations()
    if failures:
        raise SystemExit("Retired publication wording found at: " + ", ".join(failures))
    print("Public language scan passed.")


if __name__ == "__main__":
    main()
