#!/usr/bin/env python3
"""Reject common accidental data, privacy, and provenance leaks in the tree."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SKIP_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist"}
DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".json",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_CSV_BYTES = 1_000_000

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ID_HEADER_RE = re.compile(
    r"(?i)(?:^|[,;\t])\s*(?:(?:courier|order|tracking|source_tracking|shop|aoi)[_-]?)?id\s*(?:[,;\t]|$)"
)
JSON_ID_VALUE_RE = re.compile(
    r"(?i)[\"'](?:courier|order|tracking|source_tracking|shop|aoi)[_-]?id[\"']\s*:\s*[\"']?[A-Za-z0-9_-]{3,}"
)
FILENAME_ID_RE = re.compile(r"(?i)(?:courier|order|tracking)[_-]?(?:id)?[_-]?\d{3,}")
COORD_HEADER_RE = re.compile(
    r"(?i)(?:^|[,;\t])\s*(?:(?:courier_wave_start|source|target)[_-])?(?:lng|lat|longitude|latitude)\s*(?:[,;\t]|$)"
)
JSON_COORD_KEY_RE = re.compile(
    r"(?i)[\"'](?:(?:courier_wave_start|source|target)[_-])?(?:lng|lat|longitude|latitude)[\"']\s*:\s*-?\d"
)
DECIMAL_PAIR_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:\.\d{3,})?)\s*[,;\t]\s*(-?\d{1,3}(?:\.\d{3,})?)(?![\w.])")


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        yield path


def _looks_like_regional_coordinate_pair(text: str) -> bool:
    for match in DECIMAL_PAIR_RE.finditer(text):
        first, second = (float(match.group(1)), float(match.group(2)))
        if (20.0 <= first <= 30.0 and 110.0 <= second <= 125.0) or (
            20.0 <= second <= 30.0 and 110.0 <= first <= 125.0
        ):
            return True
    return False


def audit_tree(root: Path) -> list[str]:
    issues: list[str] = []
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        lower_name = path.name.casefold()
        if CJK_RE.search(path.name):
            issues.append(f"CJK filename: {relative}")
        if path.suffix.casefold() in {".ipynb", ".pdf"}:
            issues.append(f"forbidden artifact suffix: {relative}")
        if path.suffix.casefold() == ".csv" and path.stat().st_size > MAX_CSV_BYTES:
            issues.append(f"CSV exceeds 1MB: {relative}")
        if FILENAME_ID_RE.search(path.name):
            issues.append(f"courier/order identifier in filename: {relative}")

        try:
            raw = path.read_bytes()
        except OSError as exc:
            issues.append(f"unreadable file {relative}: {exc}")
            continue
        # Build the marker from components so this audit script does not flag
        # its own source while still detecting local absolute paths.
        local_path_marker = ("/" + "Users/").encode("utf-8")
        if local_path_marker in raw:
            issues.append(f"absolute local path: {relative}")

        if path.suffix.casefold() not in DATA_SUFFIXES:
            continue
        text = raw.decode("utf-8", errors="replace")
        if ID_HEADER_RE.search(text) or JSON_ID_VALUE_RE.search(text):
            issues.append(f"courier/order identifier pattern: {relative}")
        if (
            COORD_HEADER_RE.search(text)
            or JSON_COORD_KEY_RE.search(text)
            or _looks_like_regional_coordinate_pair(text)
        ):
            issues.append(f"coordinate pattern: {relative}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    issues = audit_tree(args.root.resolve())
    if issues:
        print("public-tree audit failed")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("public-tree audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
