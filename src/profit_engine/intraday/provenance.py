"""Provenance helpers: git sha of the code and a content hash of the data used."""

from __future__ import annotations

import hashlib
import subprocess

import pandas as pd

from profit_engine.intraday import config


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=config.REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_sha() -> str:
    """HEAD sha; with uncommitted changes to tracked files, ``<sha>-dirty-<hash of the diff>``.

    The diff hash makes two different uncommitted code states two different code versions
    in the trial registry. Untracked files are not covered: commit experiment code.
    """
    try:
        sha = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        diff = _git("diff", "HEAD", "--no-ext-diff") if dirty else ""
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    if not dirty:
        return sha
    return f"{sha}-dirty-{hashlib.sha256(diff.encode()).hexdigest()[:8]}"


def data_hash(df: pd.DataFrame | pd.Series) -> str:
    """Stable 16-hex-char hash of index + values, for the trial registry."""
    hashed = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]
