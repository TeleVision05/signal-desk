"""Provenance helpers: git sha of the code and a content hash of the data used."""

from __future__ import annotations

import hashlib
import subprocess

import pandas as pd

from profit_engine.intraday import config


def git_sha() -> str:
    """HEAD sha, suffixed with ``-dirty`` when tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def data_hash(df: pd.DataFrame | pd.Series) -> str:
    """Stable 16-hex-char hash of index + values, for the trial registry."""
    hashed = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]
