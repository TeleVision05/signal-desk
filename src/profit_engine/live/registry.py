"""Load runner settings and books from ``books.json`` (see ``config/live_books.example.json``).

Top-level keys are :class:`Settings` fields; ``books`` is a list of entries, each one of:

* ``{"book_id", "strategy": "demo_sma_trend", "symbol", "timeframe", "params"}`` - a built-in
* ``{"book_id", "strategy": "package.module:factory", "symbol", "timeframe", "params"}`` -
  ``factory(symbol=..., timeframe=..., **params)`` returns a Strategy
* ``{"book_id", "model": "<folder under the models dir, or an absolute path>"}`` - a versioned
  artifact (:mod:`profit_engine.live.models`); symbol and timeframe come from its metadata

Optional per book: ``title``, ``enabled`` (default true), ``starting_equity``,
``cost_bps_per_side`` and ``max_daily_loss`` (override the top-level values).
"""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from profit_engine.intraday.config import SYMBOLS, TIMEFRAMES
from profit_engine.live import paths
from profit_engine.live.models import load_model
from profit_engine.live.strategy import BUILTINS, Strategy

BOOK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


class ConfigError(ValueError):
    """books.json is missing, malformed, or describes a book the runner cannot trade."""


@dataclass(frozen=True)
class Settings:
    cost_bps_per_side: float = 5.0
    poll_seconds: float = 30.0
    jitter: float = 0.1
    max_daily_loss: float = 0.05
    stale_after_bars: float = 2.0
    max_price_jump: float = 0.25
    settle_seconds: float = 2.0
    starting_equity: float = 10_000.0
    notifications: bool = True
    notify_min_interval_s: float = 300.0


@dataclass(frozen=True)
class BookSpec:
    book_id: str
    title: str
    strategy: Strategy
    enabled: bool
    starting_equity: float
    cost_bps_per_side: float
    max_daily_loss: float

    @property
    def symbol(self) -> str:
        return self.strategy.symbol

    @property
    def timeframe(self) -> str:
        return self.strategy.timeframe

    @property
    def tf(self) -> pd.Timedelta:
        return pd.Timedelta(self.timeframe.replace("m", "min") if self.timeframe.endswith("m") else self.timeframe)


@dataclass(frozen=True)
class LiveConfig:
    settings: Settings
    books: tuple[BookSpec, ...]
    source: Path


def _settings(raw: dict) -> Settings:
    known = {f.name for f in fields(Settings)}
    values = {k: raw[k] for k in known if k in raw}
    settings = Settings(**values)
    if os.environ.get("SIGNAL_DESK_NOTIFY", "").strip() == "0":
        settings = Settings(**{**settings.__dict__, "notifications": False})
    return settings


def _import_factory(ref: str):
    module, _, attr = ref.partition(":")
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"strategy {ref!r} could not be imported") from exc


def build_strategy(entry: dict, models_dir: Path) -> Strategy:
    """The Strategy an entry describes (built-in, importable factory, or model folder)."""
    if "model" in entry:
        folder = Path(entry["model"]).expanduser()
        return load_model(folder if folder.is_absolute() else Path(models_dir) / folder)
    name = entry.get("strategy")
    if not name:
        raise ConfigError(f"book {entry.get('book_id')!r} needs 'strategy' or 'model'")
    factory = BUILTINS.get(name) or (_import_factory(name) if ":" in name else None)
    if factory is None:
        raise ConfigError(f"unknown strategy {name!r}; built-ins are {sorted(BUILTINS)}")
    where = {k: entry[k] for k in ("symbol", "timeframe") if entry.get(k)}
    try:
        return factory(**where, **(entry.get("params") or {}))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"book {entry.get('book_id')!r}: strategy {name!r} rejected its settings ({exc})") from exc


def _check_book(book_id: str, strategy: Strategy, seen: set[str]) -> None:
    if not BOOK_ID.match(book_id):
        raise ConfigError(f"book_id {book_id!r} must be lowercase letters, digits, '_' or '-'")
    if book_id in seen:
        raise ConfigError(f"book_id {book_id!r} appears twice")
    if not isinstance(strategy, Strategy):
        raise ConfigError(f"book {book_id!r}: strategy does not implement the Strategy protocol")
    if strategy.symbol not in SYMBOLS:
        raise ConfigError(f"book {book_id!r}: symbol must be one of {SYMBOLS}")
    if strategy.timeframe not in TIMEFRAMES:
        raise ConfigError(f"book {book_id!r}: timeframe must be one of {TIMEFRAMES}")
    if strategy.other_symbol not in (None, *SYMBOLS):
        raise ConfigError(f"book {book_id!r}: other_symbol must be one of {SYMBOLS}")


def _book(entry: dict, settings: Settings, models_dir: Path, seen: set[str]) -> BookSpec:
    book_id = str(entry.get("book_id", ""))
    strategy = build_strategy(entry, models_dir)
    for key in ("symbol", "timeframe"):
        if entry.get(key) and entry[key] != getattr(strategy, key):
            raise ConfigError(f"book {book_id!r}: {key} {entry[key]!r} differs from the strategy's")
    _check_book(book_id, strategy, seen)
    seen.add(book_id)
    return BookSpec(
        book_id=book_id,
        title=str(entry.get("title") or book_id),
        strategy=strategy,
        enabled=bool(entry.get("enabled", True)),
        starting_equity=float(entry.get("starting_equity", settings.starting_equity)),
        cost_bps_per_side=float(entry.get("cost_bps_per_side", settings.cost_bps_per_side)),
        max_daily_loss=float(entry.get("max_daily_loss", settings.max_daily_loss)),
    )


def parse_config(raw: dict, models_dir: Path | None = None, source: Path | None = None) -> LiveConfig:
    settings = _settings(raw)
    seen: set[str] = set()
    models = Path(models_dir or paths.models_dir())
    books = tuple(_book(e, settings, models, seen) for e in raw.get("books") or [])
    return LiveConfig(settings=settings, books=books, source=Path(source or "<memory>"))


def load_config(path: Path | None = None, models_dir: Path | None = None) -> LiveConfig:
    path = Path(path or paths.books_path())
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"books config {path} not found") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"books config {path} is not valid JSON (line {exc.lineno})") from exc
    return parse_config(raw, models_dir, path)
