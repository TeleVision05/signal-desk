"""Paper book catalog (multi-run Signal Desk)."""

from __future__ import annotations

from typing import Any

BOOKS: dict[str, dict[str, Any]] = {
    "tqqq": {
        "id": "tqqq",
        "label": "TQQQ",
        "asset": "TQQQ",
        "run_id": "ml-tqqq-hgb-paper-v1",
        "strategy": "ml_TQQQ_hgb_h5_t0.55",
        "session": "us_rth",
        "tip": (
            "TQQQ is a fund that aims to move about 3× the daily return of the "
            "Nasdaq-100 (big tech stocks). Practice trades only."
        ),
    },
    "btc": {
        "id": "btc",
        "label": "BTC",
        "asset": "BTCUSDT",
        "run_id": "ml-btc-hgb-paper-v1",
        "strategy": "ml_BTCUSDT_hgb_h5_t0.55",
        "session": "crypto_24h",
        "tip": (
            "BTCUSDT is Bitcoin priced in USDT on Binance public daily bars. "
            "This book papers 24/7 beside the equity desk. Practice trades only."
        ),
    },
}

DEFAULT_BOOK = "tqqq"


def book_ids() -> list[str]:
    return list(BOOKS.keys())


def get_book(book_id: str | None) -> dict[str, Any]:
    key = (book_id or DEFAULT_BOOK).strip().lower()
    if key not in BOOKS:
        raise KeyError(key)
    return BOOKS[key]


def resolve_book(book_id: str | None) -> dict[str, Any]:
    try:
        return get_book(book_id)
    except KeyError:
        return BOOKS[DEFAULT_BOOK]
