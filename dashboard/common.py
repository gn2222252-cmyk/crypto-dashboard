"""Shared helpers for the dashboard collectors. See dashboard/SPEC2.md (and SPEC.md for the collector interface).

The dashboard is a stand-alone display product. Collectors fetch live public data with a fixed user agent and no
keys; a source that blocks us is reported as failed, never worked around.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
STORE = CACHE / "_store"  # daily values accumulated run after run, for sources that only keep ~30 days
OUT = HERE / "out"
HEADERS = {"User-Agent": "btc-research-dashboard"}
PAUSE = 0.4
GROUPS = ("price", "cycle", "derivatives", "onchain", "flows", "whales", "macro", "breadth", "sentiment")
EVIDENCE = ("A", "B", "C")  # A: cycle insurance thresholds (weak evidence); B: published evidence; C: reference
READINGS = ("bullish", "neutral", "bearish", "none")
HISTORY_POINTS = 1500  # about four years of daily points


def get_json(url: str, params: dict | None = None, timeout: int = 30):
    """GET a public JSON endpoint politely (fixed user agent, short pause, raises on HTTP errors)."""
    reply = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    reply.raise_for_status()
    time.sleep(PAUSE)
    return reply.json()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_history(history: list | None) -> list:
    """[[YYYY-MM-DD, number], ...] oldest first, one point per day (the later one wins), finite numbers only."""
    days: dict[str, float] = {}
    for point in history or []:
        if len(point) >= 2 and _finite(point[1]):
            days[str(point[0])[:10]] = point[1]
    return [[day, days[day]] for day in sorted(days)][-HISTORY_POINTS:]


def accumulate(id: str, points: list) -> list:
    """Merge points into cache/_store/<id>.json and return the whole stored history (oldest first).

    For sources that only serve a short window (or just the current value): every run adds what it fetched, so the
    history grows by itself. A day that is already stored is overwritten, never duplicated. If the store cannot be
    read or written the points fetched now are returned unchanged.
    """
    fresh = clean_history(points)
    path = STORE / f"{id}.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(stored, dict):
            stored = {}
    except Exception:  # noqa: BLE001 - a damaged store file starts over instead of breaking the collector
        stored = {}
    stored = {str(day)[:10]: value for day, value in stored.items() if _finite(value)}
    stored.update({day: value for day, value in fresh})
    try:
        STORE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(stored.items())), ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return [[day, stored[day]] for day in sorted(stored)][-HISTORY_POINTS:]


def indicator(*, id: str, group: str, asset: str, name: str, value, unit: str, asof: str, source: str,
              source_url: str, evidence: str, reading: str = "none", rule: str = "", note: str = "",
              history: list | None = None, display: str | None = None) -> dict:
    """One dashboard indicator. history is [[YYYY-MM-DD, number], ...], oldest first, at most HISTORY_POINTS."""
    assert group in GROUPS, group
    assert evidence in EVIDENCE, evidence
    assert reading in READINGS, reading
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    return {"id": id, "group": group, "asset": asset, "name": name, "value": value, "unit": unit,
            "display": display, "asof": asof, "source": source, "source_url": source_url, "evidence": evidence,
            "reading": reading, "rule": rule, "note": note, "history": clean_history(history), "error": None}


def failed(*, id: str, group: str, asset: str, name: str, source: str, source_url: str, error: str) -> dict:
    """Placeholder for an indicator whose fetch failed, so the page can say so instead of dropping it."""
    return {"id": id, "group": group, "asset": asset, "name": name, "value": None, "unit": "", "display": None,
            "asof": None, "source": source, "source_url": source_url, "evidence": "C", "reading": "none",
            "rule": "", "note": "", "history": [], "error": error[:300]}


def save(name: str, rows: list[dict]) -> Path:
    """Write one collector's indicators to cache/<name>.json."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    path.write_text(json.dumps({"collected_utc": now_utc(), "indicators": rows}, ensure_ascii=False),
                    encoding="utf-8")
    return path
