"""Shadow live log: candidate rules recorded every day next to the main signal, without changing the main signal.

Several studies found overlays that roughly halved the deepest loss of the 50-day-average rule but did not pass the
fair exam (too little data, or the advantage could be luck). History cannot settle that any more, so from
SHADOW_START each candidate's state after the last daily close is appended to a log, and only this log - never a
rerun of history - is used to judge them later.

The candidates were fixed on 2026-11-19, before any logged day, and are read from the "today" rows that their own
studies already compute (study id, row label, position in a "a｜b｜c" value). A row that is missing or unreadable
records nothing for that day; past days are never rewritten.

Score: holding the logged state from one logged close to the next BTC close (no costs), compounded, next to holding
BTC over the same days. Written to cache/_shadow.json; the signals collector shows it. States only, not advice.
"""
from __future__ import annotations

import json
import math
import re

from dashboard import common

PATH = common.CACHE / "_shadow.json"
LOG = common.STORE / "shadow_live_log.json"
STUDIES = common.CACHE / "_studies2.json"
SHADOW_START = "2026-11-17"

# id, name, study id, row label (exact), position in a "｜"-separated value
CANDIDATES = (
    ("ma50", "50 日均線（現行主訊號，對照）", "blockspace", "50 日均線（基準）", 0),
    ("ls1", "LS1：均線，但散戶多空比在一年最高 20% 時改拿現金", "ls_ratio", None, None),
    ("ls2", "LS2：均線，但散戶比大戶偏多在一年最高 20% 時改拿現金", "ls_ratio", None, None),
    ("f1ma", "F1＋均線：手續費 z 分數 > 0 而且在均線上才持有", "blockspace", "F1＋50 日均線：兩個都說好才持有", 0),
    ("l1", "L1：均線，而且美元對美元 Coinbase 溢價 7 日平均 > 0 才持有", "lead_followup", "比特幣：50 日均線｜L1｜L2", 1),
    ("b1", "B1：比特幣在 40 日均線上，而且四成以上的幣在 200 日均線上", "breadth_timing",
     "B1 三條件系統的兩條件：比特幣在 40 日均線上，且四成以上的幣在 200 日均線上", 0),
)


def _state(text) -> float | None:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if re.match(r"^持有(?!現金)", t):
        return 1.0
    if re.match(r"^(現金|空手|持有現金)", t):
        return 0.0
    return None


def _ls_rows(rows: list[dict]) -> dict[str, float | None]:
    """ls_ratio shows its rules as one row whose label lists them; find LS1 and LS2 wherever they appear."""
    out: dict[str, float | None] = {}
    for row in rows:
        label, value = str(row.get("label", "")), row.get("value")
        if not isinstance(value, str) or "比特幣" not in label and "BTC" not in label and "LS" not in label:
            continue
        names = [n.strip() for n in re.split(r"[｜|]", re.sub(r"^.*?[：:]", "", label))]
        values = [v.strip() for v in re.split(r"[｜|]", value)]
        if len(names) == len(values):
            for n, v in zip(names, values):
                if n.upper().startswith("LS1"):
                    out["ls1"] = _state(v)
                elif n.upper().startswith("LS2"):
                    out["ls2"] = _state(v)
        elif label.upper().startswith(("LS1", "LS2")):
            out[label[:3].lower()] = _state(value)
    return out


def today() -> dict[str, tuple[str, float]]:
    """{candidate id: (day of the close the state follows, 1 or 0)} from the studies' today rows."""
    data = json.loads(STUDIES.read_text(encoding="utf-8"))
    by_id = {s.get("id"): s for s in data.get("studies", []) if isinstance(s, dict)}
    out: dict[str, tuple[str, float]] = {}
    for cid, _name, sid, label, pos in CANDIDATES:
        s = by_id.get(sid)
        if not s:
            continue
        rows = [r for r in s.get("now", []) if isinstance(r, dict)]
        if label is None:
            st = _ls_rows(rows).get(cid)
        else:
            row = next((r for r in rows if r.get("label") == label), None)
            parts = re.split(r"[｜|]", row["value"]) if row and isinstance(row.get("value"), str) else []
            st = _state(parts[pos]) if pos is not None and len(parts) > pos else None
        if st is not None and s.get("asof"):
            out[cid] = (str(s["asof"])[:10], st)
    return out


def _f(v, digits: int = 2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def collect() -> list[dict]:
    try:
        log = json.loads(LOG.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - first run or damaged file starts a new log
        log = {}
    errors = {}
    try:
        for cid, (day, st) in today().items():
            if day >= SHADOW_START:
                log.setdefault(cid, {}).setdefault(day, st)  # append only
    except Exception as exc:  # noqa: BLE001 - a missing studies file only skips today
        errors["studies"] = f"{type(exc).__name__}: {exc}"[:200]
    common.STORE.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    from dashboard.studies2 import trend_ensemble as te
    close = te._daily_close()
    px = {d.strftime("%Y-%m-%d"): float(v) for d, v in close.items()}
    rows = []
    for cid, name, *_ in CANDIDATES:
        book = log.get(cid, {})
        days = sorted(d for d in book if d in px)
        growth, hold, n = 1.0, 1.0, 0
        for d in days:
            nxt = close.index[close.index > d]
            if not len(nxt):
                continue
            r = float(close[nxt[0]]) / px[d] - 1
            growth *= 1 + book[d] * r
            hold *= 1 + r
            n += 1
        last = days[-1] if days else None
        rows.append({"id": cid, "name": name, "days": n, "first": days[0] if days else None, "last": last,
                     "state": ("持有" if book[last] >= 0.5 else "現金") if last else None,
                     "rule": _f((growth - 1) * 100), "hold": _f((hold - 1) * 100)})
    payload = {"collected_utc": common.now_utc(), "start": SHADOW_START, "rows": rows, "errors": errors}
    common.CACHE.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return []


if __name__ == "__main__":
    collect()
    print(json.dumps(json.loads(PATH.read_text(encoding="utf-8")), ensure_ascii=False, indent=1))
