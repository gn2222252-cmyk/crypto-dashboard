"""Historical study "overnight": two published bitcoin timing strategies from Quantpedia, redone on Binance data and
followed past their own samples.

1. "Overnight after a 10-day high" (Quantpedia, 2024-11-12; Gemini hourly data 2015-10-08 to 2024-10-15, in-sample
   until the first US bitcoin futures ETF on 2021-10-18, out-of-sample after): if bitcoin is at a 10-day high at the
   New York Stock Exchange close (16:00 New York time), hold it until the next NYSE open, else stay in cash. The page
   defines the day session as 10:00-16:00 New York time, so the "open" here is 10:00. The final published version
   only trades the Friday->Monday, Monday->Tuesday and Tuesday->Wednesday nights.
2. "Hold only 21:00-23:00 UTC" (Quantpedia, 2023-09-13; Gemini hourly data 2015-10-09 to 2023-06-30), and its variant
   that only trades on days whose 30-day hourly volatility (taken at 00:00 UTC) is above its one-year moving median.

Besides the strategies as published the study separates the pieces (every overnight, every US session, weekends, the
10-day-high condition on a 24-hour hold and on the following day session), varies the cost, picks one of five
variants at the start of each year from the data before it, and ranks the published three-night choice among all 31
possible choices of nights.

Data: Binance BTCUSDT 1-hour spot candles since 2017-08. The store kept by studies2.seasonality
(cache/_store/seasonality_hourly.csv) is read, never written; hours it does not have yet are fetched through
collectors.price._binance and kept in this module's own cache/_store/overnight_hourly.csv. New York clock times are
converted with zoneinfo, so daylight saving is exact; exchange holidays are ignored (every Monday-Friday counts as a
trading day). A position decided at a clock time uses only the price at that time. Public, key-less endpoints only.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import price as px

SHARED_PATH = common.STORE / "seasonality_hourly.csv"  # kept by studies2.seasonality; read only
OWN_PATH = common.STORE / "overnight_hourly.csv"  # only the hours the shared store does not have
HOUR_MS = 3_600_000
FIRST_MS = 1_502_928_000_000  # 2017-08-17 00:00 UTC, the day Binance listed BTCUSDT
MAX_PAGES = 130
MIN_HOURS = 50_000
YEAR = 365
NY = ZoneInfo("America/New_York")
TW = ZoneInfo("Asia/Taipei")
CLOSE_HOUR, OPEN_HOUR, ALT_OPEN_HOUR = 16, 10, 9  # New York clock hours
HIGH_DAYS = 10
PUBLISHED = (4, 0, 1)  # Friday->Monday, Monday->Tuesday, Tuesday->Wednesday (weekday of the close)
UTC_HOURS = (21, 22)  # the hours starting at 21:00 and 22:00 UTC = 21:00 to 23:00
VOL_HOURS = 30 * 24
MEDIAN_DAYS = 365
COST = 0.001
COSTS = (0.0, 0.0005, 0.001, 0.0025)
FIRST_PICK_YEAR = 2021
TREND_IN, TREND_OUT = 20, 10

OOS_START = "2021-10-18"  # ProShares BITO started trading; the page's in-sample / out-of-sample split
OOS_END = "2024-10-16"  # exclusive: the page's data end on 2024-10-15
NIGHT_PUBLISHED = "2024-11-12"
HOURS_SAMPLE_END = "2023-07-01"  # exclusive: the page's data end on 2023-06-30
HOURS_PUBLISHED = "2023-09-13"

SLOTS = ("週一收盤→週二開盤", "週二收盤→週三開盤", "週三收盤→週四開盤", "週四收盤→週五開盤", "週五收盤→下週一開盤")
SLOT_SHORT = ("一", "二", "三", "四", "五")
NAMES = {
    "hold": "一直抱著不動",
    "night": "每個隔夜都持有（不看條件）",
    "day": "每個美股白天時段都持有（開盤→收盤）",
    "weekend": "只在週末持有（週五收盤→週一開盤）",
    "weeknight": "只在週一到週四的隔夜持有",
    "o10": "創 10 日新高才持有隔夜（五個晚上都做）",
    "o3": "創 10 日新高才持有隔夜（原文最後版本：只做週五、週一、週二收盤後）",
    "o10_no": "沒創 10 日新高的隔夜才持有",
    "o10_next": "創 10 日新高後，改成只持有下一個美股白天時段",
    "d10": "創 10 日新高就持有 24 小時（收盤→隔天同一時間，七天都看）",
    "h2": "每天只在 UTC 21–23 點持有（台灣早上 5–7 點）",
    "h2v": "UTC 21–23 點，只在波動偏高的日子做",
    "trend": "順勢規則：創 20 日收盤新高持有、破 10 日收盤新低出場",
}
PICK_FROM = ("night", "o10", "o3", "h2", "h2v")
PICK_SHORT = {"night": "每個隔夜", "o10": "創新高的隔夜", "o3": "創新高＋三個晚上", "h2": "UTC 21–23 點",
              "h2v": "UTC 21–23 點＋波動偏高"}


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, digits: int = 1, signed: bool = True) -> str:
    """v is already in percent."""
    if not _ok(v):
        return "—"
    v = float(v)
    if abs(v) < 0.5 * 10 ** -digits:
        return f"{0:.{digits}f}%"
    if abs(v) >= 100:
        digits = 0
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _times(v) -> str:
    if not _ok(v):
        return "—"
    return f"{v:,.0f} 次" if v >= 10 else f"{v:.1f} 次"


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _clock(ms: int) -> str:
    """One moment as New York and Taiwan clock time."""
    at = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return f"紐約 {at.astimezone(NY).strftime('%m-%d %H:%M')}＝台灣 {at.astimezone(TW).strftime('%m-%d %H:%M')}"


# ---------- data ----------

def _read(path) -> dict[int, tuple[float, float]]:
    rows: dict[int, tuple[float, float]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            o, c = float(parts[1]), float(parts[2])
            if o > 0 and c > 0 and math.isfinite(o) and math.isfinite(c):
                rows[int(parts[0])] = (o, c)
    except Exception:  # noqa: BLE001 - no such file or a damaged one: nothing from it
        return {}
    return rows


def _write_own(rows: dict[int, tuple[float, float]], shared: dict) -> None:
    """Keep only the hours the shared store does not have; the shared store itself is never touched."""
    try:
        extra = {ms: v for ms, v in rows.items() if ms not in shared}
        common.STORE.mkdir(parents=True, exist_ok=True)
        text = "ms,open,close\n" + "\n".join(f"{ms},{o!r},{c!r}" for ms, (o, c) in sorted(extra.items())) + "\n"
        OWN_PATH.write_text(text, encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run simply fetches again
        pass


def _hourly() -> tuple[dict[int, tuple[float, float]], str | None]:
    """Closed BTCUSDT 1-hour candles {open time ms: (open, close)} and a problem text if the fetch failed."""
    shared = _read(SHARED_PATH)
    rows = dict(shared)
    rows.update(_read(OWN_PATH))
    start = max(rows) if rows else FIRST_MS  # the last stored hour is fetched again
    problem, added = None, 0
    try:
        for page_no in range(MAX_PAGES):
            page = px._binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 1000,
                                                  "startTime": start})
            if not page:
                break
            now_ms = int(time.time() * 1000)
            for r in page:
                if int(r[6]) < now_ms and float(r[1]) > 0 and float(r[4]) > 0:  # closed candles only
                    if int(r[0]) not in rows:
                        added += 1
                    rows[int(r[0])] = (float(r[1]), float(r[4]))
            if len(page) < 1000:
                break
            start = int(page[-1][0]) + 1
            if page_no % 20 == 19:
                _write_own(rows, shared)
    except Exception as exc:  # noqa: BLE001 - a stored history is still usable
        problem = f"幣安 1 小時線這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料，沒有改用其他方法繞過。"
    if added:
        _write_own(rows, shared)
    if len(rows) < MIN_HOURS:
        raise RuntimeError(f"幣安 1 小時線只有 {len(rows)} 筆")
    return rows, problem


def _grid(rows: dict[int, tuple[float, float]]) -> tuple[int, np.ndarray, int]:
    """(first ms, price at every full hour from the first to one past the last candle, number of missing candles).

    The price at an hour is that candle's open; where the candle is missing it is the previous candle's close, else
    the last known price (so a gap shows no move until trading is back)."""
    ms0, ms1 = min(rows), max(rows)
    n = (ms1 - ms0) // HOUR_MS + 1
    opens, closes = np.full(n, np.nan), np.full(n, np.nan)
    for ms, (o, c) in rows.items():
        i = (ms - ms0) // HOUR_MS
        opens[i], closes[i] = o, c
    price = np.concatenate([opens, [np.nan]])
    previous = np.concatenate([[np.nan], closes])
    price = np.where(np.isfinite(price), price, previous)
    return ms0, pd.Series(price).ffill().bfill().values, int(np.isnan(opens).sum())


# ---------- measuring ----------

def _eval(pos: np.ndarray, lr: np.ndarray, a: int, b: int, cost: float = 0.0) -> dict | None:
    """Hold bitcoin during the hours a..b-1 where pos is 1. Starts and ends in cash; every change is one trade."""
    if b - a < 24 * 20:
        return None
    p = pos[a:b].astype(float)
    gross = lr[a:b] * p
    edge = np.diff(np.concatenate([[0.0], p, [0.0]]))
    entry, leave = edge[:-1] > 0, edge[1:] < 0  # leave is marked on the last hour held
    trades = int(entry.sum() + leave.sum())
    fee = math.log(1 - cost) if cost > 0 else 0.0
    net = gross + fee * (entry.astype(float) + leave.astype(float))
    path = np.concatenate([[0.0], np.cumsum(net)])
    deepest = float((np.exp(path - np.maximum.accumulate(path)) - 1).min() * 100)
    days = (b - a) / 24
    total, raw = float(net.sum()), float(gross.sum())
    whole = (b - a) // 24
    daily = np.expm1(net[:whole * 24].reshape(whole, 24).sum(axis=1))
    vol = float(np.std(daily, ddof=1) * math.sqrt(YEAR) * 100)
    ret = float(math.expm1(total * YEAR / days) * 100)
    edges = np.concatenate([[0.0], np.cumsum(gross)])
    holds = edges[np.flatnonzero(leave) + 1] - edges[np.flatnonzero(entry)]
    return {"ret": ret, "total": float(math.expm1(total) * 100), "deepest": deepest, "vol": vol,
            "ratio": ret / vol if vol > 1e-9 else 0.0, "trades": trades * YEAR / days, "n_trades": trades,
            "breakeven": (1 - math.exp(-raw / trades)) * 100 if trades and raw > 0 else None,
            "share": float(p.mean() * 100), "holds": int(len(holds)),
            "win": float((holds > 0).mean() * 100) if len(holds) else None,
            "mean": float(np.expm1(holds).mean() * 100) if len(holds) else None, "log": raw, "days": days}


def _compute(rows: dict[int, tuple[float, float]]) -> dict:
    """Everything that depends on the data; study() only formats it."""
    ms0, price, missing = _grid(rows)
    n = len(price) - 1  # hours
    lr = np.diff(np.log(price))
    end_ms = ms0 + n * HOUR_MS

    def at(ms: int) -> int:
        return int((ms - ms0) // HOUR_MS)

    def cut(day: str) -> int:
        return min(max(at(_ms(day)), 0), n)

    def fill(spans) -> np.ndarray:
        pos = np.zeros(n)
        for a, b in spans:
            if 0 <= a < b <= n:
                pos[a:b] = 1.0
        return pos

    # ----- New York calendar: the 16:00 close and the 10:00 (and 09:00) open of every calendar day -----
    first = datetime.fromtimestamp(ms0 / 1000, tz=NY).date() - timedelta(days=1)
    last = datetime.fromtimestamp(end_ms / 1000, tz=NY).date() + timedelta(days=4)
    days = [first + timedelta(days=k) for k in range((last - first).days + 1)]

    def hour_index(d: date, hour: int) -> int:
        return at(int(datetime(d.year, d.month, d.day, hour, tzinfo=NY).timestamp() * 1000))

    ci = np.array([hour_index(d, CLOSE_HOUR) for d in days])
    oi = np.array([hour_index(d, OPEN_HOUR) for d in days])
    oi_alt = np.array([hour_index(d, ALT_OPEN_HOUR) for d in days])
    wd = np.array([d.weekday() for d in days])
    known = (ci >= 0) & (ci <= n)
    at_close = pd.Series(np.where(known, price[np.clip(ci, 0, n)], np.nan))
    high = at_close.rolling(HIGH_DAYS, min_periods=HIGH_DAYS).max()
    cond = (at_close >= high).fillna(False).values  # today's 16:00 price is the highest of the last 10 days' 16:00s
    weekday_close = at_close.where(wd < 5).dropna()
    high_alt = weekday_close.rolling(HIGH_DAYS, min_periods=HIGH_DAYS).max()
    cond_alt = (weekday_close >= high_alt).reindex(at_close.index).fillna(False).values  # 10 exchange days instead

    nights = {k: [] for k in range(5)}  # by weekday of the close: (day number, close index, next open index, ...)
    for j, d in enumerate(days):
        if wd[j] >= 5:
            continue
        nxt = j + (3 if wd[j] == 4 else 1)
        if nxt < len(days):
            nights[int(wd[j])].append((j, int(ci[j]), int(oi[nxt]), int(oi_alt[nxt]), int(ci[nxt])))

    def night_pos(slots, flag=None, want: bool = True, alt_open: bool = False) -> np.ndarray:
        return fill((c, (o9 if alt_open else o)) for k in slots for j, c, o, o9, _ in nights[k]
                    if flag is None or bool(flag[j]) == want)

    all_slots = range(5)
    positions = {
        "hold": np.ones(n),
        "night": night_pos(all_slots),
        "day": fill((int(oi[j]), int(ci[j])) for j in range(len(days)) if wd[j] < 5),
        "weekend": night_pos([4]),
        "weeknight": night_pos([0, 1, 2, 3]),
        "o10": night_pos(all_slots, cond),
        "o3": night_pos(PUBLISHED, cond),
        "o10_no": night_pos(all_slots, cond, want=False),
        "o10_next": fill((o, c2) for k in all_slots for j, c, o, _, c2 in nights[k] if cond[j]),
        "d10": fill((int(ci[j]), int(ci[j + 1])) for j in range(len(days) - 1) if cond[j]),
    }
    slot_pos = [night_pos([k], cond) for k in range(5)]

    # ----- UTC days: the 21:00-23:00 hold, its volatility filter, and the dashboard's plain trend rule -----
    first_mid = (-(-ms0 // (24 * HOUR_MS))) * 24 * HOUR_MS
    mids = np.arange(at(first_mid), n + 1, 24)  # index of every UTC midnight inside the data
    rolling_vol = pd.Series(lr).rolling(VOL_HOURS, min_periods=VOL_HOURS).std(ddof=1)
    vol_at = pd.Series([rolling_vol.iloc[m - 1] if m >= 1 else np.nan for m in mids])  # uses hours before midnight
    median = vol_at.rolling(MEDIAN_DAYS, min_periods=MEDIAN_DAYS).median()
    high_vol = (vol_at > median).where(median.notna())
    valid = np.flatnonzero(median.notna().values)
    vol_start = int(mids[valid[0]]) if len(valid) else n
    positions["h2"] = fill((int(m) + UTC_HOURS[0], int(m) + UTC_HOURS[-1] + 1) for m in mids)
    positions["h2v"] = fill((int(m) + UTC_HOURS[0], int(m) + UTC_HOURS[-1] + 1)
                            for m, flag in zip(mids, high_vol.values) if flag is True or flag == 1.0)

    closes = pd.Series(price[mids[1:]])  # close of each finished UTC day
    top = closes.rolling(TREND_IN, min_periods=TREND_IN).max()
    low = closes.rolling(TREND_OUT, min_periods=TREND_OUT).min()
    state = pd.Series(np.where(closes >= top, 1.0, np.where(closes <= low, 0.0, np.nan)))
    state = state.where(top.notna()).ffill().fillna(0.0).values  # cash until a first 20-day high
    positions["trend"] = fill((int(mids[k + 1]), min(int(mids[k + 1]) + 24, n)) for k in range(len(state))
                              if state[k] == 1.0)

    starts = {key: 0 for key in positions}
    starts["h2v"] = vol_start

    def measure(key: str, a: int, b: int, cost: float = 0.0, pos: np.ndarray | None = None) -> dict | None:
        return _eval(positions[key] if pos is None else pos, lr, max(a, starts.get(key, 0)), b, cost)

    # ----- periods -----
    spans = {
        "whole": (0, n),
        "night_in": (0, cut(OOS_START)), "night_oos": (cut(OOS_START), cut(OOS_END)),
        "night_after": (cut(NIGHT_PUBLISHED), n),
        "hours_in": (0, cut(HOURS_SAMPLE_END)), "hours_after": (cut(HOURS_PUBLISHED), n),
    }
    stats = {}
    for key in positions:
        stats[key] = {}
        for name, (a, b) in spans.items():
            a = max(a, starts[key])
            stats[key][name] = {"gross": measure(key, a, b), "net": measure(key, a, b, COST),
                                "hold": _eval(positions["hold"], lr, a, b), "a": a, "b": b}
    costs = {key: {c: measure(key, 0, n, c) for c in COSTS} for key in positions}
    hold_from_vol = _eval(positions["hold"], lr, vol_start, n)

    # ----- other ways to read "10-day high" and "open" (out-of-sample period, no costs) -----
    a, b = spans["night_oos"]
    alternatives = {
        "open9": _eval(night_pos(PUBLISHED, cond, alt_open=True), lr, a, b),
        "exchange_days": _eval(night_pos(PUBLISHED, cond_alt), lr, a, b),
        "open9_all": _eval(night_pos(all_slots, cond, alt_open=True), lr, a, b),
        "exchange_days_all": _eval(night_pos(all_slots, cond_alt), lr, a, b),
    }

    # ----- pick in advance -----
    last_year = datetime.fromtimestamp((end_ms - 1) / 1000, tz=timezone.utc).year
    test_years = list(range(FIRST_PICK_YEAR, last_year + 1))
    chained = np.zeros(n)
    picks, year_rows = {}, {}
    for y in test_years + [last_year + 1]:
        ya, yb = cut(f"{y}-01-01"), cut(f"{y + 1}-01-01")
        scores = {}
        for key in PICK_FROM:
            m = measure(key, 0, ya, COST)
            scores[key] = m["ratio"] if m else -1e9
        choice = max(PICK_FROM, key=lambda k: scores[k])
        picks[y] = {"choice": choice, "score": scores[choice], "scores": scores}
        if y > last_year:
            continue
        chained[ya:yb] = positions[choice][ya:yb]
        each = {key: _eval(positions[key], lr, ya, yb, COST) for key in PICK_FROM}
        mine = each[choice]
        year_rows[y] = {
            "pick": mine, "hold": _eval(positions["hold"], lr, ya, yb),
            "trend": _eval(positions["trend"], lr, ya, yb, COST),
            "rank": 1 + sum(1 for key in PICK_FROM if each[key] and mine and each[key]["total"] > mine["total"]),
            "best": max((k for k in PICK_FROM if each[k]), key=lambda k: each[k]["total"], default=None),
            "each": each,
        }
    ta = cut(f"{FIRST_PICK_YEAR}-01-01")
    pick_total = {"pick": _eval(chained, lr, ta, n, COST), "hold": _eval(positions["hold"], lr, ta, n),
                  "trend": _eval(positions["trend"], lr, ta, n, COST),
                  "each": {key: _eval(positions[key], lr, ta, n, COST) for key in PICK_FROM}}

    # ----- luck: where does the published choice of nights rank among all 31 choices? -----
    subsets = [s for size in range(1, 6) for s in itertools.combinations(range(5), size)]
    luck = {}
    for name in ("night_in", "night_oos", "night_after"):
        a, b = spans[name]
        days_in = (b - a) / 24
        if days_in < 20:
            continue
        slot_log = [float((lr[a:b] * slot_pos[k][a:b]).sum()) for k in range(5)]
        yearly = {s: math.expm1(sum(slot_log[k] for k in s) * YEAR / days_in) * 100 for s in subsets}
        mine = yearly[tuple(sorted(PUBLISHED))]
        best = max(subsets, key=lambda s: yearly[s])
        luck[name] = {"rank": 1 + sum(1 for s in subsets if yearly[s] > mine + 1e-12), "mine": mine, "best": best,
                      "best_ret": yearly[best], "median": float(np.median(list(yearly.values()))),
                      "slots": [math.expm1(v * YEAR / days_in) * 100 for v in slot_log], "a": a, "b": b}

    # ----- today -----
    usable = [j for j in range(len(days)) if wd[j] < 5 and 0 <= ci[j] <= n and np.isfinite(high.iloc[j])]
    j = usable[-1]
    nxt = j + (3 if wd[j] == 4 else 1)
    today = {
        "close_ms": ms0 + int(ci[j]) * HOUR_MS, "close": float(at_close.iloc[j]), "high": float(high.iloc[j]),
        "before": float(at_close.iloc[j - HIGH_DAYS + 1:j].max()), "met": bool(cond[j]), "slot": int(wd[j]),
        "open_ms": ms0 + int(oi[nxt]) * HOUR_MS, "running": bool(cond[j]) and int(oi[nxt]) > n,
        "vol": float(vol_at.iloc[-1] * math.sqrt(24 * YEAR) * 100) if _ok(vol_at.iloc[-1]) else None,
        "median": float(median.iloc[-1] * math.sqrt(24 * YEAR) * 100) if _ok(median.iloc[-1]) else None,
        "high_vol": bool(high_vol.iloc[-1] == 1.0), "trend": bool(len(state) and state[-1] == 1.0),
        "trend_day_ms": ms0 + int(mids[-1]) * HOUR_MS - 24 * HOUR_MS,
        "met_share": float(np.mean([cond[k] for k in usable]) * 100),
    }

    return {"ms0": ms0, "end_ms": end_ms, "n": n, "missing": missing, "last_price": float(price[-1]),
            "stats": stats, "costs": costs, "spans": spans, "vol_start_ms": ms0 + vol_start * HOUR_MS,
            "hold_from_vol": hold_from_vol, "alternatives": alternatives, "test_years": test_years, "picks": picks,
            "year_rows": year_rows, "pick_total": pick_total, "luck": luck, "today": today, "last_year": last_year}


# ---------- the study ----------

def study() -> dict:
    rows, problem = _hourly()
    r = _compute(rows)
    stats, costs, today = r["stats"], r["costs"], r["today"]
    ms0, end_ms = r["ms0"], r["end_ms"]
    first_day, last_day = _day(ms0), _day(end_ms - 1)

    def span_text(key: str, name: str) -> str:
        s = stats[key][name]
        return f"{_day(ms0 + s['a'] * HOUR_MS)} 到 {_day(ms0 + s['b'] * HOUR_MS - 1)}"

    def g(key: str, name: str, field: str = "ret", kind: str = "gross"):
        m = stats[key][name][kind]
        return m[field] if m else None

    # ----- table 1: the strategies as published, by period -----
    said_none = "原文之後的日子"
    plan = [
        ("o10", "night_in", "樣本內", "只有圖，文字沒給數字（原文從 2015-10 起）"),
        ("o10", "night_oos", "樣本外", "只有圖，文字沒給數字"),
        ("o10", "night_after", "發表之後", said_none),
        ("o3", "night_in", "樣本內", "只有圖，文字沒給數字（原文從 2015-10 起）"),
        ("o3", "night_oos", "樣本外", "「超過 35%、最深虧 −12%」，沒寫是每年還是整段"),
        ("o3", "night_after", "發表之後", said_none),
        ("h2", "hours_in", "原文樣本", "每年 +40.64%、最深虧 −22.7%（原文從 2015-10 起）"),
        ("h2", "hours_after", "發表之後", said_none),
        ("h2v", "hours_in", "原文樣本", "每年 +37.26%、最深虧 −18.87%（原文從 2015-10 起）"),
        ("h2v", "hours_after", "發表之後", said_none),
    ]
    short = {"o10": "創 10 日新高的隔夜（五晚）", "o3": "創 10 日新高的隔夜（原文最後版本，三晚）",
             "h2": "UTC 21–23 點（台灣早上 5–7 點）", "h2v": "UTC 21–23 點＋波動偏高"}
    period_rows = []
    for key, name, label, said in plan:
        if stats[key][name]["gross"] is None:
            continue
        period_rows.append([short[key], f"{label}：{span_text(key, name)}", _pct(g(key, name)),
                            _pct(g(key, name, "total")), _pct(g(key, name, kind="net")),
                            _pct(g(key, name, "deepest")), _pct(g(key, name, kind="hold")), said])

    # ----- table 2: the pieces separated (whole Binance period) -----
    piece_keys = ("hold", "night", "day", "weekend", "weeknight", "o10", "o10_no", "o10_next", "d10", "o3", "trend")
    piece_rows = []
    for key in piece_keys:
        m, net = stats[key]["whole"]["gross"], stats[key]["whole"]["net"]
        piece_rows.append([NAMES[key], _pct(m["ret"]), _pct(net["ret"]) if key != "hold" else _pct(m["ret"]),
                           _pct(m["deepest"]), f"{m['share']:.0f}%",
                           "—" if key == "hold" else _times(m["holds"] * YEAR / m["days"]).replace("次", "趟"),
                           "—" if key == "hold" else _pct(m["mean"], 2),
                           "—" if key == "hold" else _pct(m["win"], 0, signed=False)])

    # ----- table 3: costs (whole Binance period) -----
    def breakeven(m: dict) -> str:
        return f"{m['breakeven']:.2f}%" if _ok(m["breakeven"]) else "不扣成本就沒有賺"

    cost_keys = ("night", "o10", "o3", "d10", "h2", "h2v", "trend")
    cost_rows = [[NAMES[key]] + [_pct(costs[key][c]["ret"]) for c in COSTS]
                 + [breakeven(costs[key][0.0]), _times(costs[key][0.0]["trades"])] for key in cost_keys]

    # ----- table 4: pick in advance, then the luck check -----
    test_years, picks, year_rows, total = r["test_years"], r["picks"], r["year_rows"], r["pick_total"]
    nt = len(test_years)
    partial = last_day < f"{r['last_year']}-12-31"
    pick_rows = []
    beat_hold = beat_trend = 0
    for y in test_years:
        x = year_rows[y]
        if x["pick"] is None:
            continue
        beat_hold += x["pick"]["total"] > x["hold"]["total"]
        beat_trend += x["pick"]["total"] > x["trend"]["total"]
        label = f"{y} 年（到 {last_day[5:]}）" if y == r["last_year"] and partial else f"{y} 年"
        pick_rows.append([label, f"{PICK_SHORT[picks[y]['choice']]}（之前的報酬÷波動 {_num(picks[y]['score'])}）",
                          _pct(x["pick"]["total"]), _pct(x["hold"]["total"]), _pct(x["trend"]["total"]),
                          f"當年在五個裡排第 {x['rank']}，最好的是「{PICK_SHORT[x['best']]}」"])
    pick_rows.append([f"{test_years[0]}–{test_years[-1]} 接起來，換算每年", "每年年初重挑一次",
                      _pct(total["pick"]["ret"]), _pct(total["hold"]["ret"]), _pct(total["trend"]["ret"]),
                      f"{nt} 年裡 {beat_hold} 年贏過一直抱著、{beat_trend} 年贏過順勢規則"])
    pick_rows.append(["同一段的最深虧損", "—", _pct(total["pick"]["deepest"]), _pct(total["hold"]["deepest"]),
                      _pct(total["trend"]["deepest"]), "扣成本後"])
    luck = r["luck"]
    luck_names = {"night_in": "運氣檢查：樣本內", "night_oos": "運氣檢查：樣本外", "night_after": "運氣檢查：發表之後"}

    def slot_list(s) -> str:
        return "週" + "、".join(SLOT_SHORT[k] for k in s) + "收盤後"

    for name in ("night_in", "night_oos", "night_after"):
        if name not in luck:
            continue
        x = luck[name]
        pick_rows.append([f"{luck_names[name]}（{span_text('o10', name)}）", f"31 種裡排第 {x['rank']}",
                          _pct(x["mine"]), f"{slot_list(x['best'])}：{_pct(x['best_ret'])}", _pct(x["median"]),
                          "五個晚上各自每年：" + "、".join(f"{SLOT_SHORT[k]} {_pct(v)}" for k, v in enumerate(x["slots"]))])

    whole_span = f"{first_day} 到 {last_day}"
    tables = [
        {"caption": ("兩篇文章的做法照原文重做（幣安 1 小時線，不是原文用的 Gemini；幣安從 2017-08 才有，原文樣本從 2015-10 開始）。"
                     "隔夜做法：紐約時間 16:00（台灣隔天凌晨 4 點，冬令時間 5 點）的價格是最近 10 天同一時刻裡最高的，就從那時抱到"
                     "下一個交易日紐約 10:00（台灣晚上 10 點，冬令時間 11 點），否則抱現金。樣本內、樣本外照原文以 2021-10-18 "
                     f"（第一檔美國比特幣期貨 ETF 上市）為界，樣本外到原文資料結束的 2024-10-15；發表日是 {NIGHT_PUBLISHED}。"
                     f"UTC 21–23 點的做法：每天台灣早上 5 點買、7 點賣；原文樣本到 2023-06-30、發表日是 {HOURS_PUBLISHED}；"
                     f"波動偏高的版本要先累積一年多的資料，所以從 {_day(r['vol_start_ms'])} 才開始。"
                     "「最深虧損」是不扣成本、用每小時的資金變化算的。"),
         "columns": ["做法", "期間", "每年報酬（不扣成本）", "整段總報酬（不扣成本）", f"每年報酬（每次買賣扣 {COST * 100:.2f}%）",
                     "最深曾經虧多少", "同期一直抱著的每年報酬", "原文怎麼說"],
         "rows": period_rows},
        {"caption": (f"把做法拆開來看（{whole_span}，整段幣安資料）。「隔夜」是紐約 16:00 到下一個交易日 10:00，「美股白天時段」是"
                     "紐約 10:00 到 16:00，週一到週五都算交易日。「創 10 日新高後，改成只持有下一個美股白天時段」是同樣的條件、"
                     "但不抱隔夜，改抱緊接著的那個白天；「創 10 日新高就持有 24 小時」是每天（含週末）紐約 16:00 看一次，"
                     "成立就抱到隔天 16:00，連續成立就一直抱著不重複買賣。最後一列是儀表板已經檢驗過的順勢規則，用 UTC 日線。"),
         "columns": ["做法", "每年報酬（不扣成本）", f"每年報酬（每次買賣扣 {COST * 100:.2f}%）", "最深曾經虧多少（不扣成本）",
                     "持有比特幣的時間比例", "每年持有幾趟", "平均每趟賺多少", "賺錢的趟數比例"],
         "rows": piece_rows},
        {"caption": (f"成本的影響（{whole_span}）。成本是每次買或賣各扣交易金額的這個比例；抱一個隔夜要買一次、賣一次，"
                     "所以每趟扣兩次。「成本到多少就把報酬吃光」是每次買賣的成本高到這個數字時，整段剛好不賺不賠。"
                     f"波動偏高的版本從 {_day(r['vol_start_ms'])} 起算。"),
         "columns": ["做法"] + [("不扣成本" if c == 0 else f"每次扣 {c * 100:.2f}%") for c in COSTS]
                    + ["成本到多少就把報酬吃光", "每年買賣幾次"],
         "rows": cost_rows},
        {"caption": (f"上半：事先挑的測試。每年 1 月 1 日只用那天以前的幣安資料，在五個做法裡挑「每次買賣扣 {COST * 100:.2f}% 之後，"
                     "每年報酬 ÷ 每年波動」最高的一個，當年照做（一樣扣成本），再把每年接起來；和一直抱著、順勢規則"
                     f"（創 {TREND_IN} 日收盤新高持有、破 {TREND_OUT} 日收盤新低出場，同樣扣成本）比。這幾列第三到五欄是當年的報酬。"
                     "下半：運氣檢查。五個隔夜時段任選一個以上共有 31 種選法，都套上「創 10 日新高才持有」，照每年報酬（不扣成本）"
                     "排名，看原文選的「週五、週一、週二收盤後」排第幾；這幾列第三欄是原文選法的每年報酬，"
                     "第四欄是 31 種裡最好的選法，第五欄是 31 種的中間值。"),
         "columns": ["項目", "事先挑到的做法｜原文選法的排名", f"照做的結果（扣 {COST * 100:.2f}%）｜原文選法", "一直抱著｜31 種裡最好的",
                     "順勢規則｜31 種的中間值", "補充"],
         "rows": pick_rows},
    ]

    # ----- today -----
    def yes(flag: bool) -> str:
        return "成立" if flag else "不成立"

    in_final = today["slot"] in PUBLISHED
    now = [
        {"label": f"資料到（{_clock(end_ms)}）", "value": f"{r['last_price']:,.0f} 美元"},
        {"label": f"最近一個紐約 16:00 收盤（{_clock(today['close_ms'])}）的價格",
         "value": f"{today['close']:,.0f} 美元；之前 9 天同一時刻最高是 {today['before']:,.0f} 美元"},
        {"label": "「創 10 日新高」條件（做法的狀態）",
         "value": (f"{yes(today['met'])}——那一晚是「{SLOTS[today['slot']]}」，"
                   + ("在" if in_final else "不在") + "原文最後版本做的三個晚上裡"
                   + (f"；照規則持有到 {_clock(today['open_ms'])}" if today["running"] else ""))},
        {"label": "過去每個交易日收盤時這個條件成立的比例", "value": _pct(today["met_share"], 0, signed=False)},
        {"label": "波動條件：最近 30 天的每小時波動（換算一年）對一年來的中間值",
         "value": (f"{_pct(today['vol'], signed=False)} 對 {_pct(today['median'], signed=False)}，"
                   f"「波動偏高」{yes(today['high_vol'])}")},
        {"label": f"順勢規則（{_day(today['trend_day_ms'])} UTC 日線收盤後）",
         "value": "持有比特幣" if today["trend"] else "現金"},
        {"label": "這幾格的意思", "value": "只是這些做法照規則目前停在哪個狀態，不代表接下來會漲或跌"},
    ]

    # ----- verdict -----
    whole = {key: stats[key]["whole"] for key in stats}
    o3_oos, o3_oos_net = stats["o3"]["night_oos"]["gross"], stats["o3"]["night_oos"]["net"]
    o3_after, o3_after_net = stats["o3"]["night_after"]["gross"], stats["o3"]["night_after"]["net"]
    o10_after = stats["o10"]["night_after"]["gross"]
    h_in, h_after = stats["h2"]["hours_in"], stats["h2"]["hours_after"]
    hv_after = stats["h2v"]["hours_after"]
    night, o10, o10_no, o10_next, d10 = (whole[k]["gross"] for k in ("night", "o10", "o10_no", "o10_next", "d10"))
    hold_all, trend_all = whole["hold"]["gross"], whole["trend"]["gross"]
    source = (f"每個隔夜都抱（占 {night['share']:.0f}% 的時間）是每年 {_pct(night['ret'])}、最深虧 {_pct(night['deepest'])}，"
              f"和一直抱著（{_pct(hold_all['ret'])}、{_pct(hold_all['deepest'])}）差不多；其中創 10 日新高之後的隔夜是 "
              f"{_pct(o10['ret'])}、沒創新高的隔夜是 {_pct(o10_no['ret'])}，同樣的條件改成抱 24 小時是 {_pct(d10['ret'])}"
              f"（只抱隔天白天是 {_pct(o10_next['ret'])}），儀表板的順勢規則是 {_pct(trend_all['ret'])}")
    if o10["ret"] > 0 and o10_no["ret"] < o10["ret"] / 4 and d10["ret"] > o10["ret"] / 2:
        source += "——報酬幾乎都來自「創新高才持有」這個順勢條件，「隔夜」只是把沒什麼貢獻的白天幾小時拿掉"
    elif o10["ret"] > 0 and o10_no["ret"] >= o10["ret"] / 4:
        source += "——創新高和沒創新高的隔夜都有貢獻，不能只歸給順勢條件"
    verdict = (
        f"「創 10 日新高才抱隔夜」的最後版本，用幣安資料重做原文的樣本外期間（{OOS_START} 到 2024-10-15）是每年 "
        f"{_pct(o3_oos['ret'])}（整段 {_pct(o3_oos['total'])}）、最深虧 {_pct(o3_oos['deepest'])}，原文寫的是「超過 35%、最深虧 −12%」；"
        f"發表之後（{NIGHT_PUBLISHED} 起）是每年 {_pct(o3_after['ret'])}、最深虧 {_pct(o3_after['deepest'])}"
        f"（五個晚上都做是 {_pct(o10_after['ret'])}），每次買賣扣 {COST * 100:.2f}% 之後，樣本外剩每年 {_pct(o3_oos_net['ret'])}、"
        f"發表之後剩 {_pct(o3_after_net['ret'])}。"
        f"「每天只在 UTC 21–23 點（台灣早上 5–7 點）持有」在原文樣本（幣安只有 2017-08 起）是每年 {_pct(h_in['gross']['ret'])}"
        f"（原文 +40.64%），發表之後（{HOURS_PUBLISHED} 起）是 {_pct(h_after['gross']['ret'])}；一年要買賣約 "
        f"{h_in['gross']['trades']:,.0f} 次，每次扣 {COST * 100:.2f}% 就變成 {_pct(h_in['net']['ret'])} 和 {_pct(h_after['net']['ret'])}"
        f"（加上波動條件的版本發表之後是 {_pct(hv_after['gross']['ret'])}、扣成本後 {_pct(hv_after['net']['ret'])}）。"
        f"拆開來看（{first_day[:7]} 起整段、不扣成本）：{source}；每年年初事先挑一個做法照做，{test_years[0]}–{test_years[-1]} "
        f"扣成本後每年 {_pct(total['pick']['ret'])}，一直抱著是 {_pct(total['hold']['ret'])}、順勢規則是 "
        f"{_pct(total['trend']['ret'])}，{nt} 年裡 {beat_hold} 年贏過一直抱著。"
    )

    # ----- notes -----
    alt = r["alternatives"]
    pick_list = "、".join(f"{y} 年「{PICK_SHORT[picks[y]['choice']]}」" for y in test_years)
    negative = [y for y in test_years if picks[y]["score"] <= 0]
    next_pick = picks[r["last_year"] + 1]
    li = luck.get("night_in")
    notes = [
        "原文和這裡的差別：兩篇原文都用 Gemini 交易所的 1 小時線、從 2015-10 開始；這裡用幣安 BTCUSDT，2017-08 才有資料，"
        "所以原文樣本最前面將近兩年（比特幣漲得很多的一段）重做不到，兩家交易所同一時刻的價格也會差一點，數字不會一模一樣。"
        "隔夜那篇的內文只給了一組數字（樣本外「超過 35%、最深虧 −12%」），沒有寫是每年還是整段、也沒有明說是哪個版本，"
        "其他數字都在圖裡；所以表裡同時列出每年和整段的報酬。",
        "原文把美股白天定義成紐約時間 10:00 到 16:00（不是實際開盤的 09:30），這裡照原文用 10:00。1 小時線只能對到整點，"
        "實際的開盤、收盤那一分鐘的價格會有出入；改用 09:00 當開盤，最後版本的樣本外每年報酬是 "
        f"{_pct(alt['open9']['ret'] if alt['open9'] else None)}（10:00 是 {_pct(o3_oos['ret'])}）。紐約有日光節約時間，"
        "所以 16:00 在夏令是台灣隔天凌晨 4 點、冬令是凌晨 5 點，10:00 是台灣晚上 10 點或 11 點，程式是照紐約當地時間換算的。",
        "美股休市日沒有處理：週一到週五一律當成交易日，元旦、感恩節、聖誕節這類休市日照樣算一個白天和一個隔夜，"
        "一年大約 9 到 10 天，和原文會有小出入。",
        "「創 10 日新高」原文沒有寫細節，這裡的定義是：紐約 16:00 的價格，是最近 10 個日曆天（含週末、含當天）每天 16:00 價格裡最高的。"
        f"改成只看最近 10 個交易日，最後版本的樣本外每年報酬是 {_pct(alt['exchange_days']['ret'] if alt['exchange_days'] else None)}"
        f"、五個晚上都做是 {_pct(alt['exchange_days_all']['ret'] if alt['exchange_days_all'] else None)}"
        f"（原定義分別是 {_pct(o3_oos['ret'])} 和 {_pct(g('o10', 'night_oos'))}）。",
        "原文作者試過很多版本：新高的天數試了 5、10、20、30、40、50 天，再把一週拆成每個白天和每個晚上，最後才留下三個晚上；"
        "而且選哪三個晚上看起來用的是整段資料（包含他們稱為樣本外的那段），所以樣本外的數字並不是真正事後才看到的檢驗。"
        "試得越多，留下來的那個光靠運氣就會好看，要看的是第四張表的排名和發表之後的結果"
        + (f"（樣本內這個選法在 31 種裡排第 {li['rank']}）。" if li else "。"),
        f"成本：表裡用每次買賣 0%、0.05%、0.10%、0.25%。一般人實際付的手續費加買賣價差多半比 {COST * 100:.2f}% 高，"
        "一天內短進短出尤其如此；每天只抱兩小時一年要買賣七百多次，隔夜做法也有一兩百次，成本乘上次數就是每年被扣掉的部分。"
        "持有現金假設沒有利息，也沒有算稅和滑價。",
        f"波動條件的算法：每天 UTC 0 點（台灣早上 8 點）用之前 30 天的每小時漲跌算波動，和它自己過去 365 天的中間值比，"
        f"高於中間值那一天才做。要先有 30 天加 365 天的資料，所以從 {_day(r['vol_start_ms'])} 才開始，比其他做法短；"
        f"同一段一直抱著是每年 {_pct(r['hold_from_vol']['ret'] if r['hold_from_vol'] else None)}。",
        f"事先挑的每一年挑到：{pick_list}。五個做法都算進「扣 {COST * 100:.2f}% 成本後」的成績來挑，"
        + (f"其中 {'、'.join(str(y) for y in negative)} 年挑的時候連最好的一個過去也是賠的，還是照挑；" if negative else "")
        + f"如果現在就重挑（用到 {last_day} 的資料）會是「{PICK_SHORT[next_pick['choice']]}」。只有 {nt} 個測試年，"
        f"而且{'最後一年還沒過完，' if partial else ''}「{nt} 年裡幾年」多一年少一年就差很多。順勢規則只用幣安 2017-08 以後的 UTC 日線來算。",
        "運氣檢查只比了「選哪幾個晚上」這一件事，條件（10 日新高）和進出時間都固定；排名用的是不扣成本的每年報酬，"
        "晚上選得越多、只要每個都是正的就會排得越前面，所以也要看五個晚上各自的數字。發表之後的期間還很短。",
        f"資料：幣安現貨 BTCUSDT 1 小時線，{first_day} 到 {last_day}，只用已經收完的小時；整點的價格用那一小時的開盤價，"
        f"幣安早年停機造成的缺口（共缺 {r['missing']} 個小時）當成那段時間價格沒動。「最深曾經虧多少」用每小時的資金變化算，"
        "會比只看每天收盤算出來的深一點。",
        "過去的結果不保證未來會一樣。這裡只是把兩篇文章的做法照樣重算、再往後看，不是買賣建議。",
    ]
    if problem:
        notes.append(problem)

    return {
        "id": "overnight",
        "title": "只抱隔夜、只抱某兩個小時：兩個發表過的時段做法重做一次",
        "question": ("儀表板已經發現單純的「幾點、星期幾」型態扣掉成本就沒了。Quantpedia 另外發表過兩個更具體的做法，說能留得住："
                     "比特幣創 10 日新高時只抱美股收盤到隔天開盤的隔夜，以及每天只在 UTC 21–23 點持有。照原文重做，"
                     "再看原文樣本之後的表現，這些報酬還在嗎？賺的是「隔夜」還是「創新高才持有」？"),
        "asof": last_day,
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Quantpedia〈How to Profitably Trade Bitcoin's Overnight Sessions?〉（2024-11-12）",
             "url": "https://quantpedia.com/how-to-profitably-trade-bitcoins-overnight-sessions/"},
            {"name": "Quantpedia〈The Seasonality of Bitcoin〉（2023-09-13）",
             "url": "https://quantpedia.com/the-seasonality-of-bitcoin/"},
            {"name": "幣安現貨 BTCUSDT 1 小時線（2017-08 起）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
