"""Historical study "new_listings": after a coin starts trading on Binance, does it drift down for months - against
bitcoin, and against older coins over the same days?

Pre-registered design (written before the results were seen):
Event = first daily kline of a Binance USDT spot pair (exclude stablecoins, leveraged tokens, wrapped/tokenized
stocks, renames/relistings of the same asset). Day 0 = first full day; ordinary person can act at day 1 close.
1. Event study: coin return minus BTC over days 1-30, 31-90, 91-180, 181-365 after listing; mean and median; share
   negative; INCLUDE delisted coins (delisting = -100% at last price if delisted, or last close - state which, and do
   both). Compare with "seasoned" coins: same calendar windows for coins listed >= 1 year earlier (point-in-time), to
   separate "new listing" from "altcoins are weak in general". Cluster uncertainty by listing month; count
   independent months. Split by period: 2018-2020, 2021-2023, 2024-2026; by listing-day volume tercile; by whether
   the first-day move was up/down (hype).
2. Simple portfolios (monthly rebalance, equal weight, 0.10/0.30% cost): (a) coins listed in the last 6 months,
   (b) seasoned top-50 (point-in-time) excluding them, (c) BTC; report yearly return, return/vol, deepest loss; the
   spread a - b.
3. Ordinary-person relevance only as description: "if you held an altcoin basket but skipped coins within their
   first 6/12 months", vs not skipping, vs a placebo skip of random same-size coin sets (200 draws).
Pass test (stated now): new listings (a) underperform seasoned (b) in all three periods, 90% interval excluding
zero, AND the skip rule beats >= 90% of placebo draws.
Today: coins listed on Binance in the last 90 days and their return since listing vs BTC - descriptions only.

Details fixed with the design (before results):
- Data: the survivorship study's store (cache/_store/survivorship/: pair list with trading status, daily candles).
  That store only downloaded pairs that once ranked in the top 60 by monthly dollar volume - a choice that uses the
  future and would favour listings that did well - so every other USDT pair in its list is downloaded here into
  cache/_store/new_listings_daily/ (and stale tails of pairs still trading are topped up there). Nothing in the
  survivorship store is changed.
- A USDT pair added for a coin that already traded on Binance against BTC / BNB / ETH / BUSD more than 30 days
  earlier is not a new listing (first candle of those pairs: cache/_store/new_listings_firsts.json); such coins
  still count as seasoned with their true age. Tokenized shares listed from 2026-06 with a "B" suffix are dropped
  (the survivorship list names only the first few); a one-day close ratio above 20x or below 1/20 that is near a
  whole power of ten (log10 within 0.35) is a redenomination and earlier closes are rescaled (added after the first
  run showed COCOS's 1,295x day). Renames: pairs mapped to the same asset (survivorship.RENAMED plus a
  short list here) form one asset; only the earliest pair is an event.
- A pair has stopped trading at its last candle before a gap of more than 7 days, or at its last candle if its status
  is not TRADING (as in the survivorship study). "Last close" version: the holder gets the last close (cash after);
  "-100%" version: the holding is worth nothing after that close (renamed pairs never get the -100%).
- Windows run from the close of day a to the close of day b, day k = first full day + k; 1-30 means close of day 1
  to close of day 30. Cumulative horizons 30 / 90 / 180 / 365 start at the day-1 close. A window that has not ended
  yet is left out; a coin that stopped before the window began is left out of that window (its loss is already in
  the earlier window).
- Seasoned comparison: equal-weight mean over all coins (not BTC) whose asset first traded on Binance at least 365
  days before the window's start and that are trading at its start, same window, same delisting rule.
- Uncertainty: standard error clustered by listing month; 90% interval = mean +- 1.645 SE.
- Volume tercile: first-full-day dollar volume divided by BTCUSDT's dollar volume that day. Hype: close of the first
  full day versus the close of the first (partial) candle.
- Portfolios: decisions at each month's last close, held from the next day. (a) every event coin whose day-1 close
  has passed and whose first candle is within the last 182 days; (b) the 50 highest 30-day dollar volume coins (not
  BTC) whose asset is at least 365 days old; (c) BTC. Start: first month-end with >= 5 coins in (a) and >= 10 in (b).
  Pass interval: monthly return of a minus b in each period, mean +- 1.645 * sd / sqrt(months).
- Skip rule: basket = 50 highest 30-day dollar volume coins (not BTC) with >= 14 daily candles; skip = drop the ones
  whose asset first traded on Binance within the last 182 (365) days; no backfill. Placebo: drop the same number of
  randomly chosen basket coins each month (200 draws, seed 11). The pass test uses the 6-month skip, 0.10% cost,
  last-close version; "beats" = more money at the end.
"""
from __future__ import annotations

import json
import math
import re
import time

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.studies2 import survivorship as sv

SURV_DIR = common.STORE / "survivorship"
SURV_DAILY = SURV_DIR / "daily"
OWN_DAILY = common.STORE / "new_listings_daily"
FIRSTS = common.STORE / "new_listings_firsts.json"
SYMBOLS = common.STORE / "new_listings_symbols.json"
SUMMARY = common.STORE / "new_listings_summary.json"
HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")
BTC = "BTCUSDT"
OTHER_QUOTES = ("BTC", "BNB", "ETH", "BUSD")
EARLIER_DAYS = 30
COST = 0.001
COST_HIGH = 0.003
YEAR = 365
WINDOWS = ((1, 30), (30, 90), (90, 180), (180, 365))
HORIZONS = (30, 90, 180, 365)
SEASONED_DAYS = 365
NEW_DAYS = 182
SKIP_DAYS = (182, 365)
TOP = 50
VOL_DAYS = 30
BASKET_MIN_DAYS = 14
MIN_A, MIN_B = 5, 10
DRAWS = 200
SEED = 11
Z90 = 1.645
FETCH_BUDGET = 210  # seconds of downloading per run; the rest continues next run
PAUSE = 0.1
STOCKS_FROM = pd.Timestamp("2026-06-01")
JUMP = 20.0
POW_TOL = 0.35  # log10 distance from a whole power of ten
PERIODS = (("2018–2020", "2010-01-01", "2020-12-31"), ("2021–2023", "2021-01-01", "2023-12-31"),
           ("2024–2026", "2024-01-01", "2099-12-31"))
# extra renames / relistings of the same asset not in the survivorship list (pair -> asset)
MORE_ASSETS = {"LUNCUSDT": "LUNA", "BCHSVUSDT": "BSV", "BTTCUSDT": "BTTC"}

# ---------- small helpers ----------


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True, scale: float = 100.0) -> str | None:
    if not _ok(v):
        return None
    v = float(v) * scale
    if abs(v) < 0.05:
        return "0.0%"
    digits = 0 if abs(v) >= 100 else 1
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _num(v, digits: int = 2) -> str | None:
    if not _ok(v):
        return None
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _read(path, default):
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        return value if isinstance(value, type(default)) else default
    except Exception:  # noqa: BLE001 - a damaged store file starts over
        return default


def _write(path, value) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


_SESSION = requests.Session()
_SESSION.headers.update(common.HEADERS)


def _api(path: str, params: dict):
    """Binance public spot endpoint on either host; returns parsed JSON, or None for an unknown symbol."""
    last = None
    for host in HOSTS:
        try:
            reply = _SESSION.get(host + path, params=params, timeout=15)
            time.sleep(PAUSE)
            if reply.status_code == 400:
                return None  # invalid symbol: the endpoint does not know this pair
            reply.raise_for_status()
            return reply.json()
        except Exception as exc:  # noqa: BLE001 - blocked or down: try the next host
            last = exc
    raise last


# ---------- data ----------

def _archive_symbols(month: str) -> set[str]:
    stored = _read(SYMBOLS, {})
    if stored.get("month") == month and stored.get("symbols"):
        return set(stored["symbols"])
    try:
        found = sv._archive_symbols()
        _write(SYMBOLS, {"month": month, "symbols": found})
        return set(found)
    except Exception:  # noqa: BLE001 - keep last month's list if there is one
        return set(stored.get("symbols") or [])


def _fetch_daily(symbol: str, start_ms: int) -> list | None:
    rows, start = [], start_ms
    for _ in range(12):
        page = _api("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start})
        if page is None:
            return None if not rows else rows
        if not page:
            break
        rows += [[r[0], r[4], r[7]] for r in page]
        start = page[-1][0] + 1
        if len(page) < 1000:
            break
    return rows


def _daily(pairs: dict, deadline: float) -> tuple[dict, dict]:
    """{symbol: DataFrame(close, qv)}: survivorship store merged with this study's own downloads."""
    today = pd.Timestamp(common.today_utc())
    out, info = {}, {"own_full": 0, "fetched": 0, "missing": [], "stale": []}
    for symbol in pairs["symbols"]:
        surv = _read(SURV_DAILY / f"{symbol}.json", {})
        own_path = OWN_DAILY / f"{symbol}.json"
        own = _read(own_path, {})
        rows = {str(k)[:10]: v for k, v in (surv.get("rows") or {}).items()
                if isinstance(v, list) and len(v) == 2 and common._finite(v[0]) and v[0] > 0}
        has_surv = bool(rows)
        rows.update({str(k)[:10]: v for k, v in (own.get("rows") or {}).items()
                     if isinstance(v, list) and len(v) == 2 and common._finite(v[0]) and v[0] > 0})
        if not has_surv:
            info["own_full"] += 1
        trading = pairs["status"].get(symbol) == "TRADING"
        need = (not rows and not own.get("done")) or (
            rows and trading and (today - pd.Timestamp(max(rows))).days > 2 and own.get("checked") != _d(today))
        if need:
            if time.time() > deadline:
                (info["stale"] if rows else info["missing"]).append(symbol)
            else:
                try:
                    start = int((pd.Timestamp(max(rows)) - pd.Timedelta(days=3)).timestamp() * 1000) if rows else 0
                    got = _fetch_daily(symbol, start)
                    if got is None and not rows:
                        try:
                            got = sv._archive_daily(symbol, deadline)
                        except Exception:  # noqa: BLE001
                            got = []
                    mine = dict(own.get("rows") or {})
                    for ms, close, qv in got or []:
                        day = pd.to_datetime(int(ms), unit="ms").normalize()
                        if day < today and float(close) > 0:
                            mine[_d(day)] = [float(close), float(qv)]
                            rows[_d(day)] = [float(close), float(qv)]
                    _write(own_path, {"done": True, "checked": _d(today), "rows": dict(sorted(mine.items()))})
                    info["fetched"] += 1
                except Exception:  # noqa: BLE001 - one pair failing must not stop the others
                    (info["stale"] if rows else info["missing"]).append(symbol)
        if len(rows) < 2:
            continue
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=["close", "qv"])
        frame.index = pd.to_datetime(frame.index)
        out[symbol] = frame.sort_index()
    return out, info


def _first_kline(symbol: str) -> str | None:
    page = _api("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1, "startTime": 0})
    if page:
        return _d(pd.to_datetime(int(page[0][0]), unit="ms"))
    reply = _SESSION.get(sv.LISTING, params={"prefix": f"{sv.PREFIX}{symbol}/1d/", "max-keys": 1}, timeout=30)
    time.sleep(PAUSE)
    reply.raise_for_status()
    found = re.findall(r"-1d-(\d{4})-(\d{2})(?:-\d{2})?\.zip", reply.text)
    return f"{found[0][0]}-{found[0][1]}-01" if found else None


def _other_firsts(bases: list[tuple[str, str]], archive: set[str], deadline: float) -> tuple[dict, int]:
    """{base: earliest first-candle date of its BTC / BNB / ETH / BUSD pairs, or None}; bases checked oldest first."""
    stored = _read(FIRSTS, {})
    todo = 0
    for base, usdt_first in sorted(bases, key=lambda x: x[1]):
        if base in stored:
            continue
        if time.time() > deadline:
            todo += 1
            continue
        best, failed = None, False
        for q in OTHER_QUOTES:
            sym = base + q
            if archive and sym not in archive:
                continue
            try:
                first = _first_kline(sym)
            except Exception:  # noqa: BLE001
                failed = True
                continue
            if first and (best is None or first < best):
                best = first
            if best and (pd.Timestamp(usdt_first) - pd.Timestamp(best)).days > EARLIER_DAYS:
                break
        if failed and not best:
            todo += 1
            continue
        stored[base] = best
    _write(FIRSTS, stored)
    return stored, todo


# ---------- panel ----------

def _panel(raw: dict, status: dict) -> dict:
    end_day = raw[BTC].index[-1]
    cut, ended = {}, {}
    for s, f in raw.items():
        f, gap = sv._cut(f[f.index <= end_day])
        if len(f) < 2:
            continue
        cut[s] = f
        ended[s] = bool(gap or status.get(s) != "TRADING")
    coins = sorted(cut, key=lambda s: (s != BTC, s))
    days = pd.date_range(min(f.index[0] for f in cut.values()), end_day, freq="D")
    close = pd.DataFrame({s: cut[s]["close"] for s in coins}).reindex(days)
    qv = pd.DataFrame({s: cut[s]["qv"] for s in coins}).reindex(days).fillna(0.0)
    first = {s: cut[s].index[0] for s in coins}
    last = {s: (cut[s].index[-1] if ended[s] else end_day) for s in coins}
    life = pd.DataFrame({s: (days >= first[s]) & (days <= last[s]) for s in coins}, index=days)
    price = close.ffill().where(life)
    count = close.notna().cumsum()
    return {"coins": coins, "days": days, "P": price.values, "qv": qv.values, "life": life.values,
            "count": count.values, "first": first, "last": last, "ended": ended,
            "fidx": np.array([days.get_loc(first[s]) for s in coins]),
            "lidx": np.array([days.get_loc(last[s]) for s in coins])}


def _assets(p: dict, firsts: dict) -> dict:
    """Asset per pair, the asset's true first day on Binance, and which pairs are listing events."""
    coins = p["coins"]
    asset = {s: MORE_ASSETS.get(s, sv.RENAMED.get(s, sv._base(s))) for s in coins}
    group_first: dict[str, pd.Timestamp] = {}
    for s in coins:
        a = asset[s]
        group_first[a] = min(group_first.get(a, p["first"][s]), p["first"][s])
    earliest = {}
    for s in coins:
        a = asset[s]
        if p["first"][s] == group_first[a] and a not in earliest:
            earliest[a] = s
    true_first, kind = {}, {}
    for s in coins:
        a = asset[s]
        other = firsts.get(sv._base(earliest[a]))
        tf = group_first[a]
        if other and pd.Timestamp(other) < tf:
            tf = pd.Timestamp(other)
        true_first[s] = tf
        if s == BTC:
            kind[s] = "btc"
        elif earliest[a] != s:
            kind[s] = "rename"
        elif (p["first"][s] - tf).days > EARLIER_DAYS:
            kind[s] = "added"
        else:
            kind[s] = "event"
    renamed = {s for s in coins if s in sv.RENAMED or s in MORE_ASSETS}
    return {"asset": asset, "true_first": true_first, "kind": kind, "renamed": renamed,
            "tf_idx": np.array([(true_first[s] - p["days"][0]).days for s in coins])}


def _returns(p: dict, a: dict) -> dict:
    """Cumulative log growth per coin: 'cash' (frozen after the last close) and 'dead' (worth nothing after it)."""
    P, life = p["P"], p["life"]
    prev = np.vstack([np.full((1, P.shape[1]), np.nan), P[:-1]])
    both = life & np.vstack([np.zeros((1, P.shape[1]), bool), life[:-1]])
    r = np.where(both, P / prev - 1, 0.0)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    dead = r.copy()
    T = len(p["days"])
    for j, s in enumerate(p["coins"]):
        if p["ended"][s] and s not in a["renamed"] and p["lidx"][j] + 1 < T:
            dead[p["lidx"][j] + 1, j] = -1.0
    out = {}
    for name, x in (("cash", r), ("dead", dead)):
        lg = np.cumsum(np.log1p(np.maximum(x, -0.999999)), axis=0)
        out[name] = np.vstack([np.zeros((1, lg.shape[1])), lg])  # out[name][t + 1] = log growth to close t
    return out


def _wret(L: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return from close i to close j for every coin (L from _returns)."""
    return np.expm1(L[j + 1] - L[i + 1])


# ---------- event study ----------

def _events(p: dict, a: dict, logs: dict) -> list[dict]:
    days, coins, P = p["days"], p["coins"], p["P"]
    T = len(days)
    b = coins.index(BTC)
    seasoned_cache: dict[tuple, tuple] = {}

    def seasoned(i: int, k: int) -> tuple:
        if (i, k) not in seasoned_cache:
            ok = p["life"][i] & (a["tf_idx"] <= i - SEASONED_DAYS)
            ok[b] = False
            res = []
            for name in ("cash", "dead"):
                r = _wret(logs[name], i, k)[ok]
                res.append(float(r.mean()) if len(r) else None)
            seasoned_cache[(i, k)] = (res, ok)
        return seasoned_cache[(i, k)]

    out = []
    for j, s in enumerate(coins):
        if a["kind"][s] != "event":
            continue
        f = p["fidx"][j]
        d0, e = f + 1, f + 2
        if e >= T or not p["life"][e, j]:
            continue
        ratio = None
        if p["qv"][d0, b] > 0 and p["qv"][d0, j] > 0:
            ratio = p["qv"][d0, j] / p["qv"][d0, b]
        ev = {"coin": s, "first": p["first"][s], "month": p["first"][s].strftime("%Y-%m"),
              "hype": float(P[d0, j] / P[f, j] - 1) if p["life"][d0, j] else None, "vol": ratio,
              "ended": p["ended"][s] and s not in a["renamed"], "win": {}, "cum": {}}
        spans = [("win", (lo, hi)) for lo, hi in WINDOWS] + [("cum", (1, h)) for h in HORIZONS]
        for kind, (lo, hi) in spans:
            i, k = d0 + lo, d0 + hi
            if k >= T or i > p["lidx"][j]:
                continue
            btc = float(P[k, b] / P[i, b] - 1)
            (s_cash, s_dead), ok = seasoned(i, k)
            rc = float(_wret(logs["cash"], i, k)[j])
            rd = float(_wret(logs["dead"], i, k)[j])
            key = hi if kind == "cum" else (lo, hi)
            ev[kind][key] = {"x": rc - btc, "xd": rd - btc, "vs": (rc - s_cash) if s_cash is not None else None,
                             "vsd": (rd - s_dead) if s_dead is not None else None, "n_seasoned": int(ok.sum()),
                             "gone": bool(p["ended"][s] and p["lidx"][j] < k)}
        out.append(ev)
    return out


def _se(vals: list[float], months: list[str]) -> float | None:
    """Standard error of the mean, clustered by listing month."""
    n = len(vals)
    if n < 3:
        return None
    m = float(np.mean(vals))
    groups: dict[str, float] = {}
    for v, g in zip(vals, months):
        groups[g] = groups.get(g, 0.0) + (v - m)
    k = len(groups)
    if k < 3:
        return None
    return math.sqrt(sum(x * x for x in groups.values()) * k / (k - 1)) / n


def _summ(events: list[dict], kind: str, key, field: str) -> dict:
    rows = [(e[kind][key][field], e["month"]) for e in events if key in e[kind] and _ok(e[kind][key][field])]
    if not rows:
        return {"n": 0, "months": 0, "mean": None, "med": None, "neg": None, "lo": None, "hi": None}
    vals, months = [r[0] for r in rows], [r[1] for r in rows]
    m = float(np.mean(vals))
    se = _se(vals, months)
    return {"n": len(vals), "months": len(set(months)), "mean": m, "med": float(np.median(vals)),
            "neg": float(np.mean([v < 0 for v in vals])),
            "lo": m - Z90 * se if se is not None else None, "hi": m + Z90 * se if se is not None else None}


def _ci(st: dict) -> str | None:
    if st["mean"] is None:
        return None
    if st["lo"] is None:
        return f"{_pct(st['mean'])}（區間算不出）"
    return f"{_pct(st['mean'])}（{_pct(st['lo'])}～{_pct(st['hi'])}）"


# ---------- portfolios ----------

def _month_ends(days: pd.DatetimeIndex) -> list[int]:
    return [i for i, t in enumerate(days) if (t + pd.Timedelta(days=1)).day == 1 and i < len(days) - 1]


def _lists(p: dict, a: dict, ends: list[int]) -> dict:
    """Point-in-time lists at each month-end close (only data up to that close is used)."""
    coins = p["coins"]
    b = coins.index(BTC)
    kind = np.array([a["kind"][s] for s in coins])
    is_event = kind == "event"
    vol30 = pd.DataFrame(p["qv"]).rolling(VOL_DAYS, min_periods=1).sum().values
    out = {"a": {}, "b": {}, "basket": {}, "skip": {d: {} for d in SKIP_DAYS}}
    ended = np.array([p["ended"][s] for s in coins])
    for t in ends:
        # trading at this close and not stopping at it (as in the survivorship study: a delisting is announced ahead)
        alive = p["life"][t] & ((p["lidx"] > t) | ~ended)
        age_usdt = t - p["fidx"]
        age = t - a["tf_idx"]
        new = is_event & alive & (age_usdt >= 2) & (age_usdt <= NEW_DAYS)
        out["a"][t] = np.where(new)[0]
        seas = alive & (age >= SEASONED_DAYS)
        seas[b] = False
        idx = np.where(seas)[0]
        out["b"][t] = idx[np.argsort(-vol30[t, idx], kind="stable")][:TOP]
        okb = alive & (p["count"][t] >= BASKET_MIN_DAYS)
        okb[b] = False
        idx = np.where(okb)[0]
        basket = idx[np.argsort(-vol30[t, idx], kind="stable")][:TOP]
        out["basket"][t] = basket
        for dd in SKIP_DAYS:
            out["skip"][dd][t] = basket[age[basket] < dd]
    return out


def _sim(L: np.ndarray, ends: list[int], picks: dict, cost: float, start: int, T: int) -> dict:
    """Equal weight at each month-end close from `start`, drifting until the next one. Returns daily equity
    (closes start+1 .. T-1) and month returns."""
    K = L.shape[1]
    eq = 1.0
    path = []
    prev = np.zeros(K)
    months = []
    rb = [t for t in ends if t >= start]
    for i, t in enumerate(rb):
        nxt = rb[i + 1] if i + 1 < len(rb) else T - 1
        cols = picks[t]
        w = np.zeros(K)
        if len(cols):
            w[cols] = 1.0 / len(cols)
        drift = prev / prev.sum() if prev.sum() > 1e-12 else np.zeros(K)
        before = eq
        eq *= 1 - cost * float(np.abs(w - drift).sum())
        if nxt <= t:
            break
        if len(cols):
            G = np.exp(L[t + 2: nxt + 2][:, cols] - L[t + 1][cols])
            vals = eq * w[cols]
            block = G @ vals
            prev = np.zeros(K)
            prev[cols] = G[-1] * vals
        else:
            block = np.full(nxt - t, eq)
            prev = np.zeros(K)
        path.extend(block.tolist())
        eq = float(block[-1])
        months.append((t, eq / before - 1))
    return {"path": np.array(path), "months": months, "start": rb[0] if rb else None}


def _stats(path: np.ndarray) -> dict:
    if len(path) < 30:
        return {"ret": None, "vol": None, "ratio": None, "deep": None, "growth": None}
    eqs = np.concatenate([[1.0], path])
    net = eqs[1:] / eqs[:-1] - 1
    growth = float(eqs[-1] / eqs[0])
    n = len(net)
    ret = growth ** (YEAR / n) - 1 if growth > 0 else -1.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR))
    deep = float((eqs / np.maximum.accumulate(eqs) - 1).min())
    return {"ret": ret, "vol": vol, "ratio": ret / vol if vol > 1e-9 else None, "deep": deep, "growth": growth}


def _period_stats(run: dict, days: pd.DatetimeIndex) -> dict:
    path, s = run["path"], run["start"]
    dates = days[s + 1: s + 1 + len(path)]
    out = {"all": _stats(path)}
    for name, lo, hi in PERIODS:
        m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        if m.sum() < 30:
            out[name] = _stats(np.array([]))
            continue
        first = int(np.argmax(m))
        base = path[first - 1] if first > 0 else 1.0
        out[name] = _stats(path[m] / base)
        out[name]["from"] = dates[m][0]
    return out


def _spread_months(ra: dict, rb_: dict, days) -> dict:
    ma, mb = dict(ra["months"]), dict(rb_["months"])
    out = {}
    for name, lo, hi in PERIODS:
        diff = [ma[t] - mb[t] for t in ma if t in mb and pd.Timestamp(lo) <= days[t] <= pd.Timestamp(hi)
                and t + 1 < len(days)]
        if len(diff) < 6:
            out[name] = {"n": len(diff), "mean": None, "lo": None, "hi": None}
            continue
        m, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
        h = Z90 * sd / math.sqrt(len(diff))
        out[name] = {"n": len(diff), "mean": m, "lo": m - h, "hi": m + h}
    return out


# ---------- look-ahead check ----------

def _lookahead_ok(p: dict, a: dict, ends: list[int]) -> bool:
    """Lists decided at each month-end must not change when every price and volume after a cut date is scrambled."""
    rng = np.random.default_rng(SEED + 1)
    base = _lists(p, a, ends)
    for cut in rng.choice(ends[12:-2], size=min(4, max(len(ends) - 14, 0)), replace=False):
        q = dict(p)
        P, V = p["P"].copy(), p["qv"].copy()
        T = len(P)
        perm = rng.permutation(np.arange(cut + 1, T))
        P[cut + 1:] = P[perm] * rng.uniform(0.2, 5.0, size=(1, P.shape[1]))
        V[cut + 1:] = V[perm] * rng.uniform(0.2, 5.0, size=(1, V.shape[1]))
        q["P"], q["qv"] = P, V
        other = _lists(q, a, [t for t in ends if t <= cut])
        for key in ("a", "b", "basket"):
            for t in other[key]:
                if not np.array_equal(np.sort(other[key][t]), np.sort(base[key][t])):
                    return False
    return True


# ---------- the study ----------

def _load() -> tuple[dict, dict, dict, dict]:
    t0 = time.time()
    deadline = t0 + FETCH_BUDGET
    pairs = _read(SURV_DIR / "pairs.json", {})
    if not pairs.get("symbols"):
        raise RuntimeError("存活者偏差研究的交易對清單還沒建立")
    every = set(pairs["symbols"])
    bases = {sv._base(s) for s in every}
    pairs["symbols"] = [s for s in pairs["symbols"] if sv._wanted(s, bases)]
    raw, info = _daily(pairs, deadline)
    if BTC not in raw:
        raise RuntimeError("比特幣的日線取不到")
    pegged = [s for s, f in raw.items() if float((f["close"] - 1).abs().median()) < sv.PEG_BAND]
    for s in pegged:
        raw.pop(s)
    info["pegged"] = pegged
    # tokenized shares / ETFs: from 2026-06 Binance lists them with a "B" suffix (TSLAB, SPYB, ...); the
    # survivorship list only names the first few
    stocks = [s for s, f in raw.items() if s != BTC and sv._base(s).endswith("B") and len(sv._base(s)) >= 3
              and f.index[0] >= STOCKS_FROM]
    for s in stocks:
        raw.pop(s)
    info["stocks"] = stocks
    # redenominations (e.g. COCOS 2021-01, DREP 2021-04): a one-day move of more than JUMP times either way is a
    # change of units, not a price move; earlier closes are rescaled to the new units
    info["redenom"] = []
    for s, f in raw.items():
        ratio = (f["close"] / f["close"].shift(1)).values
        n_first = len(sv._cut(f)[0])  # only the first trading stretch counts (a relisting after a gap is dropped)
        lg = np.log10(np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0))
        # a change of units is a near power of ten (100x, 1000x ...); a crash such as LUNA's (about 3,300x) is not
        hits = np.where((np.abs(lg) >= math.log10(JUMP)) & (np.abs(lg - np.round(lg)) <= POW_TOL)
                        & (np.arange(len(lg)) < n_first))[0]
        if len(hits):
            close = f["close"].values.copy()
            for i in hits:
                close[:i] *= ratio[i]
            raw[s] = f.assign(close=close)
            info["redenom"].append(s)
    archive = _archive_symbols(common.today_utc()[:7])
    firsts_in = [(sv._base(s), _d(f.index[0])) for s, f in raw.items() if s != BTC]
    firsts, todo = _other_firsts(firsts_in, archive, deadline)
    info["firsts_todo"] = todo
    info["archive_n"] = len(archive)
    return pairs, raw, info, firsts


def study() -> dict:
    t0 = time.time()
    pairs, raw, info, firsts = _load()
    p = _panel(raw, pairs["status"])
    a = _assets(p, firsts)
    logs = _returns(p, a)
    days, coins = p["days"], p["coins"]
    T = len(days)
    last = days[-1]
    b = coins.index(BTC)
    events = _events(p, a, logs)
    kinds = {str(k): int(v) for k, v in pd.Series(a["kind"]).value_counts().items()}

    # ----- table 1: event study, all listings -----
    t1 = []
    for kind, keys, label in (("win", WINDOWS, "第 {}–{} 天"), ("cum", HORIZONS, "第 1 天收盤到第 {} 天")):
        for key in keys:
            name = label.format(*key) if kind == "win" else label.format(key)
            x = _summ(events, kind, key, "x")
            vs = _summ(events, kind, key, "vs")
            vsd = _summ(events, kind, key, "vsd")
            seas = [e[kind][key]["x"] - e[kind][key]["vs"] for e in events
                    if key in e[kind] and _ok(e[kind][key]["vs"])]
            gone = sum(1 for e in events if key in e[kind] and e[kind][key]["gone"])
            t1.append([name, f"{x['n']}（{x['months']} 個月；{gone} 個已下市）", f"{_pct(x['mean'])}／{_pct(x['med'])}",
                       _pct(x["neg"], signed=False), _pct(float(np.mean(seas)) if seas else None), _ci(vs),
                       _pct(vs["med"]), _ci(vsd)])

    # ----- table 2: splits at 6 and 12 months -----
    def period_of(e):
        y = e["first"].year
        return "2018–2020" if y <= 2020 else ("2021–2023" if y <= 2023 else "2024–2026")

    vols = [e["vol"] for e in events if _ok(e["vol"])]
    cuts = np.quantile(vols, [1 / 3, 2 / 3]) if len(vols) >= 9 else None
    groups = [(f"上架在 {name}", lambda e, n=name: period_of(e) == n) for name, _, _ in PERIODS]
    if cuts is not None:
        groups += [("首日成交額：低三分之一", lambda e: _ok(e["vol"]) and e["vol"] <= cuts[0]),
                   ("首日成交額：中三分之一", lambda e: _ok(e["vol"]) and cuts[0] < e["vol"] <= cuts[1]),
                   ("首日成交額：高三分之一", lambda e: _ok(e["vol"]) and e["vol"] > cuts[1])]
    groups += [("第一個整天上漲（有熱度）", lambda e: _ok(e["hype"]) and e["hype"] > 0),
               ("第一個整天下跌", lambda e: _ok(e["hype"]) and e["hype"] <= 0)]
    t2, split = [], {}
    for label, keep in groups:
        sub = [e for e in events if keep(e)]
        x6, v6 = _summ(sub, "cum", 180, "x"), _summ(sub, "cum", 180, "vs")
        x12, v12, d12 = _summ(sub, "cum", 365, "x"), _summ(sub, "cum", 365, "vs"), _summ(sub, "cum", 365, "vsd")
        split[label] = {"v6": v6, "v12": v12, "d12": d12, "x12": x12}
        t2.append([label, f"{len(sub)}（{len({e['month'] for e in sub})} 個月）", _pct(x6["med"]), _ci(v6),
                   _pct(x12["med"]), _ci(v12), _pct(v12["med"]), _ci(d12)])

    # ----- portfolios -----
    ends = _month_ends(days)
    lists = _lists(p, a, ends)
    start = next((t for t in ends if len(lists["a"][t]) >= MIN_A and len(lists["b"][t]) >= MIN_B), ends[0])
    btc_pick = {t: np.array([b]) for t in ends}
    runs = {}
    for ver in ("cash", "dead"):
        for cost in (COST, COST_HIGH):
            for key, picks in (("a", lists["a"]), ("b", lists["b"]), ("c", btc_pick)):
                runs[(key, ver, cost)] = _sim(logs[ver], ends, picks, cost, start, T)
    pst = {k: _period_stats(v, days) for k, v in runs.items()}
    spread = {ver: _spread_months(runs[("a", ver, COST)], runs[("b", ver, COST)], days) for ver in ("cash", "dead")}
    n_a = [len(lists["a"][t]) for t in ends if t >= start]
    n_b = [len(lists["b"][t]) for t in ends if t >= start]

    def prow(label, key, ver):
        m, hi = pst[(key, ver, COST)], pst[(key, ver, COST_HIGH)]
        return [label, _pct(m["all"]["ret"]), _num(m["all"]["ratio"]), _pct(m["all"]["deep"])] + \
            [_pct(m[name]["ret"]) for name, _, _ in PERIODS] + [_pct(hi["all"]["ret"])]

    t3 = [prow("(a) 上架未滿 6 個月的幣", "a", "cash"), prow("(b) 上架滿 1 年、成交額前 50 名", "b", "cash"),
          prow("(c) 比特幣", "c", "cash")]

    def diff_row(label, ver):
        A, B = pst[("a", ver, COST)], pst[("b", ver, COST)]
        Ah, Bh = pst[("a", ver, COST_HIGH)], pst[("b", ver, COST_HIGH)]

        def d(x, y):
            return x - y if _ok(x) and _ok(y) else None
        return [label, _pct(d(A["all"]["ret"], B["all"]["ret"])), None, None] + \
            [_pct(d(A[n]["ret"], B[n]["ret"])) for n, _, _ in PERIODS] + [_pct(d(Ah["all"]["ret"], Bh["all"]["ret"]))]

    t3.append(diff_row("(a) 減 (b)：每年報酬差", "cash"))
    for ver, label in (("cash", "(a) 減 (b)：每月平均差（90% 區間）"), ("dead", "下市算歸零：(a) 減 (b) 每月平均差")):
        row = [label, None, None, None]
        for name, _, _ in PERIODS:
            sp = spread[ver][name]
            row.append(None if sp["mean"] is None else
                       f"{_pct(sp['mean'])}（{_pct(sp['lo'])}～{_pct(sp['hi'])}）")
        t3.append(row + [None])
    t3 += [prow("下市算歸零：(a)", "a", "dead"), prow("下市算歸零：(b)", "b", "dead"),
           diff_row("下市算歸零：(a) 減 (b) 每年報酬差", "dead")]

    # ----- skip rule vs placebo -----
    bstart = next((t for t in ends if len(lists["basket"][t]) >= 20), ends[0])
    rng = np.random.default_rng(SEED)
    skip_res = {}
    for ver in ("cash", "dead"):
        for cost in (COST, COST_HIGH):
            base_run = _sim(logs[ver], ends, lists["basket"], cost, bstart, T)
            skip_res[(ver, cost, "none")] = base_run
            for dd in SKIP_DAYS:
                picks = {t: np.setdiff1d(lists["basket"][t], lists["skip"][dd][t]) for t in ends}
                skip_res[(ver, cost, dd)] = _sim(logs[ver], ends, picks, cost, bstart, T)
                if cost == COST:
                    growths = []
                    for _ in range(DRAWS):
                        rp = {}
                        for t in ends:
                            bk, k = lists["basket"][t], len(lists["skip"][dd][t])
                            rp[t] = np.setdiff1d(bk, rng.choice(bk, size=k, replace=False)) if k else bk
                        growths.append(_sim(logs[ver], ends, rp, cost, bstart, T)["path"][-1])
                    skip_res[(ver, "placebo", dd)] = np.array(growths)
    sst = {k: _period_stats(v, days) for k, v in skip_res.items() if isinstance(v, dict)}
    beat = {}
    for ver in ("cash", "dead"):
        for dd in SKIP_DAYS:
            g = skip_res[(ver, COST, dd)]["path"][-1]
            draws = skip_res[(ver, "placebo", dd)]
            beat[(ver, dd)] = float(np.mean(draws < g))
    avg_skip = {dd: float(np.mean([len(lists["skip"][dd][t]) for t in ends if t >= bstart])) for dd in SKIP_DAYS}

    def srow(label, key, ver, dd=None):
        m = sst[(ver, COST, key)]
        hi = sst[(ver, COST_HIGH, key)]
        tail = f"{beat[(ver, dd)] * 100:.0f}%" if dd else None
        return [label, _pct(m["all"]["ret"]), _num(m["all"]["ratio"]), _pct(m["all"]["deep"])] + \
            [_pct(m[name]["ret"]) for name, _, _ in PERIODS] + [_pct(hi["all"]["ret"]), tail]

    def prow_placebo(label, ver, dd):
        draws = skip_res[(ver, "placebo", dd)]
        n = len(skip_res[(ver, COST, "none")]["path"])
        med = float(np.median(draws))
        ret = med ** (YEAR / n) - 1 if med > 0 else -1.0
        lo, hi = np.quantile(draws, [0.05, 0.95])
        rl = lo ** (YEAR / n) - 1 if lo > 0 else -1.0
        rh = hi ** (YEAR / n) - 1 if hi > 0 else -1.0
        return [label, f"{_pct(ret)}（90% 的抽籤在 {_pct(rl)}～{_pct(rh)}）", None, None, None, None, None, None, None]

    t4 = [srow("不避開（成交額前 50 名，不含比特幣）", "none", "cash"),
          srow("避開上架未滿 6 個月的幣", 182, "cash", 182), prow_placebo("隨機避開同樣數量（6 個月版，200 次）", "cash", 182),
          srow("避開上架未滿 12 個月的幣", 365, "cash", 365), prow_placebo("隨機避開同樣數量（12 個月版，200 次）", "cash", 365),
          srow("下市算歸零：不避開", "none", "dead"), srow("下市算歸零：避開未滿 6 個月", 182, "dead", 182),
          srow("下市算歸零：避開未滿 12 個月", 365, "dead", 365)]
    # 9 columns is too many: merge the cost-0.30% column into the label column set
    t4 = [[r[0], r[1], r[2], r[3], "｜".join(x or "—" for x in r[4:7]), r[7], r[8]] for r in t4]

    # ----- pass test -----
    sp = spread["cash"]
    under_all = all(sp[n]["hi"] is not None and sp[n]["hi"] < 0 for n, _, _ in PERIODS)
    skip_pass = beat[("cash", 182)] >= 0.9
    passed = under_all and skip_pass

    # ----- now: listings of the last 90 days -----
    recent = []
    for j, s in enumerate(coins):
        if a["kind"][s] != "event" or (last - p["first"][s]).days > 90:
            continue
        f = p["fidx"][j]
        lj = min(p["lidx"][j], T - 1)
        rc = float(P_ratio(p, f, lj, j))
        rbt = float(P_ratio(p, f, lj, b))
        recent.append((p["first"][s], sv._base(s), rc, rc - rbt, p["ended"][s]))
    recent.sort()
    now = [{"label": "最後一根收完的日線", "value": _d(last)}]
    if recent:
        xs = [r[3] for r in recent]
        now.append({"label": "最近 90 天在幣安新上架的幣（USDT 交易對，不含穩定幣、槓桿代幣、股票代幣、改名）",
                    "value": f"{len(recent)} 個；從第一根日線收盤到現在，減掉同期比特幣的中位數 {_pct(float(np.median(xs)))}，"
                             f"{sum(x < 0 for x in xs)} 個比比特幣差"})
        listing = "、".join(f"{r[1]}（{_d(r[0])[5:]} 上架，{_pct(r[2])}，比比特幣 {_pct(r[3])}"
                           f"{'，已停止交易' if r[4] else ''}）" for r in recent[-40:])
        now.append({"label": "各幣上架以來的漲跌（括號：上架日、漲跌、減掉比特幣）", "value": listing})
    else:
        now.append({"label": "最近 90 天在幣安新上架的幣", "value": "資料裡沒有"})
    now.append({"label": "預先定的及格檢驗", "value": "通過" if passed else "沒通過"})

    # ----- verdict -----
    def cum(h, field="x", stat="med"):
        return _summ(events, "cum", h, field)[stat]

    per_v12 = [split[f"上架在 {n}"]["v12"] for n, _, _ in PERIODS]
    consistent = len({v["med"] is not None and v["med"] < 0 for v in per_v12}) == 1
    n_sig = sum(1 for n, _, _ in PERIODS if sp[n]["hi"] is not None and sp[n]["hi"] < 0)
    n_events = len(events)
    n_gone = sum(1 for e in events if e["ended"])
    verdict = (f"幣安新上架的 {n_events} 個幣（含後來停止交易的 {n_gone} 個），從上架第 2 天收盤起抱 1、3、6、12 個月，"
               f"減掉比特幣的中位數是 {_pct(cum(30))}、{_pct(cum(90))}、{_pct(cum(180))}、{_pct(cum(365))}，"
               f"減掉同期老幣的中位數是 {_pct(cum(30, 'vs'))}、{_pct(cum(90, 'vs'))}、{_pct(cum(180, 'vs'))}、"
               f"{_pct(cum(365, 'vs'))}（平均被少數暴漲的幣拉高，12 個月平均 {_pct(cum(365, 'vs', 'mean'))}；"
               f"下市算歸零差不多）。三個時段 12 個月減老幣的中位數是 {'、'.join(_pct(v['med']) or '—' for v in per_v12)}"
               f"（{'方向一致' if consistent else '方向不一致'}），但每月持有「新幣組合減老幣組合」只有 {n_sig} 段的 90% 區間在零以下。"
               f"每月避開上架未滿 6 個月的幣，最後的錢贏過 {beat[('cash', 182)] * 100:.0f}% 的隨機避開"
               f"（12 個月版 {beat[('cash', 365)] * 100:.0f}%），預先定的檢驗{'通過' if passed else '沒通過'}。")

    # ----- tables -----
    span = f"{_d(days[runs[('a', 'cash', COST)]['start'] + 1])} 到 {_d(last)}"
    tables = [
        {"caption": ("每個新上架的幣（USDT 交易對第一根日線＝上架；第一個整天是第 0 天，第 1 天收盤才算買得到）："
                     "上半是不重疊的時間段，下半是從第 1 天收盤一路抱到第 N 天。「減比特幣」＝這個幣的漲跌減掉同一段比特幣的漲跌；"
                     "「老幣」＝同一段時間、在幣安交易滿 1 年的其他幣（不含比特幣）的平均。停止交易的幣基本照最後收盤換現金；"
                     "最後一欄改成停止交易就歸零（只是改名的不算）。括號是 90% 區間（同一個月上架的幣算一組，因為它們遇到同一段行情）。"
                     "沒有扣成本。"),
         "columns": ["期間", "幣數（上架月份數；其中已下市）", "減比特幣：平均／中位數", "比比特幣差的比例",
                     "同期老幣減比特幣（平均）", "減老幣：平均（90% 區間）", "減老幣：中位數", "下市算歸零：減老幣平均"],
         "rows": t1[:16]},
        {"caption": ("分組看：從第 1 天收盤抱 6 個月（180 天）和 12 個月（365 天）。成交額分組用第一個整天的成交額除以同一天比特幣的成交額，"
                     "照全部事件切成三等份；「有熱度」＝第一個整天收盤比第一根（不完整的）日線收盤高。括號是 90% 區間。"),
         "columns": ["分組", "幣數（上架月份數）", "6 個月：減比特幣中位數", "6 個月：減老幣平均", "12 個月：減比特幣中位數",
                     "12 個月：減老幣平均", "12 個月：減老幣中位數", "12 個月下市算歸零：減老幣平均"],
         "rows": t2[:16]},
        {"caption": (f"三個簡單組合（{span}），每月最後一天收盤重選、平均分配，隔天開始持有，成本每次買賣 {COST * 100:.2f}%"
                     f"（最後一欄 0.30%）。(a) 每月平均 {np.mean(n_a):.0f} 個幣（最少 {min(n_a)}、最多 {max(n_a)}），"
                     f"(b) 每月 {np.mean(n_b):.0f} 個。分段欄是那幾年的每年報酬（第一段從組合開始的那天算）。"
                     "「每月平均差」是 (a) 當月報酬減 (b) 當月報酬的平均，括號是 90% 區間；預先定的及格條件要三段的區間都在零以下。"),
         "columns": ["組合", "每年報酬", "報酬÷波動", "最深曾經虧多少"] + [f"{n} 每年" for n, _, _ in PERIODS] + ["成本 0.30% 每年"],
         "rows": t3[:16]},
        {"caption": (f"一般人的情境（只是描述，不是建議）：每月底挑過去 30 天成交額前 50 名的幣（不含比特幣，至少有 {BASKET_MIN_DAYS} 根日線）平均持有，"
                     f"比較「不避開」和「避開上架未滿 6／12 個月的幣」（避掉的不補，平均每月避開 {avg_skip[182]:.1f}／{avg_skip[365]:.1f} 個），"
                     f"以及每月隨機避開同樣數量的幣（{DRAWS} 次抽籤）。從 {_d(days[bstart + 1])} 開始。"
                     "最後一欄：避開新幣最後的錢，比幾成的隨機抽籤多（及格線 90%）。"),
         "columns": ["做法", "每年報酬", "報酬÷波動", "最深曾經虧多少", "2018–20｜2021–23｜2024–26 每年", "成本 0.30% 每年",
                     "贏過幾成隨機避開"],
         "rows": t4[:16]},
    ]

    look_ok = _lookahead_ok(p, a, ends)
    renamed_n = kinds.get("rename", 0)
    added_n = kinds.get("added", 0)
    not_cov = info["missing"]
    notes = [
        "預先定的設計（看結果之前寫下，照原文）：事件＝幣安 USDT 現貨交易對的第一根日線（排除穩定幣、槓桿代幣、包裝代幣與股票代幣、"
        "同一個資產的改名或重新上架）。第 0 天＝第一個完整的一天；一般人最早在第 1 天收盤才買得到。① 事件研究：上架後第 1–30、31–90、"
        "91–180、181–365 天，幣的漲跌減比特幣；平均、中位數、比比特幣差的比例；包含下市的幣（停止交易照最後收盤、或照歸零，兩種都算）。"
        "和「老幣」比：同一段日曆時間、當時已上架滿 1 年的幣，用來分開「新上架」和「山寨幣本來就弱」。不確定性依上架月份分組，並數有幾個獨立月份。"
        "分時段 2018–2020、2021–2023、2024–2026；分首日成交額三等份；分第一天漲或跌（熱度）。② 簡單組合（每月重選、平均分配、成本 0.10%／0.30%）："
        "(a) 最近 6 個月上架的幣、(b) 上架滿 1 年的成交額前 50 名（當時名單）、(c) 比特幣；報每年報酬、報酬÷波動、最深虧損，以及 a 減 b。"
        "③ 一般人情境只做描述：持有一籃山寨幣但避開上架 6／12 個月內的幣，對照不避開、以及隨機避開同樣數量的幣（200 次）。"
        "及格條件（事先定）：新上架 (a) 在三個時段都比老幣 (b) 差、90% 區間不含零，而且避開規則贏過至少 90% 的隨機抽籤。",
        f"及格檢驗的結果：三段 (a) 減 (b) 每月差的 90% 區間{'都在零以下' if under_all else '並非都在零以下'}；"
        f"避開 6 個月贏過 {beat[('cash', 182)] * 100:.0f}% 的隨機抽籤 → {'通過' if passed else '沒通過'}。"
        f"（下市算歸零的版本：避開 6 個月贏過 {beat[('dead', 182)] * 100:.0f}%。）",
        f"資料：交易對清單和大部分日線來自存活者偏差研究存下的幣安資料（{len(pairs['symbols'])} 個 USDT 交易對，已排除穩定幣、槓桿代幣、"
        "股票代幣；收盤一直貼著 1 美元的也去掉）。那份資料只下載過「曾經排進每月成交額前 60 名」的交易對——這是用未來資訊挑的，會偏向後來"
        f"表現好的新幣——所以其餘的交易對在這裡另外下載補齊（{info['own_full']} 個只來自這裡的下載）。"
        + (f"這次還有 {len(not_cov)} 個交易對沒取到日線（下次執行會補）：{'、'.join(sv._base(s) for s in not_cov[:15])}。"
           if not_cov else "這次所有交易對都有日線。")
        + (f"另有 {len(info['stale'])} 個還在交易的幣最近幾天沒更新。" if info["stale"] else ""),
        f"哪些不算「新上架」：同一個資產換代號或重新上架（例如 MATIC→POL、EOS→A、舊 LUNA→LUNC，共 {renamed_n} 個交易對），"
        f"以及 USDT 交易對開之前、這個幣已經在幣安用 BTC／BNB／ETH／BUSD 交易超過 {EARLIER_DAYS} 天的（{added_n} 個，多半是 2018–2019 年補開 USDT 交易對的老幣）；"
        "這些幣在比較裡算老幣，年齡從真正第一次在幣安交易算起。"
        + (f"這次還有 {info['firsts_todo']} 個幣沒查到其他交易對的開始日（當成新上架；下次執行會補）。" if info["firsts_todo"] else "")
        + "改名對照是人工整理，不保證完整。"
        + f"2026 年 6 月起上架、代號以 B 結尾的股票／ETF 代幣（例如 TSLAB、SPYB）一律排除（{len(info['stocks'])} 個）。"
        + (f"幣種單位調整（一天之內價格變成 {JUMP:.0f} 倍以上或以下、而且接近 100 倍、1000 倍這種整數，例如 COCOS、DREP）當成換單位，"
           "之前的價格照比例換算；LUNA 2022 年那種崩跌不是整數倍，照真的跌算"
           f"（{len(info['redenom'])} 個：{'、'.join(sv._base(s) for s in info['redenom'][:8])}）。" if info["redenom"] else ""),
        f"事件數：{n_events} 個新上架（{len({e['month'] for e in events})} 個不同的上架月份），其中 {n_gone} 個後來停止交易"
        "（日線中斷超過 7 天或現在狀態不是交易中；中斷後又恢復的只算第一段）。第一根日線通常不是完整的一天，所以從第二天起算。",
        "只有幣安一家：很多幣在別的交易所先上、在幣安上架時已經漲過一段；這裡看的是「在幣安開始交易之後」。幣安下架的幣常常在別處繼續交易，"
        "所以「歸零」是偏嚴的假設，「照最後收盤」是偏寬的假設（真實常介於兩者之間，而且下架前通常會先公告）。",
        "平均值會被少數漲幾十倍的幣拉高，中位數比較接近「一般新幣」的情況；兩個都列。「老幣」是同期所有老幣的平均，"
        "2020–2021 年大漲時被少數漲幾十倍的老幣拉得很高，所以 2018–2020 年上架那一組「減老幣」會低於 -100%（新幣自己最多只能跌 100%）。90% 區間把同一個月上架的幣算成一組，"
        "但同一年的行情（例如 2021 年的大漲、2022 年的大跌）仍然讓不同月份彼此相關，真正獨立的觀察比月份數還少。",
        f"時間的算法：所有名單只用到當天收盤為止的資料（幣安日線在 UTC 0 點收盤），隔天才照做。檢查：把某個日期之後的價格和成交額打亂、"
        f"重新算名單，之前每個月的名單都{'沒有改變' if look_ok else '有改變（有問題）'}。報酬、波動和最深虧損用每日資料，一年用 365 天換算。"
        "成本只用粗略的數字，新幣剛上架時價差和滑價常常更大，沒有算稅，現金沒有利息。",
        f"這次執行用了 {time.time() - t0:.0f} 秒。過去的結果不代表未來；這是描述歷史，不是買賣建議。",
    ]
    sources = [
        {"name": "Li, Luo, Wang & Wei（AFT 2025）：交易所上架公告的首日報酬", "url":
            "https://drops.dagstuhl.de/storage/00lipics/lipics-vol354-aft2025/html/LIPIcs.AFT.2025.11/LIPIcs.AFT.2025.11.html"},
        {"name": "Ante（Blockchain Research Lab）：327 次交易所上架的市場反應", "url":
            "https://www.blockchainresearchlab.org/portfolio/market-reaction-to-exchange-listings-of-cryptocurrencies/"},
        {"name": "The TIE：2026 年交易所上架實際帶來什麼", "url":
            "https://www.thetie.io/insights/what-does-an-exchange-listing-actually-deliver-in-2026"},
        {"name": "幣安公開歷史資料庫（現貨日線，含已下市交易對）", "url": "https://data.binance.vision/?prefix=data/spot/monthly/klines/"},
        {"name": "幣安現貨 K 線端點說明", "url":
            "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
    ]
    summary = {"asof": _d(last), "seconds": round(time.time() - t0, 1), "events": n_events, "gone": n_gone,
               "months": len({e["month"] for e in events}), "kinds": kinds, "lookahead_ok": look_ok,
               "passed": passed, "beat": {f"{k[0]}_{k[1]}": v for k, v in beat.items()},
               "missing": not_cov, "firsts_todo": info["firsts_todo"],
               "start_a": _d(days[start]), "start_basket": _d(days[bstart])}
    _write(SUMMARY, summary)
    result = {"id": "new_listings", "title": "新上架的幣，上架之後幾個月怎麼走",
              "question": ("幣在幣安上架（交易所開始讓大家買賣這個幣）之後，接下來 1 到 12 個月，相對比特幣、相對同期的老幣，"
                           "是不是系統性地比較弱？包含後來下市的幣，結果還一樣嗎？"),
              "asof": _d(last), "verdict": verdict, "now": now, "tables": tables, "notes": notes, "sources": sources}
    return _clean(result)


def P_ratio(p: dict, i: int, j: int, col: int) -> float:
    return p["P"][j, col] / p["P"][i, col] - 1


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if x is None:
        return "—"
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else "—"
    if isinstance(x, np.integer):
        return int(x)
    return x


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
