"""Historical study "donchian_ensemble": do two published breakout systems add anything over the plain rules the
dashboard has already checked?

S1  Zarattini, Pagani, Barbon, "Catching Crypto Trends" (Swiss Finance Institute research paper 25-80): nine closing
    price channels (5 ... 360 days); a model goes long when the close is the highest close of its window, and leaves
    at the first close at or below a trailing stop that only moves up (the larger of yesterday's stop and the middle
    of the channel); the nine models are averaged and the amount held is scaled to 25% yearly volatility with the
    90-day standard deviation of daily returns, capped at 2x (we also show the 1x cap, without borrowing).
S2  The Turtle rules (Curtis Faith, "Way of the Turtle"): System 2 (55-day high in, 20-day low out) and System 1
    (20-day high in, 10-day low out, skip the signal after a winner), units sized so that a move of 1N (N = 20-day
    average true range) is 1% of equity, one more unit every 0.5N up to four, all units stopped 2N below the last
    entry. Long only, closes only.

Benchmarks rebuilt locally exactly as in trend_ensemble: buy-and-hold, the plain 20/10 closing breakout, and the blend
of 17 trend rules scaled to a 40% volatility target.

No look-ahead: every decision uses data up to and including the close of day t and sets the position held during day
t+1; stops are checked on closes only. study() really runs a perturbation check (prices after a date are changed and
the positions up to the next day must stay identical) and reports the outcome. Only closed daily candles from public,
key-less endpoints through common.get_json.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.collectors import classic
from dashboard.collectors import price as px

COST = 0.001  # 0.10% of the amount traded
BORROW = 0.08  # yearly interest charged on the borrowed part of a position above 100%
YEAR = 365
DAY_MS = 86_400_000
MAIN_START = "2018-01-01"
FULL_START = "2015-01-01"
SAMPLE_END = "2025-03-19"  # last day of the paper's sample
POST_START = "2025-04-01"
LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
TARGET = 0.25
S1_VOL_DAYS = 90
BAND = 0.20  # the paper's rebalancing threshold for changes that come only from volatility
COINS = ("BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "LTC", "LINK", "TRX", "DOT", "AVAX", "BCH", "XLM", "ATOM",
         "ETC")
STORE_FILE = "donchian_ensemble_klines.json"
# the 17 rules of trend_ensemble (benchmark)
MA_DAYS = (10, 20, 30, 50, 75, 100, 150, 200)
CHANNEL = (10, 20, 50, 100)
PAST = (14, 30, 60, 90, 180)
BLEND_VOL_DAYS = 30
BLEND_TARGET = 0.40
# Turtle
ATR_DAYS = 20
RISK = 0.01
ADD_STEP = 0.5
STOP_N = 2.0
MAX_UNITS = 4
CHECK_DATES = ("2016-06-30", "2018-12-15", "2020-03-12", "2021-11-10", "2023-06-30", "2025-03-19")
SEED = 20250319

TURTLE = {  # key: (entry days, exit days, options)
    "t2_plain": (55, 20, dict(sized=False, units=1, stop=None, skip=False, cap=1.0)),
    "t2_unit": (55, 20, dict(sized=True, units=1, stop=None, skip=False, cap=1.0)),
    "t2_stop": (55, 20, dict(sized=True, units=1, stop=STOP_N, skip=False, cap=1.0)),
    "t2_full": (55, 20, dict(sized=True, units=MAX_UNITS, stop=STOP_N, skip=False, cap=1.0)),
    "t2_free": (55, 20, dict(sized=True, units=MAX_UNITS, stop=STOP_N, skip=False, cap=None)),
    "t1_plain": (20, 10, dict(sized=False, units=1, stop=None, skip=False, cap=1.0)),
    "t1_skip": (20, 10, dict(sized=False, units=1, stop=None, skip=True, cap=1.0)),
    "t1_noskip": (20, 10, dict(sized=True, units=MAX_UNITS, stop=STOP_N, skip=False, cap=1.0)),
    "t1_full": (20, 10, dict(sized=True, units=MAX_UNITS, stop=STOP_N, skip=True, cap=1.0)),
}


# ---------- formatting ----------

def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _pct(v, signed: bool = True) -> str:
    """v is already in percent."""
    if not _ok(v):
        return "—"
    v = float(v)
    if abs(v) < 0.05:
        return "0.0%"
    digits = 0 if abs(v) >= 100 else 1
    return f"{v:+,.{digits}f}%" if signed else f"{v:,.{digits}f}%"


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _usd(v) -> str:
    return f"{float(v):,.0f} 美元" if _ok(v) else "—"


def _lvl(v) -> str:
    return f"{float(v):,.0f} 美元" if _ok(v) else "（沒有數字）"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------- data ----------

def _daily_close() -> pd.Series:
    """Closed daily closes since 2010 on a DatetimeIndex with no missing day (a missing day repeats the last close)."""
    close = classic._long_close()
    close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
    close = close[close.index < pd.Timestamp(common.today_utc())]  # never today's unfinished candle
    close = close[close > 0]
    if len(close) < 3000:
        raise RuntimeError(f"長期日線只有 {len(close)} 筆")
    return close.asfreq("D").ffill()


def _klines() -> tuple[dict, list[str]]:
    """Finished Binance spot daily candles (high, low, close) per coin from its first listed day.

    Finished candles are kept under common.STORE so a later run only asks for the newest days."""
    path = common.STORE / STORE_FILE
    try:
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(stored, dict):
            stored = {}
    except Exception:  # noqa: BLE001 - a damaged store file starts over
        stored = {}
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    frames, problems = {}, []
    for coin in COINS:
        symbol = f"{coin}USDT"
        rows = {int(r[0]): r for r in stored.get(symbol, []) if isinstance(r, list) and len(r) == 4}
        try:
            start = max(rows) - 3 * DAY_MS if rows else 0
            for _ in range(12):
                page = px._binance("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 1000,
                                                      "startTime": start})
                if not page:
                    break
                for r in page:
                    rows[int(r[0])] = [int(r[0]), float(r[2]), float(r[3]), float(r[4])]
                start = page[-1][0] + 1
                if len(page) < 1000:
                    break
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            if not rows:
                problems.append(f"{coin} 這次取不到資料（{type(exc).__name__}），沒有算進去。")
                continue
            problems.append(f"{coin} 這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的日線。")
        clean = [rows[ms] for ms in sorted(rows) if ms < today_ms and rows[ms][3] > 0]  # never today's candle
        if len(clean) < 400:
            problems.append(f"{coin} 的日線只有 {len(clean)} 筆，沒有算進去。")
            continue
        stored[symbol] = clean
        frame = pd.DataFrame(clean, columns=["ms", "high", "low", "close"])
        frame.index = pd.to_datetime(frame["ms"], unit="ms").dt.normalize()
        frame = frame[~frame.index.duplicated(keep="last")].drop(columns="ms")
        frames[coin] = frame.asfreq("D").ffill()
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return frames, problems


# ---------- benchmark rules (same as trend_ensemble) ----------

def _blend_signals(close: pd.Series) -> pd.DataFrame:
    cols = {}
    for n in MA_DAYS:
        mean = close.rolling(n, min_periods=n).mean()
        cols[f"A{n}"] = (close > mean).astype(float).where(mean.notna())
    for n in CHANNEL:
        m = n // 2
        high = close.rolling(n, min_periods=n).max()
        low = close.rolling(m, min_periods=m).min()
        state = pd.Series(np.where(close >= high, 1.0, np.where(close <= low, 0.0, np.nan)), index=close.index)
        cols[f"C{n}"] = state.where(high.notna()).ffill().fillna(0.0).where(high.notna())
    for n in PAST:
        past = close / close.shift(n) - 1
        cols[f"D{n}"] = (past > 0).astype(float).where(past.notna())
    return pd.DataFrame(cols)


def _benchmarks(close: pd.Series) -> pd.DataFrame:
    """What the two benchmark rules want to hold, decided at each day's close."""
    signal = _blend_signals(close)
    share = signal.mean(axis=1).where(signal.notna().all(axis=1))
    vol = close.pct_change().rolling(BLEND_VOL_DAYS, min_periods=BLEND_VOL_DAYS).std(ddof=1) * math.sqrt(YEAR)
    scale = (BLEND_TARGET / vol.where(vol > 1e-9)).clip(upper=1.0).fillna(1.0).where(vol.notna())
    return pd.DataFrame({"c20": signal["C20"], "vol40": share * scale, "share17": share, "vol30": vol})


# ---------- S1: channel ensemble ----------

def _channel(close: np.ndarray, n: int, trailing: bool) -> tuple[np.ndarray, np.ndarray]:
    """State at each close (1 long, 0 not) of one n-day closing channel model, and the stop that applies next day.

    trailing=True is the paper's rule: long when the close is the highest of the last n closes; out at the first close
    at or below the stop; the stop starts at the channel's midpoint and never moves down. trailing=False leaves when
    the close is the lowest of the last n closes instead. A model without n closes yet stays in cash."""
    series = pd.Series(close)
    up = series.rolling(n, min_periods=n).max().values
    down = series.rolling(n, min_periods=n).min().values
    state, stops = np.zeros(len(close)), np.full(len(close), np.nan)
    long, stop = False, math.nan
    for i in range(len(close)):
        if not math.isfinite(up[i]):
            continue
        p = close[i]
        if trailing:
            mid = 0.5 * (up[i] + down[i])
            if p >= up[i]:
                stop = max(stop, mid) if long else mid
                long = True
            elif long:
                if p <= stop:
                    long, stop = False, math.nan
                else:
                    stop = max(stop, mid)
        elif p >= up[i]:
            long = True
        elif long and p <= down[i]:
            long = False
        state[i] = 1.0 if long else 0.0
        stops[i] = stop
    return state, stops


def _s1(close: pd.Series, *, trailing: bool = True, target: float | None = TARGET, cap: float = 1.0,
        band: float = 0.0) -> dict:
    """The ensemble's wanted position at each close: share of the nine models long x min(cap, target / volatility).

    target=None holds the plain share. band > 0 keeps the previous scale unless a model switched or the scale moved by
    more than band (relative), as the paper's rebalancing threshold does."""
    values = close.values.astype(float)
    states, stops = {}, {}
    for n in LOOKBACKS:
        states[n], stops[n] = _channel(values, n, trailing)
    share = pd.DataFrame(states, index=close.index).mean(axis=1)
    sigma = close.pct_change().rolling(S1_VOL_DAYS, min_periods=S1_VOL_DAYS).std(ddof=1) * math.sqrt(YEAR)
    if target is None:
        scale = pd.Series(1.0, index=close.index).where(sigma.notna())
    else:
        scale = (target / sigma.where(sigma > 1e-9)).clip(upper=cap).fillna(cap).where(sigma.notna())
    if band > 0:
        want, part = scale.values, share.values
        kept = np.full(len(want), np.nan)
        current = math.nan
        for i in range(len(want)):
            if not math.isfinite(want[i]):
                continue
            switched = i == 0 or part[i] != part[i - 1]
            if not math.isfinite(current) or switched or abs(want[i] - current) > band * current:
                current = want[i]
            kept[i] = current
        scale = pd.Series(kept, index=close.index)
    return {"target": share * scale, "share": share, "scale": scale, "sigma": sigma, "states": states, "stops": stops}


# ---------- S2: Turtle ----------

def _true_range_average(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """N: Faith's 20-day smoothed average of the true range. Without a day's high/low the true range is |close change|."""
    before = close.shift(1)
    true_range = (np.maximum(high, before) - np.minimum(low, before)).where(high.notna() & low.notna())
    true_range = true_range.fillna((close - before).abs())
    return true_range.ewm(alpha=1 / ATR_DAYS, adjust=False, min_periods=ATR_DAYS).mean()


def _turtle(close: np.ndarray, atr: np.ndarray, entry_days: int, exit_days: int, *, sized: bool, units: int,
            stop: float | None, skip: bool, cap: float | None, cost: float = COST, borrow: float = BORROW) -> dict:
    """Simulate a long-only Turtle system on closes with real holdings (coins and cash).

    Decisions at the close of day i are traded at that close; `position` and `trade` are returned already moved to
    the day the position is held (i+1), and the cost of a trade lands in the return of that day, as in _run."""
    n = len(close)
    series = pd.Series(close)
    high = series.rolling(entry_days, min_periods=entry_days).max().values
    low = series.rolling(exit_days, min_periods=exit_days).min().values
    net, after, traded = np.zeros(n), np.zeros(n), np.zeros(n)
    cash, coins, held, last_entry, entry_n = 1.0, 0.0, 0, math.nan, math.nan
    hyp_open, hyp_entry, hyp_stop, last_winner = False, math.nan, -math.inf, False
    equity_before = 1.0
    for i in range(n):
        p = close[i]
        if i > 0 and coins > 0:
            owed = max(-cash, 0.0)
            cash -= owed * borrow / YEAR  # interest on the borrowed part for the day just held
        equity = cash + coins * p
        net[i] = equity / equity_before - 1 if i > 0 else 0.0
        equity_before = equity
        if not (math.isfinite(high[i]) and math.isfinite(low[i]) and math.isfinite(atr[i]) and atr[i] > 0):
            continue
        is_high, is_low = p >= high[i], p <= low[i] and not p >= high[i]
        signal = False  # a fresh breakout signal
        if skip:
            if hyp_open:
                if is_low or p <= hyp_stop:
                    last_winner, hyp_open = p > hyp_entry, False
            elif is_high:
                signal = True
        elif is_high:
            signal = True
        amount = 0.0
        if held > 0:
            if is_low or (stop is not None and p <= last_entry - stop * entry_n):
                amount = -coins * p
                held, last_entry, entry_n = 0, math.nan, math.nan
            elif sized and held < units and p >= last_entry + ADD_STEP * entry_n:
                amount = RISK * equity / atr[i] * p
                if cap is not None:
                    amount = min(amount, cap * equity - coins * p)
                if amount > 1e-9 * equity:
                    held, last_entry, entry_n = held + 1, p, atr[i]
                else:
                    amount = 0.0
        elif signal and not (skip and last_winner):
            amount = RISK * equity / atr[i] * p if sized else equity
            if cap is not None:
                amount = min(amount, cap * equity)
            held, last_entry, entry_n = 1, p, atr[i]
        if skip and signal:
            hyp_open, hyp_entry = True, p
            hyp_stop = p - stop * atr[i] if stop is not None else -math.inf
        if amount != 0.0:
            coins = 0.0 if amount < 0 else coins + amount / p
            cash -= amount + cost * abs(amount)
            traded[i] = abs(amount) / equity
        after[i] = coins * p / equity
    position = np.concatenate([[0.0], after[:-1]])
    trade = np.concatenate([[0.0], traded[:-1]])
    state = {"units": held, "weight": float(after[-1]), "last_entry": last_entry, "entry_n": entry_n,
             "stop": last_entry - stop * entry_n if held > 0 and stop is not None else math.nan,
             "skip_next": bool(skip and last_winner and not hyp_open), "hyp_open": bool(hyp_open),
             "enter_at": float(np.max(close[-(entry_days - 1):])), "exit_at": float(np.min(close[-(exit_days - 1):]))}
    return {"net": net, "position": position, "trade": trade, "state": state}


# ---------- simulating and measuring ----------

def _run(position: np.ndarray, change: np.ndarray, cost: float, borrow: float = BORROW) -> tuple[np.ndarray, np.ndarray]:
    """Daily net return and amount traded for a position held during each day and reset to its target every close.

    After a day's move a position of w has drifted to w(1+r)/(1+wr); the trade is the gap to the next target. The
    part of a position above 100% is borrowed and pays `borrow` a year."""
    position = np.nan_to_num(position, nan=0.0)
    change = np.nan_to_num(change, nan=0.0)
    base = 1 + position * change
    drift = position * (1 + change) / np.where(np.abs(base) > 1e-9, base, 1.0)
    trade = np.abs(position - np.concatenate([[0.0], drift[:-1]]))  # starts from cash
    return position * change - cost * trade - np.maximum(position - 1, 0.0) * borrow / YEAR, trade


def _run_many(weights: np.ndarray, change: np.ndarray, cost: float, borrow: float = BORROW):
    """The same for several coins at once: weights (days x coins) are shares of the whole portfolio."""
    weights = np.nan_to_num(weights, nan=0.0)
    change = np.nan_to_num(change, nan=0.0)
    gross = (weights * change).sum(axis=1)
    base = np.where(np.abs(1 + gross) > 1e-9, 1 + gross, 1.0)
    drift = weights * (1 + change) / base[:, None]
    before = np.vstack([np.zeros((1, weights.shape[1])), drift[:-1]])
    trade = np.abs(weights - before).sum(axis=1)
    total = weights.sum(axis=1)
    return gross - cost * trade - np.maximum(total - 1, 0.0) * borrow / YEAR, trade, total


def _measure(net: pd.Series, position: pd.Series, trade: pd.Series, bench: pd.Series | None = None) -> dict:
    values = net.values
    n = len(values)
    growth = float(np.prod(1 + values))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 and n > 1 else None
    vol = float(np.std(values, ddof=1) * math.sqrt(YEAR) * 100) if n > 1 else 0.0
    ratio = ret / vol if ret is not None and vol > 1e-9 else 0.0
    equity = np.concatenate([[1.0], np.cumprod(1 + values)])
    peak = np.maximum.accumulate(equity)
    deepest = float((equity / peak - 1).min() * 100)
    longest, since, open_end = 0, 0, False
    for i in range(1, len(equity)):
        if equity[i] < peak[i] - 1e-12:
            if i - since > longest:
                longest, open_end = i - since, i == len(equity) - 1
            elif i == len(equity) - 1 and i - since == longest:
                open_end = True
        else:
            since = i
    by_year = {int(y): float((np.prod(1 + chunk.values) - 1) * 100) for y, chunk in net.groupby(net.index.year)}
    worst = min(by_year, key=lambda y: by_year[y])
    out = {"days": n, "ret": ret, "tot": (growth - 1) * 100, "vol": vol, "ratio": ratio, "deepest": deepest,
           "under": int(longest), "under_open": bool(open_end), "years": by_year, "worst": worst,
           "share": float(position.mean() * 100), "turnover": float(trade.sum() / (n / YEAR)),
           "alpha": None, "beta": None}
    if bench is not None and n > 30:
        b = bench.values
        var = float(np.var(b, ddof=1))
        if var > 1e-12:
            beta = float(np.cov(values, b, ddof=1)[0, 1] / var)
            out["beta"], out["alpha"] = beta, float((values.mean() - beta * b.mean()) * YEAR * 100)
    return out


def _btc_wants(close: pd.Series) -> pd.DataFrame:
    """Every weight-based bitcoin variant's wanted position, decided at each day's close (before any shift)."""
    out = _benchmarks(close)[["c20", "vol40"]].copy()
    out["a_break"] = _s1(close, trailing=False, target=None)["target"]
    out["a_trail"] = _s1(close, trailing=True, target=None)["target"]
    out["s1x1"] = _s1(close, cap=1.0)["target"]
    out["s1x2"] = _s1(close, cap=2.0)["target"]
    out["s1band"] = _s1(close, cap=1.0, band=BAND)["target"]
    out["s1t15"] = _s1(close, target=0.15, cap=1.0)["target"]
    out["s1t40"] = _s1(close, target=0.40, cap=1.0)["target"]
    return out


def _btc_turtles(close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
    atr = _true_range_average(close, high, low).values
    return {key: _turtle(close.values.astype(float), atr, days_in, days_out, **options)
            for key, (days_in, days_out, options) in TURTLE.items()}


def _btc_positions(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """The position each bitcoin variant holds during every day (used by the look-ahead check and the tables)."""
    held = _btc_wants(close).shift(1)
    for key, result in _btc_turtles(close, high, low).items():
        held[key] = result["position"]
    return held


def _lookahead_check(close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
    """Change every price after a date at random and compare the positions up to and including the next day."""
    base = _btc_positions(close, high, low)
    rng = np.random.default_rng(SEED)
    passed, moved, cells = 0, 0, 0
    for day in CHECK_DATES:
        cut = pd.Timestamp(day)
        after = np.asarray(close.index > cut)
        shock = np.where(after, np.exp(np.cumsum(np.where(after, rng.normal(0, 0.06, len(close)), 0.0))), 1.0)
        other = _btc_positions(close * shock, high * shock, low * shock)
        keep = np.asarray(close.index <= cut + pd.Timedelta(days=1))
        same = np.allclose(base.values[keep], other.values[keep], rtol=0, atol=1e-10, equal_nan=True)
        later = not np.allclose(base.values[~keep], other.values[~keep], rtol=0, atol=1e-10, equal_nan=True)
        passed += int(same)
        moved += int(later)
        cells += int(keep.sum()) * base.shape[1]
    return {"dates": len(CHECK_DATES), "passed": passed, "moved": moved, "cells": cells, "variants": base.shape[1]}


def _compute(close: pd.Series, frames: dict) -> dict:
    """Everything that depends on the data; study() only formats it."""
    index, last = close.index, close.index[-1]
    change = close.pct_change()
    btc = frames.get("BTC")
    high = btc["high"].reindex(index) if btc is not None else pd.Series(np.nan, index=index)
    low = btc["low"].reindex(index) if btc is not None else pd.Series(np.nan, index=index)
    # keep the candle's range only where the stored close agrees with the long series (same exchange, same day)
    if btc is not None:
        agree = (btc["close"].reindex(index) / close - 1).abs() < 1e-6
        high, low = high.where(agree), low.where(agree)
    wants = _btc_wants(close)
    turtles = _btc_turtles(close, high, low)
    hold_net = change.fillna(0.0)

    def weight_run(key: str, delay: int = 1, cost: float = COST, borrow: float = BORROW):
        pos = wants[key].shift(delay)
        net, trade = _run(pos.values, change.values, cost, borrow)
        return pd.Series(net, index=index), pos.fillna(0.0), pd.Series(trade, index=index)

    runs = {"hold": (hold_net, pd.Series(1.0, index=index), pd.Series(0.0, index=index))}
    for key in wants.columns:
        runs[key] = weight_run(key)
    for key, result in turtles.items():
        runs[key] = tuple(pd.Series(result[k], index=index) for k in ("net", "position", "trade"))
    runs["s1_delay"] = weight_run("s1x1", delay=2)
    runs["s1_c30"] = weight_run("s1x1", cost=0.003)
    runs["s1_c50"] = weight_run("s1x1", cost=0.005)
    runs["s1_gross"] = weight_run("s1x2", cost=0.0, borrow=0.0)  # the paper's table 1: no costs, no interest

    def measure(parts, start: str, end: str | None = None) -> dict:
        rows = (index >= pd.Timestamp(start)) & (index <= (pd.Timestamp(end) if end else last))
        m = _measure(parts[0][rows], parts[1][rows], parts[2][rows], hold_net[rows])
        hold_years = _measure(hold_net[rows], runs["hold"][1][rows], runs["hold"][2][rows])["years"]
        m["beat"] = sum(1 for y in hold_years if m["years"].get(y, 0.0) > hold_years[y])
        m["n_years"] = len(hold_years)
        return m

    main = {key: measure(parts, MAIN_START) for key, parts in runs.items()}
    periods = {
        "full": (FULL_START, None), "sample": (FULL_START, SAMPLE_END), "p1": (FULL_START, "2017-12-31"),
        "p2": ("2018-01-01", "2021-12-31"), "p3": ("2022-01-01", None), "post": (POST_START, None),
    }
    by_period = {name: {key: measure(runs[key], *span) for key in ("hold", "c20", "vol40", "s1x1", "s1x2", "s1_gross")}
                 for name, span in periods.items()}

    # ----- several coins from 2018 -----
    names = [c for c in COINS if c in frames]
    prices = pd.DataFrame({c: (close if c == "BTC" else frames[c]["close"].reindex(index)) for c in names})
    coin_change = prices.pct_change()
    want1, want2, ready = {}, {}, {}
    for c in names:
        own = prices[c].dropna()
        one, two = _s1(own, cap=1.0), _s1(own, cap=2.0)
        want1[c], want2[c] = one["target"].reindex(index), two["target"].reindex(index)
        ready[c] = one["sigma"].notna().reindex(index, fill_value=False)
    ready = pd.DataFrame(ready)
    count = ready.sum(axis=1).where(lambda s: s > 0)
    many = {}
    for key, want in (("m_x1", pd.DataFrame(want1)), ("m_x2", pd.DataFrame(want2)), ("m_hold", ready.astype(float))):
        weights = want.where(ready).div(count, axis=0).shift(1)
        net, trade, total = _run_many(weights.values, coin_change.values, COST)
        many[key] = (pd.Series(net, index=index), pd.Series(total, index=index), pd.Series(trade, index=index))
        main[key] = measure(many[key], MAIN_START)
    start_row = ready.loc[pd.Timestamp(MAIN_START)] if pd.Timestamp(MAIN_START) in ready.index else ready.iloc[0]
    coins = {"names": names, "at_start": [c for c in names if bool(start_row[c])],
             "first": {c: _d(ready.index[ready[c].values.argmax()]) for c in names if ready[c].any()}}

    s1_now = {"x1": _s1(close, cap=1.0), "x2": _s1(close, cap=2.0)}
    since = s1_now["x1"]["sigma"][index >= pd.Timestamp(FULL_START)]
    calm = {"days": int((since < TARGET).sum()), "of": int(len(since)),
            "main": int((since[since.index >= pd.Timestamp(MAIN_START)] < TARGET).sum())}
    return {"last": last, "close": close, "main": main, "periods": by_period, "wants": wants, "turtles": turtles,
            "s1": s1_now, "coins": coins, "calm": calm, "range_from": _d(high.first_valid_index()) if high.notna().any() else None,
            "check": _lookahead_check(close, high, low), "bench": _benchmarks(close)}


# ---------- the study ----------

def _under(m: dict) -> str:
    return f"{m['under']:,} 天" + ("（期末還沒回到）" if m["under_open"] else "")


def _held(m: dict) -> str:
    return f"{m['share']:.0f}%｜{m['turnover']:.1f} 次"


def _three(m: dict) -> str:
    return f"每年 {_pct(m['ret'])}、報酬÷波動 {_num(m['ratio'])}、最深虧 {_pct(m['deepest'])}"


def _step(new: dict, old: dict) -> str:
    """Plain words for what changed from `old` to `new`."""
    def word(a, b, tol, up, down):
        return "幾乎沒變" if abs(a - b) <= tol else (up if a > b else down)
    return (f"每年報酬 {_pct(old['ret'])}→{_pct(new['ret'])}、"
            f"報酬÷波動 {_num(old['ratio'])}→{_num(new['ratio'])}（{word(new['ratio'], old['ratio'], 0.03, '變高', '變低')}）、"
            f"最深虧 {_pct(old['deepest'])}→{_pct(new['deepest'])}（{word(new['deepest'], old['deepest'], 1.0, '變淺', '變深')}）")


def _versus(new: dict, old: dict) -> str:
    """One short judgement of `new` against `old` on return ÷ volatility and deepest loss."""
    better_ratio, worse_ratio = new["ratio"] > old["ratio"] + 0.05, new["ratio"] < old["ratio"] - 0.05
    shallower, deeper = new["deepest"] > old["deepest"] + 2, new["deepest"] < old["deepest"] - 2
    more, less = new["ret"] > old["ret"] + 1, new["ret"] < old["ret"] - 1
    parts = ["報酬÷波動" + ("比較高" if better_ratio else "比較低" if worse_ratio else "差不多"),
             "最深虧損" + ("比較淺" if shallower else "比較深" if deeper else "差不多"),
             "每年報酬" + ("比較多" if more else "比較少" if less else "差不多")]
    return "、".join(parts)


def study() -> dict:
    close = _daily_close()
    frames, problems = _klines()
    r = _compute(close, frames)
    main, per, last = r["main"], r["periods"], r["last"]
    years = sorted(main["hold"]["years"])
    n = len(years)
    partial = last < pd.Timestamp(f"{years[-1]}-12-31")
    span = f"{MAIN_START[:7]} 到 {_d(last)}"
    names = r["coins"]["names"]

    def row(name: str, m: dict, hold: bool = False) -> list:
        wins = "—" if hold else f"{m['n_years']} 年裡 {m['beat']} 年"
        return [name, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]), _under(m),
                f"{m['worst']} 年 {_pct(m['years'][m['worst']])}｜{wins}", _held(m)]

    wide = ["做法", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "最久多久沒回到高點",
            "最差的一年｜賺得比一直抱著比特幣多的年數", "平均持有｜一年換手幾次"]

    # ----- table 1: everything, 2018 to now -----
    rows1 = [
        row("一直抱著比特幣", main["hold"], hold=True),
        row("已驗證的簡單規則：創 20 日新高進場、10 日新低出場", main["c20"]),
        row("已驗證：17 條順勢規則混合 × 依波動調整（目標 40%）", main["vol40"]),
        row("論文做法（九條新高規則＋移動停損＋目標波動 25%），不借錢", main["s1x1"]),
        row(f"論文做法，最多借到 2 倍（借的部分年息 {BORROW * 100:.0f}%）", main["s1x2"]),
        row("海龜系統二（55 日新高進、20 日新低出＋分份、加碼、停損）", main["t2_full"]),
        row("海龜系統一（20 日新高進、10 日新低出＋上次賺就跳過＋分份、加碼、停損）", main["t1_full"]),
        row(f"{len(names)} 種幣平均分配，各自照論文做法，不借錢", main["m_x1"]),
        row(f"{len(names)} 種幣平均分配，各自照論文做法，最多 2 倍", main["m_x2"]),
        row(f"對照：同一批 {len(names)} 種幣平均分配、一直抱著（每天調回平均）", main["m_hold"]),
    ]

    # ----- table 2: S1 on bitcoin by period -----
    labels = [("full", f"{FULL_START[:7]} 到 {_d(last)}（全部）"), ("sample", f"論文樣本：{FULL_START[:7]} 到 {SAMPLE_END}"),
              ("p1", "2015–2017"), ("p2", "2018–2021"), ("p3", f"2022 到 {_d(last)}"),
              ("post", f"論文樣本之後：{POST_START[:7]} 到 {_d(last)}")]
    rows2 = []
    for key, label in labels:
        p = per[key]
        m, m2x = p["s1x1"], p["s1x2"]
        rows2.append([label, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]),
                      f"{_pct(m2x['ret'])}｜{_pct(m2x['deepest'])}",
                      f"{_pct(p['hold']['ret'])}｜{_pct(p['hold']['deepest'])}",
                      f"{_pct(p['c20']['ret'])}｜{_num(p['c20']['ratio'])}｜{_pct(p['c20']['deepest'])}"])
    g = per["sample"]["s1_gross"]
    rows2.append(["論文樣本，照論文表 1 的算法（最多 2 倍、不扣成本、不算利息）", _pct(g["ret"]),
                  _pct(g["vol"], signed=False), _num(g["ratio"]), _pct(g["deepest"]), "—",
                  f"{_pct(per['sample']['hold']['ret'])}｜{_pct(per['sample']['hold']['deepest'])}",
                  f"{_pct(per['sample']['c20']['ret'])}｜{_num(per['sample']['c20']['ratio'])}｜"
                  f"{_pct(per['sample']['c20']['deepest'])}"])

    # ----- table 3: what each part adds, and robustness -----
    def row3(name: str, m: dict) -> list:
        return [name, _pct(m["ret"]), _pct(m["vol"], signed=False), _num(m["ratio"]), _pct(m["deepest"]), _under(m),
                f"{m['share']:.0f}%", f"{m['turnover']:.1f} 次"]

    rows3 = [
        row3("拆解 1：只用九條新高新低（跌到同一段期間的最低收盤才出場）", main["a_break"]),
        row3("拆解 2：出場改成移動停損", main["a_trail"]),
        row3("拆解 3：再依波動調整持有多少（目標 25%，不借錢）＝論文做法不借錢版", main["s1x1"]),
        row3(f"拆解 4：允許借到 2 倍（借的部分年息 {BORROW * 100:.0f}%）", main["s1x2"]),
        row3("改動：晚一天才照做（訊號後第 2 天）", main["s1_delay"]),
        row3("改動：成本改成 0.30%", main["s1_c30"]),
        row3("改動：成本改成 0.50%", main["s1_c50"]),
        row3("改動：目標波動改成 15%", main["s1t15"]),
        row3("改動：目標波動改成 40%", main["s1t40"]),
        row3("改動：波動造成的調整不到兩成就不動（論文扣成本時的做法）", main["s1band"]),
    ]

    # ----- table 4: Turtle with and without its extras -----
    rows4 = [
        row("系統二・最簡單：55 日新高全部買進、20 日新低全部賣出", main["t2_plain"]),
        row("系統二・改成只買一份（一份＝價格動 1N 時輸贏是資金的 1%）", main["t2_unit"]),
        row("系統二・一份＋跌破進場價下方 2N 就停損", main["t2_stop"]),
        row("系統二・完整：每漲 0.5N 加碼一份、最多 4 份、停損跟著上移（不借錢）", main["t2_full"]),
        row(f"系統二・完整但不設 100% 上限（超過的部分借錢、年息 {BORROW * 100:.0f}%）", main["t2_free"]),
        row("系統一・最簡單：20 日新高全部買進、10 日新低全部賣出", main["t1_plain"]),
        row("系統一・全部買進＋上一次訊號是賺的就跳過這一次", main["t1_skip"]),
        row("系統一・分份、加碼、停損，但不跳過", main["t1_noskip"]),
        row("系統一・完整：分份、加碼、停損＋跳過", main["t1_full"]),
        row("對照：一直抱著比特幣", main["hold"], hold=True),
    ]

    start_names = "、".join(r["coins"]["at_start"]) or "—"
    tables = [
        {"caption": (f"總表（{span}，{n} 個年度{'，最後一年還沒過完' if partial else ''}）。每種做法都是當天收盤決定、隔天才照做，"
                     f"每次買賣扣交易金額的 {COST * 100:.2f}%，現金沒有利息；一直抱著比特幣沒有扣成本。前七列只做比特幣；"
                     f"後三列是固定名單的 {len(names)} 種幣（{'、'.join(names)}）平均分配，這份名單是用今天還在的大幣回頭挑的"
                     f"（有倖存者偏誤），不是論文每月換名單的前 20 大，數字不能直接和論文比；{MAIN_START[:4]} 年初名單裡只有"
                     f" {start_names} 有足夠資料，其他幣上架滿 {S1_VOL_DAYS} 天後才加入。"
                     "最後一欄是「平均持有比例｜一年換手幾次」（換手 1 次＝一年內買賣的總金額等於全部資金一次）。"),
         "columns": wide, "rows": rows1},
        {"caption": ("論文做法只做比特幣，分段來看（扣成本後；最多 2 倍的版本另外扣借錢利息）。兩個版本幾乎一樣，因為"
                     f"「25% ÷ 最近 90 天的波動」很少超過 1：{FULL_START[:4]} 年以來 {r['calm']['of']:,} 天裡只有 "
                     f"{r['calm']['days']:,} 天比特幣的波動低於 25%（{MAIN_START[:4]} 年以來 {r['calm']['main']:,} 天），"
                     "其他日子根本用不到借錢。最後一列是照論文表 1 的算法"
                     "（不扣成本、不算利息）重做論文樣本，論文自己列的是每年 +30%、每年波動 17%、最深虧 −19%。"
                     f"論文樣本到 {SAMPLE_END} 為止，「論文樣本之後」那一段是論文作者定規則時看不到的資料，但只有 "
                     f"{per['post']['hold']['days']:,} 天，那一段的「每年報酬」是把這段的結果換算成一年。"
                     "右邊兩欄是同一段期間的對照。"),
         "columns": ["期間", "不借錢版：每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少",
                     "最多 2 倍版：每年報酬｜最深虧", "同期一直抱著：每年報酬｜最深虧",
                     "同期 20 日新高／10 日新低：每年報酬｜報酬÷波動｜最深虧"],
         "rows": rows2},
        {"caption": (f"論文做法一層一層加上去各多了什麼，以及改動條件後的結果（只做比特幣，{span}）。「拆解」四列是累加的；"
                     "「改動」各列都是以拆解 3（不借錢版、成本 0.10%、隔天照做、目標波動 25%）為基準只改一項。"),
         "columns": ["版本", "每年報酬", "每年波動", "報酬÷波動", "最深曾經虧多少", "最久多久沒回到高點", "平均持有",
                     "一年換手幾次"],
         "rows": rows3},
        {"caption": (f"海龜規則加與不加各種配件的差別（只做比特幣、只做多，{span}）。N 是最近 20 天每天價格活動範圍的平均；"
                     "新高新低和停損都只看收盤價。除了標明的那一列，持有金額最多就是全部資金、不借錢。"),
         "columns": wide, "rows": rows4},
    ]

    # ----- today -----
    price = float(close.iloc[-1])
    s1 = r["s1"]["x1"]
    long_models = [k for k in LOOKBACKS if s1["states"][k][-1] == 1]
    flat_models = [k for k in LOOKBACKS if s1["states"][k][-1] != 1]
    values = close.values
    triggers = {k: float(values[-(k - 1):].max()) for k in flat_models}
    stops = {k: float(s1["stops"][k][-1]) for k in long_models}
    if triggers:
        k = min(triggers, key=lambda key: triggers[key])
        switch_on = (f"{k} 日那一條：下一根日線收在 {_lvl(triggers[k])}以上"
                     f"（比現在{'高' if triggers[k] >= price else '低'} {abs(triggers[k] / price - 1) * 100:.1f}%）")
    else:
        switch_on = "九條都已經是持有狀態"
    if stops:
        k = max(stops, key=lambda key: stops[key])
        switch_off = f"{k} 日那一條：收在 {_lvl(stops[k])}以下（比現在低 {abs(1 - stops[k] / price) * 100:.1f}%）"
    else:
        switch_off = "九條都不在持有狀態"
    sigma = float(s1["sigma"].iloc[-1] * 100)

    def stake(v) -> str:
        if not _ok(v):
            return "—"
        return f"持有比特幣 {float(v) * 100:.0f}%" if float(v) > 0.0005 else "全部現金"

    def turtle_state(key: str) -> str:
        t = r["turtles"][key]["state"]
        if t["units"] > 0:
            text = (f"持有 {t['units']} 份（約資金的 {t['weight'] * 100:.0f}%）；收盤跌到 {_lvl(max(t['exit_at'], t['stop']))}"
                    "以下就全部出場")
            return text
        text = f"空手；下一根日線收在 {_lvl(t['enter_at'])}以上才算新高"
        if t["skip_next"]:
            text += "，但上一次訊號是賺的，照規則這一次會跳過"
        elif t["hyp_open"]:
            text += "（目前這一輪訊號已經被跳過，要等它結束後的下一次新高）"
        return text

    bench_now = r["bench"].iloc[-1]
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {_usd(price)}"},
        {"label": "論文做法：九條新高規則裡目前是持有狀態的",
         "value": f"{len(long_models)} 條" + (f"（{'、'.join(map(str, long_models))} 日）" if long_models else "")},
        {"label": f"最近 {S1_VOL_DAYS} 天的波動（換算成一年）", "value": _pct(sigma, signed=False)},
        {"label": "論文做法照規則下一天的狀態",
         "value": f"不借錢：{stake(r['wants']['s1x1'].iloc[-1])}；最多 2 倍：{stake(r['wants']['s1x2'].iloc[-1])}"},
        {"label": "離得最近、會讓一條規則轉成持有的收盤價", "value": switch_on},
        {"label": "離得最近、會讓一條規則出場的移動停損", "value": switch_off},
        {"label": "海龜系統二（55 日／20 日）目前的狀態", "value": turtle_state("t2_full")},
        {"label": "海龜系統一（20 日／10 日，會跳過）目前的狀態", "value": turtle_state("t1_full")},
        {"label": "對照的兩條已驗證規則下一天的狀態",
         "value": f"20 日新高／10 日新低：{stake(bench_now['c20'])}；17 條混合×波動 40%：{stake(bench_now['vol40'])}"},
    ]

    # ----- verdict -----
    m1, m2, c, v, h = main["s1x1"], main["s1x2"], main["c20"], main["vol40"], main["hold"]
    post, same = per["post"], main["s1t40"]
    close_copy = abs(g["ret"] - 30) <= 5 and abs(g["deepest"] + 19) <= 6
    twin = abs(m2["ret"] - m1["ret"]) < 0.3 and abs(m2["deepest"] - m1["deepest"]) < 0.5
    adds = same["ratio"] > v["ratio"] + 0.05 or (same["ratio"] > v["ratio"] - 0.05 and same["ret"] > v["ret"] - 1
                                                and same["deepest"] > v["deepest"] + 2)
    t2, t2p, t1, t1p, t1s = main["t2_full"], main["t2_plain"], main["t1_full"], main["t1_plain"], main["t1_skip"]
    verdict = (
        f"照論文的規則重做，論文樣本（{FULL_START[:7]} 到 {SAMPLE_END}、不扣成本）只做比特幣得到每年 {_pct(g['ret'])}、"
        f"最深虧 {_pct(g['deepest'])}（論文列 +30%、−19%），{'大致重現' if close_copy else '和論文有明顯差距'}；"
        f"論文樣本之後（{POST_START[:7]} 到 {_d(last)}）它累計 {_pct(post['s1x1']['tot'])}，同期一直抱著比特幣 "
        f"{_pct(post['hold']['tot'])}、已驗證的「20 日新高進、10 日新低出」{_pct(post['c20']['tot'])}。"
        f"{years[0]} 到 {_d(last)} 扣成本後，論文做法是{_three(m1)}、{n} 年裡 {m1['beat']} 年賺得比一直抱著多"
        + ("（最多借到 2 倍的版本幾乎一樣，因為比特幣的波動很少低到用得上借錢）" if twin
           else f"（最多借到 2 倍的版本是{_three(m2)}）") +
        f"，已驗證的 20／10 規則是{_three(c)}，17 條混合×波動 40% 是{_three(v)}，一直抱著是{_three(h)}；拆開看，移動停損讓"
        f"{_step(main['a_trail'], main['a_break'])}，依波動調整持有多少再讓{_step(m1, main['a_trail'])}"
        f"，但平均只持有 {m1['share']:.0f}%，虧得淺主要是因為拿得少；把目標波動調成同樣的 40% 來比是{_three(same)}，"
        f"和已驗證的混合比{_versus(same, v)}，"
        + ("算是多出了一點東西。" if adds else "沒有多出已驗證規則沒有的東西。") +
        f"海龜系統二加上分份、加碼、停損是{_three(t2)}，同樣進出但一次全部買進是{_three(t2p)}，加了配件之後{_versus(t2, t2p)}；"
        f"系統一完整版是{_three(t1)}，最簡單的 20／10 是{_three(t1p)}，只加「上次賺就跳過」是{_three(t1s)}"
        f"（和不跳過比{_versus(t1s, t1p)}）。"
    )

    check = r["check"]
    check_ok = check["passed"] == check["dates"]
    notes = [
        "論文做法的細節這次有對照論文原文（PDF 可以讀到）：通道用的是收盤價，「今天的收盤是最近 N 天（含今天）最高的收盤」就進場；"
        "移動停損＝「前一天的停損」和「通道中線（N 天最高和最低收盤的中間）」取比較高的那個，剛進場時就是當時的中線，"
        "收盤等於或低於停損就出場；九條（5、10、20、30、60、90、150、250、360 日）平均；持有量＝25% ÷ 最近 90 天的波動，最多 2 倍。"
        "和論文不一樣的地方：論文用 CoinMarketCap 的價格，這裡用 blockchain.info 和幣安；論文扣成本時另外加了"
        "「波動造成的調整不到兩成就不動」，這裡主要結果是每天調整，那個做法放在第三張表當一列（「兩成」這裡當成相對於原本"
        f"比例的兩成，論文沒有說得更細）；論文沒有算借錢的利息，這裡對超過 100% 的部分扣年息 {BORROW * 100:.0f}%；"
        "波動換算成一年用 365 天。",
        "論文宣稱的「報酬÷波動 1.5 以上、每年比比特幣多 10.8%」是每月換名單的前 20 大流動性幣種組合、2015 到 2025-03，"
        "而且「多 10.8%」是扣掉跟著比特幣漲跌的部分之後算出來的，不是每年報酬直接比比特幣高 10.8%（論文那個組合每年報酬約 18%）。"
        f"照同樣的算法，論文樣本內只做比特幣、不扣成本的版本，我們算出每年多 {_pct(g['alpha'])}、跟漲跌的程度 {_num(g['beta'])}"
        "（論文列 14%、0.17）。本頁的「報酬÷波動」是每年複利報酬除以每年波動，和論文用的算法不完全一樣。",
        f"多幣版用的是固定名單（{'、'.join(names)}），全部是今天還在、而且成交量大的幣，中途消失或沒落的幣不在名單裡，"
        "所以有倖存者偏誤，結果會比當時真的能做到的好看；它也不是論文每月依成交量換名單的做法，兩邊的數字不是同一件事。"
        "每種幣從幣安上架那天開始算，滿 90 天有波動數字後加入、平均分配給當時已加入的幣；還沒有足夠天數的長期規則先當成不持有。"
        "「年數」那一欄比的是一直抱著比特幣。",
        "海龜規則照 Curtis Faith 書裡的寫法：N 是每天真實波幅的 20 日平滑平均；一份的大小＝資金的 1% ÷ N；每比上一次進場價"
        "高 0.5N 加一份，最多 4 份；所有部位的停損是最後一次進場價下方 2N；系統一在上一次訊號（不管有沒有真的進場）是賺的時候"
        "跳過這一次。這裡的簡化：新高、新低、停損、加碼都只在每天收盤判斷一次（原版是盤中價格碰到就做），只做多；"
        "系統一原版在跳過之後還有「創 55 日新高就一定進場」的保險，這裡沒有做；主要版本把持有金額限制在全部資金以內。"
        + (f"每天的最高最低價來自幣安（{r['range_from']} 起），更早的日子沒有最高最低價，用「收盤到收盤的變動」代替真實波幅，"
           "會把 N 估得偏小；第四張表的期間都在有最高最低價之後。" if r["range_from"]
           else "這次取不到幣安的每日最高最低價，N 全部用「收盤到收盤的變動」代替，會把 N 估得偏小。"),
        f"有沒有偷看未來的檢查（這次執行時真的跑了）：在 {check['dates']} 個日期（{'、'.join(CHECK_DATES)}）之後把所有價格"
        f"隨機改掉，重新算 {check['variants']} 種比特幣做法的每日部位，比對「那一天和隔天為止」的部位共 {check['cells']:,} 格："
        + (f"{check['dates']} 次全部一模一樣，而之後的部位有 {check['moved']} 次跟著改變（表示這個檢查確實有作用）。"
           if check_ok else f"只有 {check['passed']} 次一模一樣，表示有地方用到了未來的資料，表裡的數字不能相信。")
        + "多幣版用的是同一套計算。",
        "這兩套都是公開發表的規則，作者定規則的時候已經看過過去的資料：論文的九個天數、25% 的目標、移動停損的寫法，"
        "都是在知道 2015–2025 年走勢之後定的；海龜規則雖然是 1980 年代的，拿來套在比特幣上、選哪個版本也是事後的決定。"
        "所以樣本內的數字一定偏好看，樣本之後的那一小段才比較接近事前的檢驗，但它太短。",
        f"成本很粗略：每次買賣一律扣交易金額的 {COST * 100:.2f}%，實際的手續費、價差和滑價會隨交易所、金額和年代不同，"
        f"小幣和早年會貴得多；借錢一律算年息 {BORROW * 100:.0f}%，實際的借款利率或合約資金費率起伏很大，也沒有算被強制平倉的風險；"
        "現金假設沒有利息，也沒有算稅。",
        "所有停損都只在每天收盤判斷一次，盤中跌得多深都不會提早出場；真的在盤中掛停損單，結果會不一樣（可能更好也可能更差），"
        "而且這裡的「最深曾經虧多少」也是用每天收盤算的，盤中實際的虧損會更深。",
        f"主要期間只有 {n} 個年度，而且這些年不是彼此獨立的（大致就是兩三輪大漲大跌）；「{n} 年裡幾年」這種數字多一年少一年就會差很多，"
        "做法之間的小差距不能當成確定的結論。各分段的起點都沿用前面已經在跑的部位，沒有每段重新開始。",
        "價格來源：2017-08 以前是 blockchain.info 的每日比特幣價格（多家交易所的平均），之後是幣安現貨日線（UTC 0 點收盤）；"
        "其他幣只用幣安現貨對 USDT 的日線。只用已經收完的日線。",
        "過去的結果不保證未來會一樣。這裡只是整理這些規則過去做了什麼、現在處在什麼狀態，不是買賣建議。",
    ] + problems

    return {
        "id": "donchian_ensemble",
        "title": "兩套有名的突破系統：多出來的配件有沒有用",
        "question": ("論文〈Catching Crypto Trends〉的「九條新高規則＋移動停損＋依波動調整」和海龜交易法的「分份、加碼、停損、跳過」，"
                     "比起已經驗證過的簡單規則（創 20 日新高進場、10 日新低出場，以及 17 條規則混合）有沒有多出什麼？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Zarattini、Pagani、Barbon〈Catching Crypto Trends〉論文全文（Concretum）",
             "url": "https://concretumgroup.com/wp-content/uploads/2026/02/Catching-Crypto-Trends.pdf"},
            {"name": "Concretum 的論文摘要頁",
             "url": "https://concretumgroup.com/catching-crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/"},
            {"name": "海龜交易實驗（Richard Dennis，維基百科；規則依 Curtis Faith《Way of the Turtle》）",
             "url": "https://en.wikipedia.org/wiki/Richard_Dennis"},
            {"name": "blockchain.info 每日比特幣價格（2010 年起）",
             "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
