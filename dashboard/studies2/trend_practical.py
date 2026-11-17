"""Historical study "trend_practical": how well does the bitcoin rule "long at a 20-day closing high, cash at a
10-day closing low" survive being followed the way a real person would follow it?

The textbook backtest assumes somebody looks at the exact daily close (00:00 UTC = 08:00 in Taiwan) every single day
and trades at that price for 0.10%. Here every one of those assumptions is relaxed in turn:

1. which hour is called "the close" (daily bars rebuilt from hourly data, the day ending at each of the 24 hours);
2. checking less often (every 2 or 3 days, once a week on each weekday, weekly bars with a 4-week high / 2-week low);
3. acting late (12 hours, 1, 2, 3, 5 days after the signal);
4. missing signals (random skipped entries or exits, 1,000 simulations each; skipping after two losses in a row);
5. hesitation (only buying back when the price is not above the last selling price);
6. higher costs (0.10% to 1.00% of the amount traded);
7. an "ordinary person" who checks at 22:00 Taiwan time, acts the next morning, pays 0.25% and misses 10% of entries.

Conventions are those of trend_ensemble: a signal decided at a close is the position held from that moment on (the
day after a daily close), the cost is charged on the amount traded, cash earns nothing, buy-and-hold pays no cost.

Data: Binance BTCUSDT 1-hour spot candles since 2017-08. The store that seasonality.py keeps
(cache/_store/seasonality_hourly.csv) is only read here, never written; if it is missing this module fetches the
candles itself and keeps its own store (cache/_store/trend_practical_hourly.csv). Only closed candles and finished
UTC days, only public key-less endpoints through common.get_json.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common

SHARED_STORE = common.STORE / "seasonality_hourly.csv"  # kept by seasonality.py - read only
OWN_STORE = common.STORE / "trend_practical_hourly.csv"  # only used when the shared store is not there
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
FIRST_MS = 1_502_928_000_000  # 2017-08-17 00:00 UTC, the day Binance listed BTCUSDT
MAX_PAGES = 130
MIN_HOURS = 50_000

COST = 0.001  # 0.10% of the amount traded
COSTS = (0.001, 0.0025, 0.005, 0.01)
YEAR = 365
MAIN_START = "2018-01-01"  # first full year of the hourly data
N_IN, N_OUT = 20, 10
SIMS = 1000
SEED = 20180101
TW = 8  # Taiwan is UTC+8
ENTRY_SKIPS = (0.10, 0.25, 0.50)
EXIT_SKIPS = (0.10, 0.25)
LATE_DAYS = (1, 2, 3, 5)
EVERY = (2, 3)
TABLE_HOURS_TW = (8, 12, 16, 18, 20, 22, 0, 4)  # Taiwan clock hours shown in the first table
PERSON_HOUR_UTC = 14  # 22:00 in Taiwan
PERSON_COST = 0.0025
PERSON_SKIP = 0.10
WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


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


def _pp(v) -> str:
    """A difference between two percentages, in percentage points."""
    if not _ok(v):
        return "—"
    v = float(v)
    return "0.0" if abs(v) < 0.05 else f"{v:+,.1f}"


def _num(v, digits: int = 2) -> str:
    if not _ok(v):
        return "—"
    text = f"{float(v):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _tw(hour_utc: int) -> str:
    """UTC hour as Taiwan clock time."""
    t = (hour_utc + TW) % 24
    part = "半夜" if t == 0 else "凌晨" if t < 6 else "早上" if t < 12 else "中午" if t == 12 else "下午" if t < 18 else "晚上"
    return f"{part} {12 if t % 12 == 0 else t % 12} 點"


def _span(values, fmt) -> str:
    lo, hi = min(values), max(values)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)} ～ {fmt(hi)}"


# ---------- data ----------

def _read(path) -> dict[int, tuple[float, float]]:
    """{candle open time in ms: (open, close)} from a store file; empty if it is missing or damaged."""
    rows: dict[int, tuple[float, float]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            o, c = float(parts[1]), float(parts[2])
            if o > 0 and c > 0 and math.isfinite(o) and math.isfinite(c):
                rows[int(parts[0])] = (o, c)
    except Exception:  # noqa: BLE001 - no file or a damaged one
        return {}
    return rows


def _write_own(rows: dict[int, tuple[float, float]]) -> None:
    try:
        common.STORE.mkdir(parents=True, exist_ok=True)
        text = "ms,open,close\n" + "\n".join(f"{ms},{o!r},{c!r}" for ms, (o, c) in sorted(rows.items())) + "\n"
        OWN_STORE.write_text(text, encoding="utf-8")
    except Exception:  # noqa: BLE001 - without the store the next run simply fetches again
        pass


def _fetch(rows: dict[int, tuple[float, float]], keep: bool) -> int:
    """Add closed 1-hour candles from the last stored hour on (in place). Returns how many were added."""
    from dashboard.collectors import price as px

    start = max(rows) if rows else FIRST_MS
    added = 0
    try:
        for page_no in range(MAX_PAGES):
            page = px._binance("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 1000,
                                                  "startTime": start})
            if not page:
                break
            now_ms = int(time.time() * 1000)
            for r in page:
                if int(r[6]) < now_ms and float(r[1]) > 0 and float(r[4]) > 0:  # closed candles only
                    rows[int(r[0])] = (float(r[1]), float(r[4]))
                    added += 1
            if len(page) < 1000:
                break
            start = int(page[-1][0]) + 1
            if keep and page_no % 20 == 19:
                _write_own(rows)  # keep what a long first run has fetched so far
    finally:
        if keep and added:
            _write_own(rows)
    return added


def _hourly_close() -> tuple[dict[int, float], str | None]:
    """{candle open time in ms: close} for closed BTCUSDT 1-hour candles, and a problem text if an update failed.

    The store of seasonality.py is read and never written: hours it does not have yet are fetched and used in
    memory only. Only when that store is not there does this module fetch everything and keep its own file."""
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    rows = _read(SHARED_STORE)
    shared = len(rows) >= MIN_HOURS
    if not shared:
        rows = _read(OWN_STORE)
    problem = None
    if not rows or max(rows) < today_ms - HOUR_MS:  # yesterday's last hour is not there yet
        try:
            _fetch(rows, keep=not shared)
        except Exception as exc:  # noqa: BLE001 - a stored history is still usable
            problem = f"幣安 1 小時線這次沒有更新成功（{type(exc).__name__}），用的是之前存下來的資料，沒有改用其他方法繞過。"
    if len(rows) < MIN_HOURS:
        raise RuntimeError(f"幣安 1 小時線只有 {len(rows)} 筆")
    return {ms: c for ms, (_, c) in rows.items()}, problem


def _grid(rows: dict[int, float]) -> tuple[np.ndarray, pd.Timestamp, int]:
    """Price at every full hour from the first UTC midnight to the last one: P[k] is the price at B0 + k hours
    (= the close of the candle that opened one hour earlier). A missing hour repeats the last known price.
    Returns (P, B0, number of missing hours); len(P) = 24 * days + 1."""
    first = min(rows) + HOUR_MS
    b0 = -(-first // DAY_MS) * DAY_MS
    today_ms = int(pd.Timestamp(common.today_utc()).value // 1_000_000)
    end = min(today_ms, ((max(rows) + HOUR_MS) // DAY_MS) * DAY_MS)  # finished UTC days only
    k = (end - b0) // HOUR_MS + 1
    if k < 24 * 200:
        raise RuntimeError("1 小時線太短")
    price = np.full(k, np.nan)
    for ms, c in rows.items():
        i = (ms + HOUR_MS - b0) // HOUR_MS
        if 0 <= i < k:
            price[i] = c
    missing = int(np.isnan(price).sum())
    filled = pd.Series(price).ffill().bfill().values
    return filled, pd.Timestamp(b0, unit="ms"), missing


# ---------- the rule ----------

def _rule(closes: np.ndarray, check: np.ndarray | None = None, n_in: int = N_IN, n_out: int = N_OUT) -> np.ndarray:
    """State after each close (1 = long, 0 = cash): long when the close is the highest of the last n_in closes,
    cash when it is the lowest of the last n_out; only acted on where check is True. Cash until a first high."""
    s = pd.Series(closes)
    high = s.rolling(n_in, min_periods=n_in).max()
    low = s.rolling(n_out, min_periods=n_out).min()
    ready = high.notna().values
    if check is not None:
        ready = ready & check
    raw = np.where(ready & (s >= high).values, 1.0, np.where(ready & (s <= low).values, 0.0, np.nan))
    return pd.Series(raw).ffill().fillna(0.0).values


def _stats(net: np.ndarray) -> dict:
    """Yearly compound return %, yearly volatility %, return / volatility and deepest loss % of daily net returns."""
    n = len(net)
    growth = float(np.prod(1 + net))
    ret = (growth ** (YEAR / n) - 1) * 100 if growth > 0 else -100.0
    vol = float(np.std(net, ddof=1) * math.sqrt(YEAR) * 100)
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    deepest = float((equity / np.maximum.accumulate(equity) - 1).min() * 100)
    return {"ret": float(ret), "vol": vol, "ratio": float(ret / vol) if vol > 1e-9 else 0.0, "deepest": deepest}


class _Market:
    """The hourly price grid and the two ways of turning positions into daily net returns."""

    def __init__(self, price: np.ndarray, b0: pd.Timestamp):
        self.price = price
        self.b0 = b0
        self.k = len(price)
        self.days = (self.k - 1) // 24
        self.dates = pd.date_range(b0, periods=self.days, freq="D")  # day j runs from boundary j to boundary j+1
        self.bound = price[::24]  # price at every UTC midnight, len days + 1
        self.change = self.bound[1:] / self.bound[:-1] - 1  # return of UTC day j
        self.hour_change = np.zeros(self.k)
        self.hour_change[1:] = price[1:] / price[:-1] - 1  # return of the hour ending at boundary k
        self.w0 = int((pd.Timestamp(MAIN_START) - b0).days)
        if self.w0 < N_IN + 5 or self.days - self.w0 < 3 * YEAR:
            raise RuntimeError("1 小時線涵蓋的期間不夠")

    # ----- daily positions: pos[j] is held during UTC day j (decided at midnight boundary j) -----
    def daily(self, pos: np.ndarray, cost: float = COST) -> dict:
        pos = np.asarray(pos, dtype=float)[:self.days]
        trade = np.abs(np.diff(pos, prepend=0.0))
        net = pos * self.change - cost * trade
        w = self.w0
        out = _stats(net[w:])
        entries = int(((pos[w:][1:] > 0) & (pos[w:][:-1] == 0)).sum()) + int(pos[w] > 0)
        out.update(trades=entries, share=float(pos[w:].mean() * 100), net=net[w:])
        return out

    # ----- hourly states: state[k] = what was decided at or before hour boundary k; acted on lag hours later -----
    def hourly(self, state: np.ndarray, lag: int = 0, cost: float = COST) -> dict:
        pos = np.zeros(self.k)  # pos[k] is held during the hour that ends at boundary k
        pos[lag + 1:] = state[:self.k - 1 - lag]
        trade = np.abs(np.diff(pos, prepend=0.0))
        gross = (1 + (pos * self.hour_change)[1:].reshape(self.days, 24)).prod(axis=1) - 1
        net = gross - cost * trade[1:].reshape(self.days, 24).sum(axis=1)
        w = self.w0
        out = _stats(net[w:])
        tail = pos[1 + 24 * w:]
        entries = int(((tail[1:] > 0) & (tail[:-1] == 0)).sum()) + int(tail[0] > 0)
        out.update(trades=entries, share=float(tail.mean() * 100), net=net[w:])
        return out

    def spread(self, values: np.ndarray, start: int, step: int = 24) -> np.ndarray:
        """Decisions made at boundaries start, start+step, ... carried forward to every hour boundary."""
        state = np.full(self.k, np.nan)
        idx = np.arange(start, self.k, step)
        state[idx] = values[:len(idx)]
        return pd.Series(state).ffill().fillna(0.0).values

    # ----- many simulated daily position paths at once: pos is (days, sims) -----
    def many(self, pos: np.ndarray, cost: float) -> dict:
        trade = np.abs(np.diff(pos, axis=0, prepend=0.0))
        net = (pos * self.change[:, None] - cost * trade)[self.w0:]
        n = len(net)
        equity = np.cumprod(1 + net, axis=0)
        ret = (equity[-1] ** (YEAR / n) - 1) * 100
        peak = np.maximum(np.maximum.accumulate(equity, axis=0), 1.0)
        deepest = (equity / peak - 1).min(axis=0) * 100
        return {"ret": ret, "deepest": deepest}


def _trades(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """For a 0/1 daily position: the trade number of every held day (-1 when in cash), the number of the trade
    that ended last before every cash day (-1 before the first trade), and the number of trades."""
    held = pos > 0
    starts = held & ~np.concatenate([[False], held[:-1]])
    number = np.cumsum(starts) - 1
    return np.where(held, number, -1), np.where(held, -1, number), int(starts.sum())


def _summary(sim: dict, hold_ret: float, base_ret: float) -> dict:
    ret, deep = sim["ret"], sim["deepest"]
    return {"ret": float(np.median(ret)), "ret_low": float(np.percentile(ret, 10)),
            "deepest": float(np.median(deep)), "deepest_low": float(np.percentile(deep, 10)),
            "below_hold": int((ret < hold_ret).sum()), "below_base": int((ret < base_ret).sum()), "sims": len(ret)}


def _compute(price: np.ndarray, b0: pd.Timestamp) -> dict:
    """Everything that depends on the data; study() only formats it."""
    m = _Market(price, b0)
    days, bound = m.days, m.bound
    closes = bound[:days]  # the close known at midnight boundary j (decides day j)
    base_state = _rule(closes)
    base = m.daily(base_state)
    hold = _stats(m.change[m.w0:])
    hold.update(trades=1, share=100.0)

    # ----- 1. which hour is the close -----
    hours = []
    for h in range(24):
        values = _rule(price[h::24])
        run = m.hourly(m.spread(values, h))
        run["hour"] = h
        run["state"] = float(values[-1])
        hours.append(run)
    order = sorted(range(24), key=lambda h: -hours[h]["ret"])
    for rank, h in enumerate(order, 1):
        hours[h]["rank"] = rank

    # ----- 2. checking less often -----
    index = np.arange(days)
    weekday = (m.b0.weekday() + index) % 7  # weekday of the midnight (= 08:00 Taiwan the same day) of the decision
    every = {k: [m.daily(_rule(closes, check=index % k == phase)) for phase in range(k)] for k in EVERY}
    weekly = [m.daily(_rule(closes, check=weekday == d)) for d in range(7)]
    mondays = np.where(weekday == 0)[0]  # weekly bars end on Sunday's close = Monday 08:00 Taiwan
    week_state = np.full(days, np.nan)
    week_state[mondays] = _rule(closes[mondays], n_in=4, n_out=2)
    week_bar = m.daily(pd.Series(week_state).ffill().fillna(0.0).values)

    # ----- 3. acting late -----
    base_hourly = m.spread(_rule(bound), 0)
    late = {"12h": m.hourly(base_hourly, lag=12)}
    for d in LATE_DAYS:
        late[d] = m.daily(np.concatenate([np.zeros(d), base_state[:-d]]))

    # ----- 4. missing signals -----
    trade_no, gap_no, n_trades = _trades(base_state)
    skip_entry, skip_exit = {}, {}
    for i, p in enumerate(ENTRY_SKIPS):
        rng = np.random.default_rng(SEED + i)
        keep = (rng.random((n_trades, SIMS)) >= p).astype(float)
        pos = np.where(trade_no[:, None] >= 0, keep[np.maximum(trade_no, 0)], 0.0)
        skip_entry[p] = _summary(m.many(pos, COST), hold["ret"], base["ret"])
    for i, p in enumerate(EXIT_SKIPS):
        rng = np.random.default_rng(SEED + 10 + i)
        stay = (rng.random((n_trades, SIMS)) < p).astype(float)
        pos = np.where(gap_no[:, None] >= 0, stay[np.maximum(gap_no, 0)], 0.0)
        pos = np.maximum(pos, base_state[:, None])
        skip_exit[p] = _summary(m.many(pos, COST), hold["ret"], base["ret"])

    # skip the entry right after two losing trades in a row (the person's own trades; a skip resets the count)
    starts = np.where((trade_no >= 0) & (np.concatenate([[-1], trade_no[:-1]]) != trade_no))[0]
    after_two = base_state.copy()
    losses, skipped_two, skipped_two_win = 0, 0, 0
    for t, s in enumerate(starts):
        inside = np.where(trade_no == t)[0]
        e = inside[-1] + 1  # the boundary where the trade is closed (or the last boundary if still open)
        result = bound[e] / bound[s] * (1 - COST) ** 2 - 1
        if losses >= 2:
            after_two[inside] = 0.0
            losses = 0
            if s >= m.w0:
                skipped_two += 1
                skipped_two_win += int(result > 0)
            continue
        losses = losses + 1 if result < 0 else 0
    two = m.daily(after_two)
    two.update(skipped=skipped_two, skipped_win=skipped_two_win)

    # ----- 5. hesitation: only buy back when the price is not above the last selling price -----
    high = pd.Series(closes).rolling(N_IN, min_periods=N_IN).max().values
    low = pd.Series(closes).rolling(N_OUT, min_periods=N_OUT).min().values
    hes = np.zeros(days)
    state, last_exit, refused = 0.0, None, 0
    refused_run = False
    for j in range(days):
        c = closes[j]
        if math.isfinite(high[j]):
            if state == 0 and c >= high[j]:
                if last_exit is None or c <= last_exit:
                    state = 1.0
                elif not refused_run:
                    refused_run = True
                    refused += int(j >= m.w0)
            elif c <= low[j]:
                if state == 1:
                    last_exit = c
                state = 0.0
                refused_run = False
        hes[j] = state
    hesitate = m.daily(hes)
    held_days = np.where(hes[m.w0:] > 0)[0]
    hesitate.update(refused=refused,
                    last_held=m.dates[m.w0 + held_days[-1]] if len(held_days) else None,
                    in_now=bool(hes[-1] > 0))

    # ----- 6. costs -----
    costs = {c: m.daily(base_state, cost=c) for c in COSTS}

    # ----- 7. the ordinary person -----
    person_values = _rule(price[PERSON_HOUR_UTC::24])
    person_hourly = m.spread(person_values, PERSON_HOUR_UTC)
    person_pos = person_hourly[::24][:days]  # what was decided at 22:00 Taiwan, acted on at the next 08:00
    step_now = hours[PERSON_HOUR_UTC]
    step_morning = m.daily(person_pos, cost=COST)
    step_cost = m.daily(person_pos, cost=PERSON_COST)
    p_trade, _, p_n = _trades(person_pos)
    rng = np.random.default_rng(SEED + 20)
    keep = (rng.random((p_n, SIMS)) >= PERSON_SKIP).astype(float)
    pos = np.where(p_trade[:, None] >= 0, keep[np.maximum(p_trade, 0)], 0.0)
    person_sim = m.many(pos, PERSON_COST)
    person = _summary(person_sim, hold["ret"], base["ret"])
    person["ratio"] = None

    since = int(np.argmax(base_state[::-1] != base_state[-1])) if (base_state != base_state[-1]).any() else days
    return {"m": m, "last": m.dates[-1], "last_price": float(bound[-1]), "base": base, "hold": hold, "hours": hours,
            "every": every, "weekly": weekly, "week_bar": week_bar, "late": late, "skip_entry": skip_entry,
            "skip_exit": skip_exit, "two": two, "hesitate": hesitate, "costs": costs, "n_trades": n_trades,
            "person": person, "step_now": step_now, "step_morning": step_morning, "step_cost": step_cost,
            "base_state": float(base_state[-1]), "base_since": m.dates[-1] - pd.Timedelta(days=since - 1),
            "person_state": float(person_values[-1]), "hours_long": int(sum(h["state"] > 0 for h in hours))}


def _daily_check(r: dict) -> str | None:
    """Compare the 08:00 version built from hourly candles with the dashboard's long daily series."""
    try:
        from dashboard.studies2 import trend_ensemble as te

        close = te._daily_close()
        state = _rule(close.values)
        pos = pd.Series(state, index=close.index).shift(1)
        net, _ = te._run(pos.values, close.pct_change().values, COST)
        net = pd.Series(net, index=close.index)
        net = net[(net.index >= pd.Timestamp(MAIN_START)) & (net.index <= r["last"])]
        ref = _stats(net.values)
        b = r["base"]
        return (f"核對：用儀表板原本的日線（幣安日線）跑同一條規則，{MAIN_START[:4]} 年到 {_d(net.index[-1])} 是每年 {_pct(ref['ret'])}、"
                f"最深虧 {_pct(ref['deepest'])}；這裡用 1 小時線自己拼出來的「早上 8 點收盤」版本是每年 {_pct(b['ret'])}、"
                f"最深虧 {_pct(b['deepest'])}，相差 {abs(ref['ret'] - b['ret']):.2f} 和 {abs(ref['deepest'] - b['deepest']):.2f} 個百分點。")
    except Exception:  # noqa: BLE001 - the cross-check is optional
        return None


# ---------- the study ----------

def study() -> dict:
    rows, problem = _hourly_close()
    price, b0, missing = _grid(rows)
    r = _compute(price, b0)
    m, last, base, hold, hours = r["m"], r["last"], r["base"], r["hold"], r["hours"]
    first_year = int(MAIN_START[:4])
    span = f"{MAIN_START} 到 {_d(last)}"
    sims = f"{SIMS:,} 次模擬"

    def gap(x: dict) -> str:
        return _pp(x["ret"] - base["ret"])

    # ----- table 1: which hour is the close -----
    rets = np.array([h["ret"] for h in hours])
    deeps = np.array([h["deepest"] for h in hours])
    ratios = np.array([h["ratio"] for h in hours])
    best, worst = int(np.argmax(rets)), int(np.argmin(rets))
    beat_hold = int((rets > hold["ret"]).sum())
    shallower = int((deeps > hold["deepest"]).sum())

    def hour_row(name: str, h: dict) -> list:
        return [name, _pct(h["ret"]), _pct(h["deepest"]), _num(h["ratio"]), f"{h['trades']} 次",
                f"24 個裡第 {h['rank']}"]

    hour_rows = []
    for t in TABLE_HOURS_TW:
        h = (t - TW) % 24
        name = f"台灣{_tw(h)}" + ("（教科書版：UTC 0 點）" if h == 0 else "")
        hour_rows.append(hour_row(name, hours[h]))
    hour_rows.append(hour_row(f"24 個裡每年報酬最高的：台灣{_tw(best)}", hours[best]))
    hour_rows.append(hour_row(f"24 個裡每年報酬最低的：台灣{_tw(worst)}", hours[worst]))
    hour_rows.append(["24 個的中位數", _pct(float(np.median(rets))), _pct(float(np.median(deeps))),
                      _num(float(np.median(ratios))), f"{int(np.median([h['trades'] for h in hours]))} 次", "—"])
    hour_rows.append(["對照：一直抱著不動", _pct(hold["ret"]), _pct(hold["deepest"]), _num(hold["ratio"]), "—", "—"])

    # ----- table 2: checking less often, acting late -----
    def row2(name: str, x: dict) -> list:
        return [name, _pct(x["ret"]), gap(x), _pct(x["deepest"]), _num(x["ratio"]), f"{x['trades']} 次",
                f"{x['share']:.0f}%"]

    def range_row(name: str, runs: list) -> list:
        return [name, _span([x["ret"] for x in runs], _pct), _span([x["ret"] - base["ret"] for x in runs], _pp),
                _span([x["deepest"] for x in runs], _pct), _span([x["ratio"] for x in runs], _num),
                _span([x["trades"] for x in runs], lambda v: f"{int(v)}") + " 次",
                _span([x["share"] for x in runs], lambda v: f"{v:.0f}%")]

    rows2 = [row2("教科書版：每天早上 8 點看、當下就做", base)]
    for k in EVERY:
        rows2.append(range_row(f"每 {k} 天看一次（{k} 種起算日，最差～最好）", r["every"][k]))
    for d in range(7):
        rows2.append(row2(f"一週只看一次：{WEEKDAYS[d]}早上 8 點", r["weekly"][d]))
    rows2.append(row2("週線版：創 4 週新高進場、2 週新低出場（每週一早上 8 點看）", r["week_bar"]))
    rows2.append(row2("每天看，但晚 12 小時才做（當天晚上 8 點的價格）", r["late"]["12h"]))
    for d in LATE_DAYS:
        rows2.append(row2(f"每天看，但晚 {d} 天才做", r["late"][d]))

    # ----- table 3: missed signals, hesitation -----
    def sim_row(name: str, x: dict) -> list:
        return [name, _pct(x["ret"]), _pct(x["ret_low"]), _pct(x["deepest"]), _pct(x["deepest_low"]),
                f"{sims}裡 {x['below_hold']:,} 次", f"{sims}裡 {x['below_base']:,} 次"]

    def one_row(name: str, x: dict) -> list:
        return [name, _pct(x["ret"]), "—", _pct(x["deepest"]), "—",
                "輸" if x["ret"] < hold["ret"] else "沒有輸", "輸" if x["ret"] < base["ret"] - 1e-9 else "沒有輸"]

    two, hes = r["two"], r["hesitate"]
    rows3 = [[f"教科書版：每個訊號都照做（{base['trades']} 次進場）", _pct(base["ret"]), "—", _pct(base["deepest"]), "—",
              "沒有輸" if base["ret"] >= hold["ret"] else "輸", "—"]]
    rows3 += [sim_row(f"每次進場訊號有 {p * 100:.0f}% 的機會漏掉", r["skip_entry"][p]) for p in ENTRY_SKIPS]
    rows3 += [sim_row(f"每次出場訊號有 {p * 100:.0f}% 的機會漏掉", r["skip_exit"][p]) for p in EXIT_SKIPS]
    rows3.append(one_row(f"連續虧兩次之後，下一次進場不敢跟（共跳過 {two['skipped']} 次）", two))
    rows3.append(one_row(f"只有不比上次賣出價高才肯買回（共 {hes['refused']} 波沒有跟）", hes))
    rows3.append(["對照：一直抱著不動", _pct(hold["ret"]), "—", _pct(hold["deepest"]), "—", "—", "—"])

    # ----- table 4: costs and the ordinary person -----
    def row4(name: str, x: dict, extra: str = "—") -> list:
        return [name, _pct(x["ret"]), _pct(x["deepest"]), _num(x.get("ratio")), _pp(x["ret"] - hold["ret"]),
                gap(x), extra]

    person = r["person"]
    rows4 = [row4(f"教科書版，每次買賣成本 {c * 100:.2f}%", r["costs"][c]) for c in COSTS]
    rows4 += [
        row4("一般人①：每天晚上 10 點看、當下就做，成本 0.10%", r["step_now"]),
        row4("一般人②：晚上 10 點看、隔天早上 8 點才做，成本 0.10%", r["step_morning"]),
        row4(f"一般人③：同②，成本改成 {PERSON_COST * 100:.2f}%", r["step_cost"]),
        row4(f"一般人④：同③，再隨機漏掉 {PERSON_SKIP * 100:.0f}% 的進場（{sims}的中間值）", person,
             f"{sims}裡 {person['below_hold']:,} 次輸給一直抱著"),
        [f"一般人④裡最差的一成（{sims}裡排第 {SIMS // 10} 差的那條線）", _pct(person["ret_low"]),
         _pct(person["deepest_low"]), "—", _pp(person["ret_low"] - hold["ret"]),
         _pp(person["ret_low"] - base["ret"]), "—"],
        ["對照：一直抱著不動（沒有扣成本）", _pct(hold["ret"]), _pct(hold["deepest"]), _num(hold["ratio"]), "0.0",
         _pp(hold["ret"] - base["ret"]), "—"],
    ]

    swing = float(rets.max() - rets.min())
    edge = base["ret"] - hold["ret"]
    tables = [
        {"caption": (f"把哪個時刻當成「一天的收盤」（{span}，成本 {COST * 100:.2f}%）。用 1 小時線重新拼出 24 種日線——一天分別結束在"
                     "每一個整點——每一種都跑同一條「創 20 日新高進場、10 日新低出場」，在那個時刻看、當下成交。表裡列出台灣時間幾個常見的時刻，"
                     f"加上 24 種裡最高、最低和中位數。24 種的每年報酬從 {_pct(rets.min())} 到 {_pct(rets.max())}，相差 {swing:.1f} 個百分點；"
                     f"24 種裡 {beat_hold} 種的每年報酬比一直抱著高、{shallower} 種的最深虧損比一直抱著淺。"),
         "columns": ["把哪個時刻當成一天的收盤", "每年報酬", "最深曾經虧多少", "報酬÷波動", "進場次數", "每年報酬排名"],
         "rows": hour_rows},
        {"caption": (f"少看幾次、晚一點才做（{span}，成本 {COST * 100:.2f}%，都用台灣早上 8 點的日收盤）。「少看」是只有在看的那一天才判斷"
                     "「是不是最近 20 個日收盤的最高／最近 10 個日收盤的最低」，其他日子出現的訊號都當作沒看到；"
                     "「晚做」是訊號照樣每天看，但過了那麼久才用當時的價格成交。「和教科書版差」是每年報酬差幾個百分點。"),
         "columns": ["做法", "每年報酬", "和教科書版差（百分點）", "最深曾經虧多少", "報酬÷波動", "進場次數", "持有比特幣的時間"],
         "rows": rows2},
        {"caption": (f"漏掉訊號、不敢買回（{span}，成本 {COST * 100:.2f}%）。隨機漏掉的幾列各做 {SIMS:,} 次模擬：漏掉一次進場，就整波不參加、"
                     "抱現金等到下一次新的進場訊號；漏掉一次出場，就繼續抱著、等到下一次出場訊號才賣。「中間值」是 "
                     f"{SIMS:,} 次裡排在正中間的結果，「最差的一成」是 {SIMS:,} 次裡第 {SIMS // 10} 差的那條線。"),
         "columns": ["情況", "每年報酬（中間值）", "每年報酬（最差的一成）", "最深虧損（中間值）", "最深虧損（最差的一成）",
                     "每年報酬輸給一直抱著", "每年報酬輸給教科書版"],
         "rows": rows3},
        {"caption": (f"成本，以及把幾種不完美疊在一起的「一般人」情境（{span}）。成本是每次買或賣各扣交易金額的那個比例（手續費加買賣價差加滑價）。"
                     f"一般人情境一步一步疊上去：先把看盤時間改成台灣晚上 10 點，再改成隔天早上 8 點才成交，再把成本調到 {PERSON_COST * 100:.2f}%，"
                     f"最後每次進場有 {PERSON_SKIP * 100:.0f}% 的機會漏掉。"),
         "columns": ["情境", "每年報酬", "最深曾經虧多少", "報酬÷波動", "比一直抱著每年多幾個百分點", "和教科書版差（百分點）",
                     "模擬結果"],
         "rows": rows4},
    ]

    # ----- today -----
    def stake(v: float) -> str:
        return "持有比特幣" if v > 0 else "全部現金"

    now = [
        {"label": f"最後一個收完的日子（{_d(last)}，台灣時間隔天早上 8 點收）", "value": f"收在 {r['last_price']:,.0f} 美元"},
        {"label": "教科書版規則目前的狀態", "value": f"{stake(r['base_state'])}（從 {_d(r['base_since'])} 起）"},
        {"label": "24 種「收盤時刻」的版本裡，目前是持有的", "value": f"24 種裡 {r['hours_long']} 種"},
        {"label": "「晚上 10 點看」的版本目前的狀態", "value": stake(r["person_state"])},
        {"label": "這幾格的意思", "value": "只是描述規則在各種看法下目前停在哪裡，不是買賣建議"},
    ]

    # ----- verdict -----
    def mean_ret(runs: list) -> float:
        return float(np.mean([x["ret"] for x in runs]))

    week_rets = [x["ret"] for x in r["weekly"]]
    items = [
        ("漏掉一成出場（中間值）", r["skip_exit"][0.10]["ret"]),
        ("漏掉一成進場（中間值）", r["skip_entry"][0.10]["ret"]),
        ("連虧兩次後跳過下一次進場", two["ret"]),
        ("成本 0.25%", r["costs"][0.0025]["ret"]),
        ("每 3 天看一次（3 種起算日平均）", mean_ret(r["every"][3])),
        ("晚 1 天才做", r["late"][1]["ret"]),
        ("成本 0.50%", r["costs"][0.005]["ret"]),
        ("晚 12 小時才做", r["late"]["12h"]["ret"]),
        ("一週只看一次（7 種星期幾的中位數）", float(np.median(week_rets))),
        ("改用週線", r["week_bar"]["ret"]),
        ("晚 3 天才做", r["late"][3]["ret"]),
        ("漏掉一半進場（中間值）", r["skip_entry"][0.50]["ret"]),
        ("成本 1.00%", r["costs"][0.01]["ret"]),
        ("比上次賣出價高就不買回", hes["ret"]),
    ]
    items.sort(key=lambda pair: -pair[1])
    small = [(n, v) for n, v in items if v >= base["ret"] - 2.0]
    eaten = [(n, v) for n, v in items if v <= hold["ret"]]
    middle = [(n, v) for n, v in items if hold["ret"] < v < base["ret"] - 2.0]

    def listing(pairs: list) -> str:
        return "、".join(f"{n} {_pct(v)}" for n, v in pairs)

    rank0 = hours[0]["rank"]
    only = "只有" if beat_hold <= 12 else ""
    if swing >= max(edge, 8.0):
        clock = (f"相差 {swing:.1f} 個百分點，比這條規則贏過一直抱著的幅度（{edge:.1f} 個百分點）還大，24 種裡{only} {beat_hold} 種贏過一直抱著——"
                 "回測的每年報酬有一大部分是剛好選對時鐘的運氣")
    elif swing >= 4.0:
        clock = (f"相差 {swing:.1f} 個百分點，24 種裡 {beat_hold} 種贏過一直抱著——選哪個時鐘有明顯影響，"
                 "回測的每年報酬有一部分靠的是時鐘選得好不好")
    else:
        clock = f"只相差 {swing:.1f} 個百分點，24 種裡 {beat_hold} 種贏過一直抱著，選哪個時鐘影響不大"
    part1 = (f"{first_year} 年到 {_d(last)}，「創 20 日新高進場、10 日新低出場」照教科書做（每天台灣早上 8 點看、當下成交、成本 "
             f"{COST * 100:.2f}%）是每年 {_pct(base['ret'])}、最深虧 {_pct(base['deepest'])}，一直抱著是每年 {_pct(hold['ret'])}、"
             f"最深虧 {_pct(hold['deepest'])}；但把「一天的收盤」換成 24 個整點各跑一次，每年報酬從 {_pct(rets.min())} 到 {_pct(rets.max())}"
             f"（中位數 {_pct(float(np.median(rets)))}，早上 8 點排第 {rank0}），{clock}；比較站得住的是虧損比較淺："
             f"24 種的最深虧損在 {_pct(deeps.min())} 到 {_pct(deeps.max())} 之間，"
             + ("全部比一直抱著淺。" if shallower == 24 else f"其中 {shallower} 種比一直抱著淺。"))
    parts = []
    if small:
        parts.append(f"幾乎沒影響（每年少不到 2 個百分點）的只有：{listing(small)}")
    else:
        parts.append("沒有哪一項不完美是幾乎沒影響的（每一項都讓每年報酬少 2 個百分點以上）")
    if middle:
        parts.append(f"少掉一截但每年報酬還比一直抱著高的：{listing(middle)}")
    if eaten:
        parts.append(f"把報酬上的優勢整個吃掉（不比一直抱著高）的：{listing(eaten)}")
    else:
        parts.append("單獨一項不完美都還沒有把每年報酬壓到一直抱著以下")
    part2 = "各種不完美單獨發生時的每年報酬——" + "；".join(parts) + "。"
    part3 = (f"把幾項疊起來的一般人版本（晚上 10 點看、隔天早上 8 點才做、成本 {PERSON_COST * 100:.2f}%、隨機漏掉一成進場），"
             f"{sims}的中間值是每年 {_pct(person['ret'])}、最深虧 {_pct(person['deepest'])}，"
             f"{sims}裡 {person['below_hold']:,} 次的每年報酬輸給一直抱著（最差的一成是每年 {_pct(person['ret_low'])} 以下）"
             f"——也就是教科書版多出來的那 {edge:.1f} 個百分點，"
             + ("照一般人的做法大致上沒有留下來。" if person["ret"] <= hold["ret"] + 1.0
                else f"照一般人的做法還剩 {person['ret'] - hold['ret']:.1f} 個百分點左右。"))
    verdict = part1 + part2 + part3

    # ----- notes -----
    e10, x25 = r["skip_entry"][0.10], r["skip_exit"][0.25]
    notes = [
        "規則本身：收盤是最近 20 個日收盤裡最高的就進場（全部換成比特幣），是最近 10 個日收盤裡最低的就出場（全部換回現金），"
        "其他時候維持原狀。教科書版假設每天在台灣早上 8 點（UTC 0 點）看收盤價，而且就用那個價格成交。"
        "每次買或賣扣交易金額的 0.10%，現金沒有利息，沒有算稅；一直抱著沒有扣成本。",
        "「每年報酬」是把整段的總成果換算成每年的複利；「最深曾經虧多少」是資金從最高點往下掉最多的幅度；"
        "「報酬÷波動」是每年報酬除以每天漲跌換算成一年的起伏，越高代表每承受一分起伏換到的報酬越多。"
        "所有版本的每日損益都用同一種切法（台灣早上 8 點到隔天早上 8 點）來算，才能互相比較。",
        f"隨機模擬用固定的亂數種子（{SEED} 起），所以每次重跑、只要資料沒變，結果都一樣；每種情況做 {SIMS:,} 次，"
        f"「{SIMS:,} 次裡幾次」本身也有十幾二十次的誤差。",
        f"漏掉訊號的定義：進場訊號是指教科書版「從現金變成持有」的那一天（{first_year} 年以來共 {base['trades']} 次）；漏掉就是整波不參加，"
        "等下一次從頭開始的進場訊號。出場訊號是指「從持有變成現金」的那一天；漏掉就一路抱到下一次出場訊號。"
        "實際上隔天多半還會再看到同方向的訊號，那種情況比較像第二張表的「晚幾天才做」，所以這裡的漏掉是比較嚴重的那一種。"
        f"這條規則的獲利集中在少數幾波大行情，所以漏掉哪一波差別很大：漏掉一成進場時，中間值是每年 {_pct(e10['ret'])}，"
        f"最差的一成只剩每年 {_pct(e10['ret_low'])} 以下。漏掉 25% 的出場時，最深虧損的中間值是 {_pct(x25['deepest'])}"
        f"（教科書版 {_pct(base['deepest'])}），最差的一成到 {_pct(x25['deepest_low'])}。",
        f"「連續虧兩次之後不敢跟」：自己實際做的交易連續兩次虧錢（扣成本後），下一次進場訊號就跳過，跳過之後重新計算。"
        f"{first_year} 年以來共跳過 {two['skipped']} 次，其中 {two['skipped_win']} 次是照做會賺錢的。"
        f"「不比上次賣出價高才肯買回」：出現進場訊號時，收盤價高於自己上一次賣出的價格就不買，那一波整個不參加"
        f"（{first_year} 年以來共 {hes['refused']} 波）"
        + ("" if hes["in_now"] or hes["last_held"] is None
           else f"；照這樣做，最後一次持有比特幣是到 {_d(hes['last_held'])}，之後價格一直高於上次賣出價，就再也沒有買回") + "。",
        "「少看幾次」的幾列：每 2 天、每 3 天看一次各有 2 種、3 種起算日，表裡列的是其中最差到最好的範圍；一週只看一次是 7 種星期幾各跑一次。"
        "星期幾指的是台灣時間那天早上 8 點看到的收盤價。差別有一部分只是剛好哪一天看的運氣，不代表某個星期幾比較好。"
        "週線版用每週一早上 8 點的收盤當週線，「2 週新低」等於這一週收得比上一週低就出場。",
        "只測了比特幣一種資產、不到九個年度，大致就是兩三輪大漲大跌；各列之間一兩個百分點的差距不能當成確定的結論。"
        "24 種收盤時刻用的是同一段歷史，它們之間的差距主要來自「少數幾筆關鍵交易的進出價格剛好落在哪裡」；"
        "也因為這樣，「晚上 10 點看、隔天早上 8 點才做」的每年報酬反而比「晚上 10 點看、當下就做」高，"
        "這比較像是早上 8 點這個成交時刻在這段歷史裡剛好有利，不代表拖到隔天比較好。",
        "「創 20 日新高進場、10 日新低出場」這條規則是事先選定的，但這一頁的各種壓力測試都是事後、已經知道這段歷史之後才做的；"
        "要測哪些不完美、機率設多少、一般人情境怎麼組合，都是這次才決定的，不是事先登記好的。",
        f"資料是幣安現貨 BTCUSDT 的 1 小時線，從 2017-08 開始，所以主要期間從第一個完整年度 {first_year} 年算起；更早的年代沒有辦法做這些測試。"
        f"整段共缺 {missing:,} 個小時（幣安早年停機維護），缺的小時沿用前一個小時的價格。只用已經收完的小時和已經過完的日子。"
        "成交價一律用整點的收盤價，沒有另外算下單當下的滑價——想看滑價的影響請看第四張表的成本列。",
        "過去的結果不保證未來會一樣。這裡只是整理這條規則在各種不完美的做法下過去做了什麼，不是買賣建議。",
    ]
    check = _daily_check(r)
    if check:
        notes.insert(2, check)
    if problem:
        notes.append(problem)

    return {
        "id": "trend_practical",
        "title": "照一般人的方式做，順勢規則還剩多少？",
        "question": ("回測假設每天台灣早上 8 點準時看盤、當下用收盤價成交、一次訊號都不漏。真的人做不到——"
                     "換個時間看、少看幾天、晚一點才做、漏掉幾次、成本高一點，這條規則的成績還剩多少？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "幣安現貨 BTCUSDT 1 小時線（2017-08 起）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "幣安公開行情 API（K 線）",
             "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
