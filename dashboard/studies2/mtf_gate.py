"""Historical study "mtf_gate": a daily trend rule that only PERMITS trades, with faster bars timing the entries and
exits inside it ("multi-timeframe gate") - does it improve the dashboard's main signal ma50 (hold while the daily
close is above its 50-day average)?

Claim (Quantpedia, "How to Design a Simple Multi-Timeframe Trend Strategy on Bitcoin", 13 Nov 2025): on Gemini
BTC/USD hourly bars, Dec 2018 - Nov 2025, costs not stated, a daily MACD(12,26,9) filter (read only after the daily
close) that only permits long entries, with hourly MACD crosses timing them, raised the Sharpe ratio from 0.33 (pure
hourly MACD, 2,262 trades, -23.9% deepest) to 0.80 (enter on the hourly cross-up, hold ONE hourly bar, ~1,000 trades,
-12.4%) and 1.07 ("+STOP": hold while hourly bars stay positive, leave at the close of the first negative bar).
intraday_trend.py tested faster bars ALONE; the gate structure had not been tested.

Pre-registered BEFORE any result was seen (nothing below was changed after running):

Q1 - replication, Binance BTCUSDT 1-hour bars, measured 2018-12-01 .. 2025-11-30 (and to the last finished day):
  V1 pure hourly MACD: long from the close of an hourly MACD cross-up to the close of the next cross-down.
  V2 D1H1: daily MACD line above its signal line at the last finished UTC day's close = uptrend; while it is up, an
     hourly cross-up opens a long at that bar's close, held for exactly one hourly bar.
  V3 D1H1+STOP: same entry; held while hourly bars close above their open; leaves at the close of the first bar that
     closes below its open (a bar closing exactly at its open does not stop it); the daily filter turning down
     does not close an open position (the article does not say; only the stop exits).
  Sharpe = mean daily return / its standard deviation x sqrt(365), no risk-free rate. Costs 0 and 0.10% per side
  (and 0.25%). "Replicated" = V2 and V3 Sharpe within about +-0.25 of 0.80 / 1.07 at zero cost AND the order
  V1 < V2 < V3 holds.
Q2 - daily ma50 gate + 4-hour MACD(12,26,9): long from the close of a FRESH 4-hour cross-up that happens while the
  gate is on, until the close of a 4-hour cross-down or the gate turning off (whichever first); flat whenever the
  gate is off; after a gate-off exit a new fresh cross-up is needed.
Q3 - daily ma50 gate + 4-hour close above its 50-bar average: long whenever both are true.
Baseline: ma50 itself (= the gate alone), built on the same hourly grid.

No look-ahead: the daily gate value is the 50-day average of closes up to the close at the day boundary; it is used
from that boundary on. A 4-hour (or 1-hour) signal is computed at the bar's close and held during the next bar.
Every position is held hour by hour on one grid, costs are charged on the amount traded, daily results are read at
every UTC midnight (so all rules are measured on the same UTC days).

Fair exam (main_signal_check's grid, adapted as intraday_trend did):
 A. Textbook, bitcoin: 72 monthly start dates 2018-01 .. 2023-12 x 24 clocks, 0.10% per side. A "clock" shifts the
    WHOLE system by h hours: the daily gate closes at h:00 UTC and the 4-hour bars close at h, h+4, ... (so their
    phase is h mod 4). For sub-daily rules this is the only clock that makes sense - shifting only the daily close
    but not the 4-hour bars would misalign the gate with the bars, and the 4-hour bars have only 4 distinct phases,
    all of which appear among the 24 shifts.
 B. Ordinary person, bitcoin, the dashboard's clock (00:00 UTC daily close, 4-hour bars at 00/04/08... UTC): the
    person checks only at fixed Taiwan times and trades only then - main schedule 3 checks (08:00, 12:00, 22:00
    Taiwan = 00, 04, 14 UTC), second schedule 2 checks (08:00, 22:00) - at each check the person sets the position to
    what the rule wants at that moment (latest finished day and latest finished 4-hour bar); 0.25% per side; 10% of
    entries are missed at random and a missed entry sits out that whole trade (as robust_trend._missed); 200
    simulations; 72 monthly start dates. ma50 is run the same way.
 C. Other coins ETH, SOL, BNB, XRP (Binance 1-hour), clock 0, textbook, monthly start dates from the first date every
    rule has a value to three years before today.
 Ranking score = the worst of three shares "yearly return beats holding from the same start": A, B (3 checks),
 C pooled. Deepest loss = median over the A cells.
 Placebo (bitcoin, clock 0, textbook 0.10%, measured from 2018-01-01): the same number of trades with the same
 lengths (so the same time in market) placed at random inside the hours the SAME gate is on, 200 draws; p = share of
 draws whose yearly return is at least the rule's.
 PASS (each candidate Q2, Q3) = ranking score >= ma50's AND median deepest loss (A) no worse than ma50's AND
 placebo p <= 10% AND the ordinary person (B, 3 checks) still beats ordinary-person ma50 (median yearly-return edge
 over ma50, same simulations / start dates, > 0).

Data: Binance spot 1-hour candles from the stores kept by seasonality.py (bitcoin, which also keeps the hourly open)
and clock_robustness.py (other coins), read only; missing recent hours are fetched into memory, never written.
Bitcoin's daily gate before 2017-08 is warmed with the dashboard's long daily closes (cr._prehistory, same series as
te._daily_close). Only public key-less endpoints through the dashboard's own helpers.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import intraday_trend as it
from dashboard.studies2 import main_signal_check as msc

COST, P_COST, P_MISS = 0.001, 0.0025, 0.10
COSTS = (0.0, 0.001, 0.0025)
SIMS, DRAWS = 200, 200
SEED = 20251113
YEAR = 365
STARTS = ("2018-01-01", "2023-12-01")
MEASURE = "2018-01-01"
REP = ("2018-12-01", "2025-12-01")  # Quantpedia's window: Dec 2018 .. Nov 2025
CHECKS3, CHECKS2 = (0, 4, 14), (0, 14)  # UTC hours = 08:00 / 12:00 / 22:00 Taiwan
COINS = ("ETH", "SOL", "BNB", "XRP")
P_BAR = 0.10
RULES = ("ma50", "q2", "q3")
NAMES = {"ma50": "日線 50 日均線（主訊號，閘門本身）",
         "q2": "Q2：50 日均線閘門 ＋ 4 小時 MACD 交叉進出",
         "q3": "Q3：50 日均線閘門 ＋ 4 小時收盤高於 50 根均線"}
SHORT = {"ma50": "50 日均線", "q2": "Q2 閘門＋4 小時 MACD", "q3": "Q3 閘門＋4 小時均線"}
Q1N = {"v1": "V1 純 1 小時 MACD 交叉", "v2": "V2 日線 MACD 閘門＋1 小時交叉、只抱 1 根",
       "v3": "V3 同 V2，抱到第一根收黑（＋STOP）"}
Q1PUB = {"v1": "0.33（2,262 筆，最深 -23.9%）", "v2": "0.80（約 1,000 筆，最深 -12.4%）", "v3": "1.07"}

_pct, _pp, _num, _d = cr._pct, cr._pp, cr._num, cr._d


# ---------- data ----------

def _opens(b0: pd.Timestamp, k: int) -> np.ndarray:
    """Open of the 1-hour candle that closes at grid hour i (NaN where the shared store has no open)."""
    out = np.full(k, np.nan)
    b = int(b0.value // 1_000_000)
    try:
        for line in cr.SHARED_STORE.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            i = (int(parts[0]) + cr.HOUR_MS - b) // cr.HOUR_MS
            o = float(parts[1])
            if 0 <= i < k and o > 0 and math.isfinite(o):
                out[i] = o
    except Exception:  # noqa: BLE001 - without opens the previous close stands in
        pass
    return out


# ---------- building blocks ----------

def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def _macd_up(x: np.ndarray) -> np.ndarray:
    """MACD(12,26,9) line above its signal line (False during the first 35 values)."""
    line = _ema(x, 12) - _ema(x, 26)
    up = line > _ema(line, 9)
    up[:35] = False
    return up


def _above(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x)
    mean = s.rolling(n, min_periods=n).mean()
    return ((s > mean) & mean.notna()).values


def _spread(vals: np.ndarray, first: int, step: int, k: int) -> np.ndarray:
    """Value decided at the close at hour first + step*i, held for the hours first + step*i ... + step - 1."""
    out = np.zeros(k)
    idx = np.arange(k)
    i = (idx - first) // step
    ok = (idx >= first) & (i < len(vals))
    out[ok] = np.asarray(vals, dtype=float)[i[ok]]
    return out


def _cross_in_gate(up: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Q2's state machine on 4-hour bars: in on a fresh cross-up while the gate is on; out on a cross-down or gate off."""
    pos = np.zeros(len(up))
    cur = False
    for i in range(1, len(up)):
        cur = (up[i] and gate[i]) if cur else (up[i] and not up[i - 1] and gate[i])
        pos[i] = cur
    return pos


def _targets(price: np.ndarray, h: int, pre: np.ndarray) -> dict[str, np.ndarray]:
    """Hourly targets of ma50, Q2, Q3 for clock h (index k = position held from hour k to k+1)."""
    k = len(price)
    daily = np.concatenate([pre, price[h::24]])
    gate = _above(daily, 50)[len(pre):]
    gate_h = _spread(gate, h, 24, k)
    p = h % 4
    c4 = price[p::4]
    gate4 = gate_h[p::4] > 0.5
    q2 = _cross_in_gate(_macd_up(c4), gate4)
    q3 = (gate4 & _above(c4, 50)).astype(float)
    return {"ma50": gate_h, "q2": _spread(q2, p, 4, k), "q3": _spread(q3, p, 4, k)}


def _days_net(price: np.ndarray, w: np.ndarray, cost: float) -> np.ndarray:
    """Net return of every UTC day of an hourly position series."""
    k = len(price)
    r = price[1:] / price[:-1] - 1
    w = np.asarray(w, dtype=float)[:k - 1]
    trade = np.abs(np.diff(w, prepend=0.0))
    net = 1 + w * r - cost * trade
    days = (k - 1) // 24
    return np.prod(net[:24 * days].reshape(days, 24), axis=1) - 1


def _sharpe(net: np.ndarray) -> float:
    sd = float(np.std(net, ddof=1))
    return float(np.mean(net) / sd * math.sqrt(YEAR)) if sd > 1e-12 else 0.0


def _entries(w: np.ndarray, lo: int, hi: int) -> int:
    seg = np.asarray(w[lo:hi]) > 0.5
    return int(np.sum(seg[1:] & ~seg[:-1]) + int(seg[0]))


def _measure(price: np.ndarray, w: np.ndarray, lo_day: int, hi_day: int, cost: float) -> dict:
    net = _days_net(price, w, cost)[lo_day:hi_day]
    out = cr._stats(net)
    out["sharpe"] = _sharpe(net)
    out["trades"] = _entries(w, 24 * lo_day, 24 * hi_day)
    out["years"] = len(net) / YEAR
    out["share"] = float(np.mean(np.asarray(w[24 * lo_day:24 * hi_day]) > 0.5) * 100)
    return out


# ---------- Q1: the published rule ----------

def _q1(price: np.ndarray, opens: np.ndarray, pre: np.ndarray) -> dict[str, np.ndarray]:
    k = len(price)
    daily = np.concatenate([pre, price[0::24]])
    gate_h = _spread(_macd_up(daily)[len(pre):], 0, 24, k) > 0.5
    up = _macd_up(price)
    cross = np.concatenate([[False], up[1:] & ~up[:-1]])
    v1 = up.astype(float)
    v2 = (cross & gate_h).astype(float)
    o = np.where(np.isfinite(opens), opens, np.concatenate([[price[0]], price[:-1]]))
    neg = price < o  # the bar that closes at hour i closed below its open
    v3 = np.zeros(k)
    cur = False
    for i in range(k):
        if cur and neg[i]:
            cur = False
        if not cur and cross[i] and gate_h[i]:
            cur = True
        v3[i] = cur
    return {"v1": v1, "v2": v2, "v3": v3, "gate": gate_h.astype(float)}


# ---------- the ordinary person ----------

def _person(price: np.ndarray, w: np.ndarray, checks: tuple, days: int, rng: np.random.Generator) -> np.ndarray:
    """(days, SIMS) net daily returns of people who look only at the check hours (UTC) of every day."""
    hours = (np.arange(days)[:, None] * 24 + np.array(sorted(checks))[None, :]).ravel()
    bounds = np.append(hours, 24 * days)
    seg = price[bounds[1:]] / price[bounds[:-1]] - 1
    tgt = np.asarray(w)[hours] > 0.5
    pos = np.zeros((len(hours), SIMS))
    cur = np.zeros(SIMS, dtype=bool)
    out = np.zeros(SIMS, dtype=bool)  # sitting out a missed trade
    draws = rng.random((len(hours), SIMS))
    for i in range(len(hours)):
        if tgt[i]:
            enter = ~cur & ~out
            miss = enter & (draws[i] < P_MISS)
            out |= miss
            cur |= enter & ~miss
        else:
            cur[:] = False
            out[:] = False
        pos[i] = cur
    trade = np.abs(np.diff(pos, axis=0, prepend=np.zeros((1, SIMS))))
    net = 1 + pos * seg[:, None] - P_COST * trade
    return np.prod(net.reshape(days, len(checks), SIMS), axis=1) - 1


# ---------- placebo ----------

def _placebo(price: np.ndarray, w: np.ndarray, gate: np.ndarray, lo_day: int, hi_day: int,
             rng: np.random.Generator) -> tuple[float, np.ndarray, int]:
    """Yearly return of the rule and of DRAWS random rules with the same trade lengths inside the same gate."""
    k = len(price)
    lo, hi = 24 * lo_day, 24 * hi_day
    act = np.zeros(k)
    act[lo:hi] = np.asarray(w[lo:hi]) > 0.5
    seg = act[lo:hi] > 0.5
    edges = np.flatnonzero(np.diff(np.concatenate([[0], seg.astype(int), [0]])))
    lens = edges[1::2] - edges[0::2]
    inside = np.flatnonzero(np.asarray(gate[lo:hi]) > 0.5) + lo
    free = len(inside) - int(lens.sum())
    base = cr._stats(_days_net(price, act, COST)[lo_day:hi_day])["ret"]
    rets = np.zeros(DRAWS)
    n = len(lens)
    for d in range(DRAWS):
        ln = lens[rng.permutation(n)]
        before = np.concatenate([[0], np.cumsum(ln)[:-1]])
        cuts = np.sort(rng.integers(0, free + 1, n))
        first = np.repeat(cuts + before, ln)
        within = np.arange(int(ln.sum())) - np.repeat(before, ln)
        rand = np.zeros(k)
        rand[inside[first + within]] = 1.0
        rets[d] = cr._stats(_days_net(price, rand, COST)[lo_day:hi_day])["ret"]
    return float(base), rets, n


# ---------- the computation ----------

def _other(symbol: str) -> tuple[dict | None, str | None]:
    rows, problem = it._hourly(symbol)
    try:
        price, b0, _ = cr._grid(rows)
        days = (len(price) - 1) // 24
        dates = pd.date_range(b0, periods=days, freq="D")
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        starts, labels = msc._month_starts(dates, b0 + pd.Timedelta(days=60), three)
        if not len(starts):
            raise RuntimeError("沒有起算日")
        t = _targets(price, 0, np.array([]))
        bound = price[::24]
        out = {"labels": labels, "hold": msc._suffix(msc._equity(bound[1:] / bound[:-1] - 1), starts, YEAR)[0][:, 0]}
        for key in RULES:
            out[key] = msc._suffix(msc._equity(_days_net(price, t[key], COST)), starts, YEAR)[0][:, 0]
        return out, problem
    except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
        return None, (problem or "") + f"{symbol[:-4]} 的 1 小時線這次拿不到或太短（{type(exc).__name__}），沒有算進去。"


def _compute() -> dict:
    rows, problem = it._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, missing = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    k = len(price)
    days = (k - 1) // 24
    dates = pd.date_range(b0, periods=days, freq="D")
    at = lambda day: int(dates.searchsorted(pd.Timestamp(day)))  # noqa: E731
    bound = price[::24]
    change = bound[1:] / bound[:-1] - 1
    opens = _opens(b0, k)

    # ----- Q1 replication -----
    q1 = _q1(price, opens, pre)
    rep_lo, rep_hi = at(REP[0]), min(at(REP[1]), days)
    rep = {v: {c: _measure(price, q1[v], rep_lo, rep_hi, c) for c in COSTS} for v in ("v1", "v2", "v3")}
    rep_full = {v: _measure(price, q1[v], rep_lo, days, COST) for v in ("v1", "v2", "v3")}
    rep_hold = cr._stats(change[rep_lo:rep_hi])
    rep_hold["sharpe"] = _sharpe(change[rep_lo:rep_hi])

    # ----- A. textbook grid -----
    starts, labels = msc._month_starts(dates, STARTS[0], STARTS[1])
    hold_r, hold_d = msc._suffix(msc._equity(change), starts, YEAR)
    hold_r, hold_d = hold_r[:, 0], hold_d[:, 0]
    grid = {key: (np.zeros((len(starts), 24)), np.zeros((len(starts), 24))) for key in RULES}
    t0 = None
    for h in range(24):
        t = _targets(price, h, pre)
        if h == 0:
            t0 = t
        for key in RULES:
            r, dd = msc._suffix(msc._equity(_days_net(price, t[key], COST)), starts, YEAR)
            grid[key][0][:, h], grid[key][1][:, h] = r[:, 0], dd[:, 0]

    # ----- clock 0, whole period, costs -----
    m0 = at(MEASURE)
    main = {key: {c: _measure(price, t0[key], m0, days, c) for c in COSTS} for key in RULES}
    hold_main = cr._stats(change[m0:])

    # ----- B. ordinary person -----
    person = {}
    for name, checks in (("3", CHECKS3), ("2", CHECKS2)):
        for ki, key in enumerate(RULES):
            rng = np.random.default_rng(SEED + 100 * ki + len(checks))
            net = _person(price, t0[key], checks, days, rng)
            r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), np.cumprod(1 + net, axis=0)]), starts, YEAR)
            person[(name, key)] = (r, dd)

    # ----- placebo -----
    placebo = {}
    for ki, key in enumerate(("q2", "q3")):
        rng = np.random.default_rng(SEED + 7 + ki)
        placebo[key] = _placebo(price, t0[key], t0["ma50"], m0, days, rng)

    # ----- C. other coins -----
    coins, problems = {}, [problem] if problem else []
    for coin in COINS:
        got, prob = _other(f"{coin}USDT")
        if got:
            coins[coin] = got
        if prob:
            problems.append(prob)

    # ----- summary -----
    s = {}
    for key in RULES:
        tr, td = grid[key]
        edge = tr - hold_r[:, None]
        pr3, pd3 = person[("3", key)]
        pr2, _ = person[("2", key)]
        o_n = sum(len(c["labels"]) for c in coins.values())
        o_beat = sum(int((c[key] > c["hold"]).sum()) for c in coins.values())
        x = {"a_n": edge.size, "a_beat": int((edge > 0).sum()), "a_edge": float(np.median(edge)),
             "a_p10": float(np.percentile(edge, 10)), "a_deep_med": float(np.median(td)), "a_deep_worst": float(td.min()),
             "b_n": pr3.size, "b_beat": int((pr3 - hold_r[:, None] > 0).sum()),
             "b_edge": float(np.median(pr3 - hold_r[:, None])), "b_deep_med": float(np.median(pd3)),
             "b2_beat": int((pr2 - hold_r[:, None] > 0).sum()), "b2_n": pr2.size,
             "o_n": o_n, "o_beat": o_beat,
             "o_each": {c: (len(v["labels"]), int((v[key] > v["hold"]).sum()),
                            int((v[key] > v["ma50"]).sum())) for c, v in coins.items()}}
        x["checks"] = [x["a_beat"] / x["a_n"], x["b_beat"] / x["b_n"], (o_beat / o_n) if o_n else 0.0]
        x["score"] = min(x["checks"])
        s[key] = x
    for key in ("q2", "q3"):
        x = s[key]
        x["vs_a"] = float(np.median(grid[key][0] - grid["ma50"][0]))
        x["vs_a_beat"] = int((grid[key][0] > grid["ma50"][0]).sum())
        x["vs_b3"] = float(np.median(person[("3", key)][0] - person[("3", "ma50")][0]))
        x["vs_b3_beat"] = int((person[("3", key)][0] > person[("3", "ma50")][0]).sum())
        x["vs_b2"] = float(np.median(person[("2", key)][0] - person[("2", "ma50")][0]))
        x["vs_b2_beat"] = int((person[("2", key)][0] > person[("2", "ma50")][0]).sum())
        base, rets, n_tr = placebo[key]
        x["placebo_p"] = float(np.mean(rets >= base))
        x["placebo_med"] = float(np.median(rets))
        x["placebo_base"] = base
        x["pass_rank"] = x["score"] >= s["ma50"]["score"]
        x["pass_deep"] = x["a_deep_med"] >= s["ma50"]["a_deep_med"]
        x["pass_placebo"] = x["placebo_p"] <= P_BAR
        x["pass_person"] = x["vs_b3"] > 0
        x["passed"] = all((x["pass_rank"], x["pass_deep"], x["pass_placebo"], x["pass_person"]))

    now = {"ma50": float(t0["ma50"][-1]), "q2": float(t0["q2"][-1]), "q3": float(t0["q3"][-1]),
           "v3": float(q1["v3"][-1]), "dgate": float(q1["gate"][-1])}
    return {"price": price, "b0": b0, "missing": missing, "days": days, "dates": dates, "rep": rep, "rep_full": rep_full,
            "rep_hold": rep_hold, "rep_span": (dates[rep_lo], dates[rep_hi - 1]), "labels": labels,
            "hold_r": hold_r, "hold_d": hold_d, "grid": grid, "main": main, "hold_main": hold_main, "person": person,
            "placebo": placebo, "coins": coins, "problems": problems, "s": s, "now": now,
            "opens_ok": float(np.isfinite(opens[24 * rep_lo:24 * rep_hi]).mean() * 100),
            "last": dates[-1], "last_price": float(price[-1])}


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    s, main, rep = r["s"], r["main"], r["rep"]
    last = r["last"]
    n_st = len(r["labels"])
    n_a = n_st * 24
    m50, q2, q3 = s["ma50"], s["q2"], s["q3"]
    years_main = main["ma50"][COST]["years"]

    def share(n: int, m: int) -> str:
        return f"{m:,}／{n:,}（{m / n * 100:.0f}%）" if n else "—"

    # ----- table 1: Q1 replication -----
    r0, r1 = r["rep_span"]
    rows1 = []
    for v in ("v1", "v2", "v3"):
        x0, x1, x25 = rep[v][0.0], rep[v][0.001], rep[v][0.0025]
        rows1.append([Q1N[v], Q1PUB[v], f"{_num(x0['sharpe'])}｜{_pct(x0['ret'])}｜{_pct(x0['deepest'])}",
                      f"{_num(x1['sharpe'])}｜{_pct(x1['ret'])}｜{_pct(x1['deepest'])}",
                      f"{_num(x25['sharpe'])}｜{_pct(x25['ret'])}",
                      f"{x0['trades']:,} 筆（一年 {x0['trades'] / x0['years']:.0f} 筆）",
                      f"{x0['share']:.1f}%"])
    h = r["rep_hold"]
    rows1.append(["對照：一直抱著（不扣成本）", "—", f"{_num(h['sharpe'])}｜{_pct(h['ret'])}｜{_pct(h['deepest'])}",
                  "—", "—", "—", "100%"])
    v2s, v3s = rep["v2"][0.0]["sharpe"], rep["v3"][0.0]["sharpe"]
    replicated = (abs(v2s - 0.80) <= 0.25 and abs(v3s - 1.07) <= 0.25
                  and rep["v1"][0.0]["sharpe"] < v2s < v3s)

    # ----- table 2: clock 0, whole period -----
    rows2 = [["一直抱著（不扣成本）", f"{_pct(r['hold_main']['ret'])}｜{_num(r['hold_main']['ratio'])}",
              _pct(r["hold_main"]["deepest"]), "—", "—", "—", "100%"]]
    for key in RULES:
        x0, x1, x25 = main[key][0.0], main[key][0.001], main[key][0.0025]
        rows2.append([NAMES[key], f"{_pct(x1['ret'])}｜{_num(x1['ratio'])}", _pct(x1["deepest"]),
                      _pct(x0["ret"]), _pct(x25["ret"]), f"{x1['trades'] / x1['years']:.1f} 次",
                      f"{x1['share']:.0f}%"])

    # ----- table 3: fair exam -----
    rows3 = []
    for key in RULES:
        x = s[key]
        rows3.append([NAMES[key], share(x["a_n"], x["a_beat"]), f"{_pp(x['a_edge'])}｜{_pp(x['a_p10'])}",
                      f"{_pct(x['a_deep_med'])}｜{_pct(x['a_deep_worst'])}",
                      share(x["b_n"], x["b_beat"]), share(x["b2_n"], x["b2_beat"]), share(x["o_n"], x["o_beat"]),
                      f"{x['score'] * 100:.0f}%"])
    hd = r["hold_d"]
    rows3.append(["對照：一直抱著", "—", "—", f"{_pct(float(np.median(hd)))}｜{_pct(float(hd.min()))}", "—", "—", "—", "—"])

    # ----- table 4: against ma50 and the pass test -----
    def yes(b: bool) -> str:
        return "過" if b else "沒過"

    rows4 = []
    for key in ("q2", "q3"):
        x = s[key]
        rows4.append([NAMES[key],
                      f"{x['score'] * 100:.0f}% 對 {m50['score'] * 100:.0f}%：{yes(x['pass_rank'])}",
                      f"{_pct(x['a_deep_med'])} 對 {_pct(m50['a_deep_med'])}：{yes(x['pass_deep'])}",
                      f"{_pct(x['placebo_base'])} 對隨機中位數 {_pct(x['placebo_med'])}，p = {x['placebo_p']:.2f}："
                      f"{yes(x['pass_placebo'])}",
                      f"{_pp(x['vs_b3'])}（{share(x['b_n'], x['vs_b3_beat'])} 贏）：{yes(x['pass_person'])}",
                      f"{_pp(x['vs_b2'])}（{share(x['b2_n'], x['vs_b2_beat'])} 贏）",
                      f"{_pp(x['vs_a'])}（{share(x['a_n'], x['vs_a_beat'])} 贏）",
                      "通過" if x["passed"] else "沒通過"])

    # ----- table 5: other coins -----
    rows5 = []
    for coin, c in r["coins"].items():
        cells = [f"{coin}（{_d(c['labels'][0])[:7]} 到 {_d(c['labels'][-1])[:7]}，{len(c['labels'])} 個）"]
        for key in RULES:
            n, b_hold, b_ma = s[key]["o_each"][coin]
            cells.append(f"{b_hold}／{n}" + ("" if key == "ma50" else f"｜贏 50 日均線 {b_ma}／{n}"))
        rows5.append(cells)

    tables = [
        {"caption": (f"Q1 照原文重做：幣安 BTCUSDT 1 小時線，{_d(r0)} 到 {_d(r1)}（原文是 Gemini BTC/USD 2018-12 到 2025-11）。"
                     "格子是「夏普值（每日報酬平均÷標準差×√365，不扣無風險利率）｜每年報酬｜最深虧損」。日線 MACD 只在 UTC 0 點日線收完後才讀；"
                     "「收黑」用 1 小時 K 線的開盤價比收盤價。交易筆數是進場次數，一筆來回算一筆。"),
         "columns": ["做法", "原文的夏普值", "不扣成本", "每邊扣 0.10%", "每邊扣 0.25%：夏普值｜每年報酬", "交易筆數",
                     "在場時間"],
         "rows": rows1},
        {"caption": (f"Q2、Q3 和 50 日均線，比特幣，{MEASURE} 到 {_d(last)}，UTC 0 點日線、4 小時線在 UTC 0／4／8… 點收。"
                     "第二、三欄每邊扣 0.10%；「一年交易」是一年平均進場幾次。"),
         "columns": ["做法", "每年報酬｜報酬÷波動（0.10%）", "最深虧損（0.10%）", "不扣成本時每年報酬", "每邊扣 0.25% 時每年報酬",
                     "一年交易", "在場時間"],
         "rows": rows2},
        {"caption": (f"公平考試。A：比特幣 {STARTS[0][:7]} 到 {STARTS[1][:7]} 每個月 1 日起算（{n_st} 個）× 24 個時鐘（整套規則一起往後挪 "
                     f"0 到 23 小時：日線在那個整點收，4 小時線也跟著那個整點），共 {n_a:,} 格，每邊扣 0.10%，和同一天起一直抱著比。"
                     f"B：一般人一天只在台灣早上 8 點、中午 12 點、晚上 10 點看三次（另一種是只看早上 8 點和晚上 10 點兩次），只在那時候照規則調整，"
                     f"每邊扣 {P_COST * 100:.2f}%、要進場時有 {P_MISS * 100:.0f}% 機會漏掉（漏掉就整筆不做），每個起算日模擬 {SIMS} 次。"
                     "C：ETH、SOL、BNB、XRP，UTC 0 點，從規則都有值起每個月 1 日起算到三年前。「排名分數」是 A、B（看三次）、C 三個「贏過一直抱著的比例」裡最低的。"),
         "columns": ["做法", "A 贏過抱著的格", "A 多賺幾個百分點：中位數｜最差一成", "A 最深虧損：中位數｜最深一格",
                     "B 看三次：贏過抱著", "B 看兩次：贏過抱著", "C 其他幣：贏過抱著", "排名分數"],
         "rows": rows3},
        {"caption": ("事先定好的通過條件（四項都要過）：排名分數不低於 50 日均線；A 的最深虧損中位數不比 50 日均線深；安慰劑——把同樣筆數、"
                     f"同樣長度的交易隨機放進 50 日均線閘門開著的時間裡 {DRAWS} 次，隨機的每年報酬不低於規則本身的比例 p 要 ≤ {P_BAR:.2f}"
                     f"（比特幣 {MEASURE} 起，UTC 0 點，0.10%）；一般人看三次的版本，每年報酬比同樣方式做的 50 日均線多（中位數 > 0）。"
                     "後兩欄只是參考：看兩次的版本、照課本做時和 50 日均線的差距。"),
         "columns": ["做法", "排名分數", "最深虧損中位數", "安慰劑（每年報酬）", "一般人看三次：比 50 日均線多（百分點）",
                     "參考：看兩次", "參考：照課本 A", "結果"],
         "rows": rows4},
        {"caption": "其他幣（C）各自的結果：「贏過抱著的起算日／起算日個數」，Q2、Q3 另外列贏過同一幣 50 日均線的個數。成本 0.10%，UTC 0 點。",
         "columns": ["幣（起算日範圍）"] + [SHORT[k] for k in RULES],
         "rows": rows5},
    ]

    # ----- today -----
    def state(v: float) -> str:
        return "持有" if v > 0.5 else "現金"

    nw = r["now"]
    now = [
        {"label": f"最後一個收完的日子（{_d(last)}，UTC 0 點收）", "value": f"比特幣 {r['last_price']:,.0f} 美元"},
        {"label": "日線 50 日均線（閘門）", "value": "開（收盤在均線上）" if nw["ma50"] > 0.5 else "關（收盤在均線下）"},
        {"label": SHORT["q2"], "value": state(nw["q2"])},
        {"label": SHORT["q3"], "value": state(nw["q3"])},
        {"label": "原文的日線 MACD 閘門", "value": "開" if nw["dgate"] > 0.5 else "關"},
        {"label": "原文 V3（＋STOP）", "value": state(nw["v3"])},
        {"label": "這幾格的意思", "value": "只是描述規則在最後一個收完的整點停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    passed = [k for k in ("q2", "q3") if s[k]["passed"]]
    if passed:
        head = "、".join(SHORT[k] for k in passed) + " 通過事先定好的四項條件，可以考慮當成 50 日均線的改良版。"
    else:
        head = "Q2、Q3 都沒有通過事先定好的四項條件，日線閘門裡改用 4 小時線抓進出點，沒有證據比單純的 50 日均線好。"
    rep_text = (f"原文的規則在幣安比特幣上{'大致重現' if replicated else '沒有重現'}：不扣成本時夏普值 V1 {_num(rep['v1'][0.0]['sharpe'])}、"
                f"V2 {_num(v2s)}、V3 {_num(v3s)}（原文 0.33／0.80／1.07），每邊扣 0.10% 變成 "
                f"{_num(rep['v1'][0.001]['sharpe'])}／{_num(rep['v2'][0.001]['sharpe'])}／{_num(rep['v3'][0.001]['sharpe'])}"
                f"（V1 {rep['v1'][0.0]['trades']:,} 筆、V2 {rep['v2'][0.0]['trades']:,} 筆、V3 {rep['v3'][0.0]['trades']:,} 筆）。")

    def brief(k: str) -> str:
        x = s[k]
        return (f"{SHORT[k]}排名分數 {x['score'] * 100:.0f}%（50 日均線 {m50['score'] * 100:.0f}%）、最深虧損中位數 "
                f"{_pct(x['a_deep_med'])}（{_pct(m50['a_deep_med'])}）、安慰劑 p = {x['placebo_p']:.2f}、一般人看三次比 50 日均線多 "
                f"{_pp(x['vs_b3'])} 點")

    verdict = head + rep_text + brief("q2") + "；" + brief("q3") + "。"

    rf = r["rep_full"]
    notes = [
        "比較的問題：Quantpedia 說「日線趨勢只負責准不准做、1 小時線負責抓進出點」能把夏普值從 0.33 提高到 0.80／1.07、最深虧損減半。"
        "這裡先照原文重做（Q1），再把同樣的結構套到儀表板的主訊號：50 日均線當閘門，4 小時線抓進出（Q2、Q3），看能不能改良主訊號。",
        "所有規則、成本、考試方式和通過條件都在看結果之前寫進程式開頭的說明裡，跑完沒有改。",
        f"Q1 的細節原文沒寫清楚的地方：日線 MACD 也用 12／26／9；閘門轉空時 V3 已經持有的部位不平倉（只有收黑才出）；收盤等於開盤不算收黑；"
        f"成交價都用收盤價。開盤價取自幣安 1 小時 K 線（{r['opens_ok']:.0f}% 的小時有，其餘用前一小時收盤代替）。"
        f"原文用 Gemini，這裡用幣安，價格和停機時段不同。延長到 {_d(last)}、扣 0.10% 時夏普值：V1 {_num(rf['v1']['sharpe'])}、"
        f"V2 {_num(rf['v2']['sharpe'])}、V3 {_num(rf['v3']['sharpe'])}。",
        "Q2 要「新的」4 小時 MACD 黃金交叉才進場：閘門打開時 MACD 已經在訊號線上面，要等它掉下去再交叉上來；閘門關掉就立刻出場。",
        f"安慰劑把交易放進「閘門開著的小時」排成的時間軸上，一筆隨機交易如果跨過閘門關掉的空檔，會被切成兩段，所以隨機版本的筆數可能略多、成本略高"
        f"（Q2 實際 {r['placebo']['q2'][2]:,} 筆、Q3 {r['placebo']['q3'][2]:,} 筆）。",
        "一般人版本：一天只在固定的台灣時間看盤、看的當下照規則調整；50 日均線也用同樣方式做，所以兩者是公平比較。"
        "台灣早上 8 點剛好是 UTC 0 點日線收完的時候，所以 50 日均線一般人版本幾乎沒有延遲；4 小時線的訊號最多要等到下一次看盤才做。",
        f"每邊 0.10% 是幣安現貨一般帳戶手續費；0.25% 粗略加上價差和滑價。成交價一律用整點收盤價，滑價沒有另外算，交易越多的規則這部分低估越多。"
        f"比特幣的 1 小時線整段缺 {r['missing']:,} 個小時（幣安早年停機維護），缺的小時沿用前一個小時的價格。",
        "相鄰的起算日、相鄰的時鐘大部分歷史相同，這些「幾格裡幾格」不是互相獨立的證據；其他幣是今天還在的大幣，有倖存者偏誤。",
        "現金假設沒有利息、沒有算稅；一直抱著不扣成本。過去的結果不保證未來會一樣，這裡不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "mtf_gate",
        "title": "日線准許、4 小時線抓時機",
        "question": ("讓日線趨勢（50 日均線）只負責「准不准做多」，在准許的時間裡改用 4 小時線決定何時進出，"
                     "會比單純的 50 日均線好嗎？Quantpedia 用日線 MACD＋1 小時 MACD 報告的進步，在幣安資料上重做得出來嗎？"),
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Quantpedia：How to Design a Simple Multi-Timeframe Trend Strategy on Bitcoin（2025-11-13）",
             "url": "https://quantpedia.com/how-to-design-a-simple-multi-timeframe-trend-strategy-on-bitcoin/"},
            {"name": "幣安現貨 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "幣安公開行情 API（K 線）",
             "url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"},
            {"name": "幣安現貨手續費", "url": "https://www.binance.com/zh-TC/fee/trading"},
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前暖機用）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
