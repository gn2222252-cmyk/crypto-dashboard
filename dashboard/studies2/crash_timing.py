"""Historical study "crash_timing": every trend rule leaves bitcoin's big bear markets late and comes back late. The
whole edge of the 50-day rule is sitting out the middle of these crashes. Can a rule leave earlier and come back
earlier than "close above the 50-day average" (ma50, the page's main signal) without losing more to whipsaws - judged
across start dates, closing hours, ordinary-person execution and other coins exactly as main_signal_check judges?

Candidates, all written down before any result was looked at (long or cash, decided at the close, held from the
next day, 0.10% of the amount traded):
1. Fast exit / slow entry on averages: while holding, out on a close below the X-day average; in cash, back in on a
   close above both the Y-day and the X-day average, X in {10, 20, 30} x Y in {30, 50, 100}. Also "20-day average
   both ways" (a20_20), "out on a 10-day closing low, in on a 55-day closing high" (b55_10) and slow exit / fast
   entry (s50_20: in on a fresh cross above the 20-day average or any close above the 50-day; once the close has
   been above the 50-day average the position leaves only below it, before that already below the 20-day).
2. Drawdown exit (dd10/15/20): ma50, and also out when the close is X% below the highest close since entry; after
   such an exit back in only once the 50-day rule has been off and turns on again, or the close tops the old peak.
3. Slope of the average (slope5/10): hold while the 50-day average is above its value 5 / 10 days earlier.
4. Faster re-entry (r20): hold when ma50 holds, or when the close is above a 20-day average that is above its value
   10 days earlier.
5. Volatility exit (volx): ma50, but out while 10-day volatility is above twice its 90-day median and the close is
   below the 20-day average.
6. Partial (part): half on the 20-day average, half on the 50-day average.
Benchmarks: ma50, 55/20, "17 rules, at least 9 long -> all in" (thresh), holding.

Measures: (a) each bitcoin bear market of 50% or more since 2013 (a close 50% under the running peak; it ends when
the close doubles from the low): exit and re-entry against top and low, and the share of the crash avoided / the
recovery caught, from the rule's own gross growth over top->low and low->next top (so whipsaws inside a crash
count); (b) whipsaws: exits followed within 30 days by an entry at a higher close; (c) main_signal_check's fair grid
re-run with its own code pieces for every rule (108 daily start dates, 72 start months x 24 closing hours, ordinary
person with 200 simulations per clock, ETH/SOL/BNB/XRP/MSTR) and its pre-stated ranking score.
A candidate counts as an improvement only if, against ma50, it avoids a larger median share of the crashes, has a
ranking score no lower and a median deepest loss (1,728 cells) no deeper.

No look-ahead: every position uses closes up to and including the decision close (checked by perturbing the future
in study()). Data: the dashboard's long daily bitcoin series, the Binance hourly store read through
clock_robustness, Binance / Yahoo daily closes for the other assets through trend_assets. Public, key-less only.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

COST = msc.COST
P_COST, P_MISS, SIMS, SEED = msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
BEAR_FROM = "2013-01-01"
BEAR_DROP, BEAR_END = 0.50, 1.00  # a close 50% under the peak starts a bear; doubling from the low ends it
WHIP_DAYS = 30
LOSS_BAR = msc.LOSS_BAR

GRID_X, GRID_Y = (10, 20, 30), (30, 50, 100)
GRID = tuple(f"a{x}_{y}" for x in GRID_X for y in GRID_Y)
BENCH = ("ma50", "b5520", "thresh")
OTHERS = ("a20_20", "b55_10", "s50_20", "dd10", "dd15", "dd20", "slope5", "slope10", "r20", "volx", "part")
CANDS = GRID + OTHERS
KEYS = BENCH + CANDS
MSC_KI = {"b5520": 0, "ma50": 1, "thresh": 2}  # same random draws as main_signal_check for the benchmarks

_pct, _pp, _d = cr._pct, cr._pp, cr._d


def _name(k: str) -> str:
    fixed = {
        "ma50": "50 日均線（現行主訊號）", "b5520": "55／20", "thresh": "17 條過半",
        "a20_20": "20 日均線（進出都快）", "b55_10": "10 日新低下車、55 日新高上車",
        "s50_20": "慢下快上：破 50 日均線下車、重新站上 20 日均線就上車",
        "slope5": "50 日均線本身比 5 天前高才持有", "slope10": "50 日均線本身比 10 天前高才持有",
        "r20": "50 日均線，或站上往上走的 20 日均線也持有",
        "volx": "50 日均線＋波動暴增且跌破 20 日均線就下車", "part": "一半 20 日均線、一半 50 日均線",
    }
    if k in fixed:
        return fixed[k]
    if k.startswith("dd"):
        return f"50 日均線＋從進場後最高收盤跌 {k[2:]}% 就下車"
    x, y = k[1:].split("_")
    return f"跌破 {x} 日均線下車、站上 {y} 日均線上車"


def _short(k: str) -> str:
    fixed = {"ma50": "50 日均線", "b5520": "55／20", "thresh": "17 條過半", "a20_20": "20 日均線",
             "b55_10": "10 日低出／55 日高進", "s50_20": "慢下快上", "r20": "50 日或往上的 20 日",
             "volx": "50 日＋波動暴增出場", "part": "一半 20 日、一半 50 日"}
    if k in fixed:
        return fixed[k]
    if k.startswith("dd"):
        return f"50 日＋回落 {k[2:]}% 出場"
    if k.startswith("slope"):
        return f"50 日均線 {k[5:]} 天上升"
    x, y = k[1:].split("_")
    return f"破 {x} 日出、上 {y} 日進"


# ---------- the rules ----------

def _mean(c: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(c).rolling(n, min_periods=n).mean().values


def _ago(a: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k, np.nan), a[:-k]])


def _gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 where a > b, 0 where not, NaN where either is missing."""
    return np.where(np.isnan(a) | np.isnan(b), np.nan, (a > b).astype(float))


def _asym(c: list, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """Holding: out on a close below mx. Cash: in on a close above my and above mx."""
    out = np.full(len(c), np.nan)
    a, b, st = mx.tolist(), my.tolist(), 0
    for i, x in enumerate(c):
        if a[i] != a[i] or b[i] != b[i]:
            continue
        if st:
            if x < a[i]:
                st = 0
        elif x > b[i] and x > a[i]:
            st = 1
        out[i] = st
    return out


def _slow_fast(c: list, m20: np.ndarray, m50: np.ndarray) -> np.ndarray:
    out = np.full(len(c), np.nan)
    a, b = m20.tolist(), m50.tolist()
    st, flag = 0, False
    for i in range(1, len(c)):
        x = c[i]
        if a[i] != a[i] or b[i] != b[i] or a[i - 1] != a[i - 1]:
            continue
        if st == 0:
            if x > b[i] or (x > a[i] and c[i - 1] <= a[i - 1]):
                st, flag = 1, x > b[i]
        else:
            if x > b[i]:
                flag = True
            if (flag and x < b[i]) or (not flag and x < a[i]):
                st = 0
        out[i] = st
    return out


def _drawdown(c: list, m50: np.ndarray, drop: float) -> np.ndarray:
    out = np.full(len(c), np.nan)
    m = m50.tolist()
    st, peak, armed, old_peak = 0, 0.0, True, math.inf
    for i, x in enumerate(c):
        if m[i] != m[i]:
            continue
        if st == 0:
            if x <= m[i]:
                armed = True
            if (x > m[i] and armed) or x > old_peak:
                st, peak = 1, x
        else:
            peak = max(peak, x)
            if x < m[i] or x <= (1 - drop) * peak:
                st, old_peak, armed = 0, peak, x <= m[i]
        out[i] = st
    return out


def _breakout(c: np.ndarray, n_in: int, n_out: int) -> np.ndarray:
    high = pd.Series(c).rolling(n_in, min_periods=n_in).max().values
    return np.where(np.isnan(high), np.nan, cr._breakout(pd.Series(c), n_in, n_out))


def _legs(c, keys=KEYS) -> dict[str, list[tuple[float, np.ndarray]]]:
    """(weight, 1/0/NaN position after each close) of the on/off parts every rule is made of."""
    c = np.asarray(c, dtype=float)
    cl = c.tolist()
    means = {n: _mean(c, n) for n in (10, 20, 30, 50, 100)}
    m20, m50 = means[20], means[50]
    ma50 = _gt(c, m50)
    out = {}
    for k in keys:
        if k == "ma50":
            leg = ma50
        elif k == "b5520":
            leg = _breakout(c, 55, 20)
        elif k == "b55_10":
            leg = _breakout(c, 55, 10)
        elif k == "thresh":
            signal, _ = te._signals(pd.Series(c))
            share = signal.mean(axis=1).where(signal.notna().all(axis=1)).values
            leg = np.where(np.isnan(share), np.nan, (share >= 0.5).astype(float))
        elif k == "a20_20":
            leg = _gt(c, m20)
        elif k in GRID:
            x, y = (int(v) for v in k[1:].split("_"))
            leg = _asym(cl, means[x], means[y])
        elif k == "s50_20":
            leg = _slow_fast(cl, m20, m50)
        elif k.startswith("dd"):
            leg = _drawdown(cl, m50, int(k[2:]) / 100)
        elif k.startswith("slope"):
            leg = _gt(m50, _ago(m50, int(k[5:])))
        elif k == "r20":
            fast = np.where(np.isnan(_ago(m20, 10)), np.nan, ((c > m20) & (m20 > _ago(m20, 10))).astype(float))
            leg = np.where(np.isnan(ma50) | np.isnan(fast), np.nan, np.maximum(ma50, fast))
        elif k == "volx":
            vol = pd.Series(c).pct_change().rolling(10, min_periods=10).std(ddof=1)
            med = vol.rolling(90, min_periods=90).median().values
            storm = (vol.values > 2 * med) & (c < m20)
            leg = np.where(np.isnan(med) | np.isnan(ma50), np.nan, ma50 * (~storm))
        elif k == "part":
            out[k] = [(0.5, _gt(c, m20)), (0.5, ma50)]
            continue
        else:
            raise KeyError(k)
        out[k] = [(1.0, leg)]
    return out


def _wants(c, keys=KEYS) -> pd.DataFrame:
    """Share each rule wants after each close (NaN until every part has a value)."""
    legs = _legs(c, keys)
    return pd.DataFrame({k: sum(w * leg for w, leg in parts) for k, parts in legs.items()})


# ---------- the fair grid (main_signal_check's pieces) ----------

def _daily(close: pd.Series, year: int, first, last) -> dict:
    target = _wants(close.values)
    target.index = close.index
    change = close.pct_change().fillna(0.0).values
    valid = target.notna().all(axis=1)
    begin = max(close.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1), pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels, "target": target}
    hr, hd = msc._suffix(msc._equity(change), starts, year)
    out["hold"] = (hr[:, 0], hd[:, 0])
    for k in KEYS:
        net, _ = te._run(target[k].shift(1).values, change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[k] = (r[:, 0], dd[:, 0])
    return out


def _clock(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray) -> dict:
    """72 start months x 24 clocks: textbook and ordinary person, summarised per rule."""
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    starts, labels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
    hr, hd = msc._suffix(msc._equity(m.change), starts, YEAR)
    hr, hd = hr[:, 0], hd[:, 0]
    wants = []
    for h in range(24):
        t = _wants(np.concatenate([pre, price[h::24]])).fillna(0.0)
        wants.append({k: t[k].values[npre:][:days].astype(float) for k in KEYS})
    out = {}
    for j, k in enumerate(KEYS):
        ki = MSC_KI.get(k, 10 + j)
        tr, td = np.zeros((len(starts), 24)), np.zeros((len(starts), 24))
        pr, pdd = np.zeros((len(starts), 24, SIMS)), np.zeros((len(starts), 24, SIMS))
        for h in range(24):
            w = wants[h][k]
            r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(SEED + 1000 * ki + h)
            pos = rt._missed(wl, P_MISS, rng, SIMS)
            r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), eng._mid(b, pos, P_COST, None)]), starts, YEAR)
            pr[:, h, :], pdd[:, h, :] = r, dd
        edge, pedge = tr - hr[:, None], pr - hr[:, None, None]
        out[k] = {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
                  "c_shallow": int((td > hd[:, None]).sum()), "c_deep_med": float(np.median(td)),
                  "c_deep_worst": float(td.min()),
                  "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
                  "p_deep_med": float(np.median(pdd))}
    return {"starts": labels, "rules": out, "hold_deep_med": float(np.median(hd))}


def _summary(daily: dict, clock: dict, other: dict) -> dict:
    s = {}
    for k in KEYS:
        dr, dd = daily[k]
        x = dict(clock["rules"][k])
        x["d_n"], x["d_beat"] = len(dr), int((dr > daily["hold"][0]).sum())
        x["d_shallow"] = int((dd > daily["hold"][1]).sum())
        x["o_each"] = {a: (len(o["starts"]), int((o[k][0] > o["hold"][0]).sum())) for a, o in other.items()}
        x["o_n"] = sum(v[0] for v in x["o_each"].values())
        x["o_beat"] = sum(v[1] for v in x["o_each"].values())
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else 0.0]
        x["low"] = min(x["checks"])
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        s[k] = x
    eligible = [k for k in KEYS if s[k]["loss_ok"] >= LOSS_BAR]
    order = sorted(KEYS, key=lambda k: (k not in eligible, -s[k]["low"], -s[k]["c_edge"]))
    return {"s": s, "eligible": eligible, "order": order}


# ---------- bear markets ----------

def _bears(close: pd.Series) -> list[dict]:
    """Bear markets of BEAR_DROP or more from BEAR_FROM: top, low, next top (end of the recovery), finished?"""
    first = int(close.index.searchsorted(pd.Timestamp(BEAR_FROM)))
    v = close.values
    out, peak, low, falling = [], first, first, False
    for i in range(first + 1, len(v)):
        if not falling:
            if v[i] > v[peak]:
                peak = i
            elif v[i] <= v[peak] * (1 - BEAR_DROP):
                falling, low = True, i
        else:
            if v[i] < v[low]:
                low = i
            elif v[i] >= v[low] * (1 + BEAR_END):
                out.append({"top": peak, "low": low, "done": True})
                falling, peak = False, i
    if falling:
        out.append({"top": peak, "low": low, "done": False})
    for j, b in enumerate(out):
        if b["done"]:
            stop = out[j + 1]["top"] if j + 1 < len(out) else len(v) - 1
            b["next"] = int(b["low"] + np.argmax(v[b["low"]:stop + 1]))
    return out


def _growth(pos: np.ndarray, c: np.ndarray, a: int, b: int) -> float:
    """Gross growth of a position path from close a to close b (position after close d-1 held during day d)."""
    p = np.nan_to_num(pos[a:b], nan=0.0)
    return float(np.prod(1 + p * (c[a + 1:b + 1] / c[a:b] - 1)))


def _events(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.nan_to_num(pos, nan=0.0)
    step = np.diff(p, prepend=0.0)
    return np.flatnonzero(step < -0.5), np.flatnonzero(step > 0.5)


def _crash(c: np.ndarray, legs: list, bear: dict) -> dict:
    t, lo = bear["top"], bear["low"]
    top, low = c[t], c[lo]
    g = sum(w * _growth(leg, c, t, lo) for w, leg in legs)
    exit_px = top * g
    out = {"exit_eq": exit_px / top - 1, "avoid": (exit_px - low) / (top - low)}
    if bear.get("done"):
        n = bear["next"]
        g2 = sum(w * _growth(leg, c, lo, n) for w, leg in legs)
        entry = c[n] / g2 if g2 > 0 else math.inf
        out["entry_eq"] = entry / low - 1
        out["catch"] = (c[n] - entry) / (c[n] - low)
    return out


def _first_moves(c: np.ndarray, leg: np.ndarray, bear: dict) -> tuple:
    """(exit index or None, entry index or None) as the rule actually traded around one bear market."""
    exits, entries = _events(leg)
    t, lo = bear["top"], bear["low"]
    p = np.nan_to_num(leg, nan=0.0)
    if p[t] > 0.5:
        after = exits[exits > t]
        ex = int(after[0]) if len(after) else None
    else:
        before = exits[exits <= t]
        ex = int(before[-1]) if len(before) else None
    en = None
    if bear.get("done"):
        if p[lo] > 0.5:
            before = entries[entries <= lo]
            en = int(before[-1]) if len(before) else None
        else:
            after = entries[(entries > lo) & (entries <= bear["next"])]
            en = int(after[0]) if len(after) else None
    return ex, en


def _whipsaws(c: np.ndarray, legs: list, first: int) -> dict:
    n_exit = n_whip = cost = 0.0
    for w, leg in legs:
        exits, entries = _events(leg)
        for e in exits[exits >= first]:
            n_exit += w
            nxt = entries[entries > e]
            if len(nxt) and nxt[0] - e <= WHIP_DAYS and c[nxt[0]] > c[e]:
                n_whip += w
                cost += w * (c[nxt[0]] / c[e] - 1 + 2 * COST)
    years = (len(c) - first) / YEAR
    return {"exits": n_exit / years, "whips": n_whip / years, "cost": cost / years * 100}


# ---------- today ----------

def _flip(c: np.ndarray, k: str, now: float) -> tuple:
    """Nearest next close (within -50% ... +60%) at which rule k would hold a different share than now."""
    last = float(c[-1])

    def held(p: float) -> float:
        return float(np.nan_to_num(_wants(np.append(c, p), (k,))[k].values[-1], nan=0.0))

    grid = last * np.exp(np.linspace(math.log(0.5), math.log(1.6), 61))
    states = [held(p) for p in grid]
    best = None
    i0 = int(np.argmin(np.abs(grid - last)))
    for i in range(len(grid) - 1):
        if abs(states[i] - states[i + 1]) > 1e-9 and (abs(states[i] - now) > 1e-9 or abs(states[i + 1] - now) > 1e-9):
            lo_p, hi_p, s_lo = grid[i], grid[i + 1], states[i]
            for _ in range(25):
                mid = math.sqrt(lo_p * hi_p)
                if abs(held(mid) - s_lo) < 1e-9:
                    lo_p = mid
                else:
                    hi_p = mid
            level = math.sqrt(lo_p * hi_p)
            new = states[i + 1] if level > last else states[i]
            dist = abs(i + 0.5 - i0)
            if abs(new - now) > 1e-9 and (best is None or dist < best[0]):
                best = (dist, level, new)
    return (best[1], best[2]) if best else (None, None)


def _word(v) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return "全部持有" if v >= 0.999 else "全部現金" if v <= 0.001 else f"持有 {v * 100:.0f}%"


# ---------- look-ahead check ----------

def _lookahead(c: np.ndarray) -> tuple[int, int]:
    """(cut points checked, rule paths that changed before the cut) when every close after the cut is scrambled."""
    base = _wants(c)
    rng = np.random.default_rng(7)
    bad = cuts = 0
    for cut in (len(c) - 900, len(c) - 400, len(c) - 60):
        noisy = c.copy()
        noisy[cut + 1:] *= np.exp(rng.normal(0, 0.2, len(c) - cut - 1))
        alt = _wants(noisy)
        cuts += 1
        for k in KEYS:
            a, b = base[k].values[:cut + 1], alt[k].values[:cut + 1]
            if not np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)):
                bad += 1
    return cuts, bad


# ---------- the study ----------

def _compute() -> dict:
    close = te._daily_close()
    c = close.values.astype(float)
    daily = _daily(close, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1])
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _clock(price, b0, pre)
    other_data, problems = msc._load_other()
    if problem:
        problems.insert(0, problem)
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    other = {}
    for key, (oc, since, year) in other_data.items():
        try:
            other[key] = _daily(oc, year, since or oc.index[0], three)
            if not other[key]["starts"]:
                raise RuntimeError("沒有起算日")
        except Exception as exc:  # noqa: BLE001 - one asset failing must not stop the others
            other.pop(key, None)
            problems.append(f"{key} 算不出來（{type(exc).__name__}），沒有算進去。")
    legs = _legs(c)
    bears = _bears(close)
    first = int(close.index.searchsorted(pd.Timestamp(BEAR_FROM)))
    crash = {k: [_crash(c, legs[k], b) for b in bears] for k in KEYS}
    whip = {k: _whipsaws(c, legs[k], first) for k in KEYS}
    return {"close": close, "c": c, "daily": daily, "clock": clock, "other": other, "problems": problems,
            "legs": legs, "bears": bears, "crash": crash, "whip": whip, "sm": _summary(daily, clock, other),
            "look": _lookahead(c)}


def _med(vals) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.median(vals)) if vals else None


def _avg(vals) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _share(v) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{v * 100:.0f}%"


def study() -> dict:
    r = _compute()
    close, c, bears, crash, whip = r["close"], r["c"], r["bears"], r["crash"], r["whip"]
    sm = r["sm"]
    s, order = sm["s"], sm["order"]
    done = [i for i, b in enumerate(bears) if b["done"]]
    open_ = [i for i, b in enumerate(bears) if not b["done"]]

    def stat(k: str, field: str, how=_med):
        return how([crash[k][i].get(field) for i in done])

    avoid = {k: stat(k, "avoid") for k in KEYS}
    base = "ma50"
    tests = {}
    for k in KEYS:
        a = avoid[k] is not None and avoid[k] > avoid[base] + 1e-12
        b = s[k]["low"] >= s[base]["low"] - 1e-12
        d = s[k]["c_deep_med"] >= s[base]["c_deep_med"] - 1e-12
        tests[k] = (a, b, d)
    winners = [k for k in CANDS if all(tests[k])]
    closest = sorted(CANDS, key=lambda k: (-sum(tests[k]), -(s[k]["low"] - s[base]["low"]),
                                            -((avoid[k] or 0) - (avoid[base] or 0))))[0]
    grid_order = [k for k in order if k in GRID]
    grid_best, grid_worst = grid_order[0], grid_order[-1]
    rows_keys = list(BENCH) + [grid_best, grid_worst] + list(OTHERS)
    label = {k: _name(k) for k in KEYS}
    label[grid_best] = f"快出慢進 9 格裡排名最前：{_name(grid_best)}"
    label[grid_worst] = f"快出慢進 9 格裡排名最後：{_name(grid_worst)}"
    if grid_best == grid_worst:
        rows_keys.remove(grid_worst)

    # table 1: crash by crash
    show = ["ma50", "b5520", winners[0] if winners else closest]

    def move(k: str, i: int, which: str) -> str:
        b = bears[i]
        leg = r["legs"][k][0][1] if len(r["legs"][k]) == 1 else None
        if leg is None:
            return "（分兩半，看第二表）"
        ex, en = _first_moves(c, leg, b)
        if which == "out":
            if ex is None or (b.get("done") and ex > b["next"]):
                return "沒下車"
            return f"{_d(close.index[ex])} {c[ex]:,.0f}（比高點 {(c[ex] / c[b['top']] - 1) * 100:+.0f}%）"
        if not b["done"]:
            return "熊市還沒走完"
        if en is None:
            return "回到高點前沒上車"
        return f"{_d(close.index[en])} {c[en]:,.0f}（比低點 {(c[en] / c[b['low']] - 1) * 100:+.0f}%）"

    rows1 = []
    for i, b in enumerate(bears):
        t, lo = b["top"], b["low"]
        head = f"{_d(close.index[t])} {c[t]:,.0f} → {_d(close.index[lo])} {c[lo]:,.0f}" + ("" if b["done"] else "（還在進行，低點未定）")
        rows1.append([head, _pct((c[lo] / c[t] - 1) * 100)] + [move(k, i, w) for k in show for w in ("out", "in")])

    # table 2: crash share vs whipsaws
    rows2 = []
    for k in rows_keys:
        w = whip[k]
        rows2.append([label[k], _pct(stat(k, "exit_eq") * 100), _pct(stat(k, "entry_eq") * 100),
                      f"{_share(stat(k, 'avoid', _avg))}｜{_share(avoid[k])}",
                      f"{_share(stat(k, 'catch', _avg))}｜{_share(stat(k, 'catch'))}",
                      f"{w['whips']:.1f} 次｜{w['cost']:.1f}%", f"{w['exits']:.1f} 次"])

    # table 3: fair grid
    n_d, n_c = s[base]["d_n"], s[base]["c_n"]
    rows3 = []
    for k in rows_keys:
        x = s[k]
        rank = order.index(k) + 1
        if k == base:
            verdict_cell = "比較的基準"
        else:
            a, b_, d = tests[k]
            verdict_cell = ("、".join([("避開更多" if a else "避開沒更多"), ("分數不輸" if b_ else "分數輸"),
                                     ("虧損不更深" if d else "虧損更深")]) + ("→ 算改進" if all(tests[k]) else "→ 不算"))
        rows3.append([label[k]] + [_share(v) for v in x["checks"]]
                     + [f"{_share(x['low'])}（第 {rank}／{len(KEYS)}）",
                        f"{_pct(x['c_deep_med'])}｜{_share(x['loss_ok'])}", verdict_cell])

    # table 4: other coins
    assets = list(r["other"])
    rows4 = []
    for k in rows_keys:
        x = s[k]
        rows4.append([label[k]] + [f"{x['o_each'][a][0]} 個裡 {x['o_each'][a][1]} 個" for a in assets]
                     + [f"{x['o_n']} 個裡 {x['o_beat']} 個（{_share(x['o_beat'] / x['o_n'] if x['o_n'] else None)}）"])

    tables = [
        {"caption": (f"比特幣 {BEAR_FROM[:4]} 年以來每一次跌 50% 以上的大熊市（UTC 0 點日收盤；收盤比之前的高點低一半算開始，從低點漲一倍算結束），"
                     "各規則實際第一次下車和上車的日期、收盤價。下車＝高點後第一次賣出（高點時已經是現金的，寫高點前最後一次賣出）；"
                     "上車＝低點時手上那一筆的買進，或低點後第一次買回。中間被洗（賣了又買回）的部分看第二表的等效價。"
                     f"最後兩欄是「{_name(show[2])}」（{'唯一算改進的做法' if winners else '最接近改進的做法，看完結果才挑出來列的'}）。"),
         "columns": ["大熊市（收盤高點 → 低點）", "跌幅", "50 日均線：下車", "50 日均線：上車", "55／20：下車", "55／20：上車",
                     f"{_short(show[2])}：下車", f"{_short(show[2])}：上車"],
         "rows": rows1},
        {"caption": (f"已經走完的 {len(done)} 次大熊市（不含還在進行的）。等效下車價＝照規則從高點到低點實際剩下多少錢，換算成「等於在哪個價位一次下車」；"
                     "等效上車價同理（從低點到下一個高點）。所以中間被洗的損失都算進去，沒有扣手續費。避開的跌幅＝（等效下車價−低點）÷（高點−低點）；"
                     "抓到的回升＝（下一個高點−等效上車價）÷（下一個高點−低點）。被洗＝下車後 30 天內又用更高的收盤價買回，"
                     f"每年次數和每年多付（買回價比賣出價高多少，加來回手續費 {2 * COST * 100:.1f}%）從 {BEAR_FROM[:4]} 年算到 {_d(close.index[-1])}。"),
         "columns": ["規則", "等效下車價比高點（中位數）", "等效上車價比低點（中位數）", "避開的跌幅：平均｜中位數",
                     "抓到的回升：平均｜中位數", "被洗：每年次數｜每年多付", "一年下車幾次"],
         "rows": rows2},
        {"caption": (f"main_signal_check 的公平檢驗，原封不動套在每條規則上：比特幣 {n_d} 個起算月（UTC 0 點）、{n_c:,} 格起算月×收盤時間（照課本做）、"
                     f"同樣 {n_c:,} 格照一般人方式做（10 小時後才做、成本 {P_COST * 100:.2f}%、一成進場漏掉，每個收盤時間 {SIMS} 次模擬）、"
                     "其他幣和 MSTR。前四個數字是報酬贏過同一天起一直抱著的比例；排名分數＝四項裡最低的那一項（事先定好的規則），"
                     f"最深虧損比抱著淺的比例要至少 {LOSS_BAR * 100:.0f}% 才排名。最後一欄是對 50 日均線的三項條件：中位數避開更多跌幅、排名分數不輸、"
                     f"{n_c:,} 格最深虧損中位數不更深，三項都要過才算改進。"),
         "columns": ["規則", f"比特幣 {n_d} 個起算月", f"{n_c:,} 格（照課本）", "照一般人方式做", "其他幣和 MSTR",
                     "排名分數（名次）", "最深虧損中位數｜比抱著淺的比例", "對 50 日均線"],
         "rows": rows3},
        {"caption": ("換資產：規則原封不動用在其他幣（幣安日線）和 MSTR（2020-08-11 起），每個月 1 日起算到三年前，照課本做、成本 0.10%。"
                     "每格是「起算日個數裡報酬贏過一直抱著的個數」。"),
         "columns": ["規則"] + assets + ["合計"],
         "rows": rows4},
    ]

    # today
    last = float(c[-1])
    now = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）", "value": f"比特幣 {last:,.0f} 美元"}]
    for k in rows_keys:
        cur = float(np.nan_to_num(sum(w * leg for w, leg in r["legs"][k])[-1], nan=0.0))
        level, new = _flip(c, k, cur)
        text = _word(cur)
        if level is not None:
            text += f"；收盤{'≤' if level < last else '≥'} 約 {level:,.0f} 美元（離最後收盤 {(level / last - 1) * 100:+.1f}%）會變成{_word(new)}"
        else:
            text += "；收盤在 −50% 到 +60% 之間都不會改變"
        now.append({"label": _name(k), "value": text})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態、收盤到哪裡會翻轉，不是買賣建議"})

    # verdict
    real = []
    for i in done:
        ex, en = _first_moves(c, r["legs"][base][0][1], bears[i])
        if ex is not None and en is not None:
            real.append((c[ex] / c[bears[i]["top"]] - 1, c[en] / c[bears[i]["low"]] - 1))
    out_avg = float(np.mean([a for a, _ in real])) * 100 if real else None
    in_avg = float(np.mean([b for _, b in real])) * 100 if real else None
    first = (f"{BEAR_FROM[:4]} 年以來走完的 {len(done)} 次大熊市，50 日均線第一次下車平均已比高點低 {abs(out_avg):.0f}%、"
             f"買回平均比低點高 {in_avg:.0f}%，算進中途被洗後只避開中位數 {_share(avoid[base])} 的跌幅。")
    best_avoid = max(CANDS, key=lambda k: avoid[k] or -9)
    second = (f"更早下車的做法確實避開更多（最多是「{_short(best_avoid)}」{_share(avoid[best_avoid])}），"
              f"但上車更晚（等效買回價比低點高 {_pct(stat(best_avoid, 'entry_eq') * 100)}，50 日均線 {_pct(stat(base, 'entry_eq') * 100)}），"
              f"公平檢驗排名分數只剩 {_share(s[best_avoid]['low'])}（50 日均線 {_share(s[base]['low'])}）；"
              f"均線出得越快被洗越多（20 日均線每年 {whip['a20_20']['whips']:.1f} 次，50 日 {whip[base]['whips']:.1f} 次）。")
    wants_now = _wants(c)
    def differ(k: str) -> float:
        ok = wants_now[base].notna() & wants_now[k].notna()
        return float((wants_now[k][ok] != wants_now[base][ok]).mean())

    if winners:
        w0 = winners[0]
        if differ(w0) < 0.05:
            third = (f"照事先定的三項條件，只有「{_short(w0)}」勉強算改進，但它和 50 日均線只有 {differ(w0) * 100:.0f}% 的日子不同，"
                     f"避開 {avoid[w0] * 100:.1f}% 對 {avoid[base] * 100:.1f}%、排名分數 {s[w0]['low'] * 100:.1f}% 對 {s[base]['low'] * 100:.1f}%，"
                     "差距小到是雜訊，沒有找到真正更早下車又不吃虧的做法。")
        else:
            third = (f"照四項公平檢驗，{len(winners)} 種做法三項條件都過，例如「{_short(w0)}」：避開 {_share(avoid[w0])}、"
                     f"排名分數 {_share(s[w0]['low'])}、最深虧損中位數 {_pct(s[w0]['c_deep_med'])}（50 日均線 {_pct(s[base]['c_deep_med'])}）。")
    else:
        third = (f"照四項公平檢驗，{len(CANDS)} 種做法沒有一種三項都勝過 50 日均線；最接近的是「{_name(closest)}」"
                 f"（過 {sum(tests[closest])} 項，排名分數 {_share(s[closest]['low'])}、最深虧損中位數 {_pct(s[closest]['c_deep_med'])}）。")
    verdict = first + second + third

    grid_note = "快出慢進 9 格完整結果（避開跌幅中位數｜每年被洗｜排名分數｜最深虧損中位數）：" + "；".join(
        f"{k[1:].replace('_', '／')} {_share(avoid[k])}｜{whip[k]['whips']:.1f} 次｜{_share(s[k]['low'])}｜{_pct(s[k]['c_deep_med'])}"
        for k in GRID) + "（斜線前是下車均線天數、後是上車）。"
    cuts, bad = r["look"]
    close_call = []
    for k in winners or [closest]:
        close_call.append(
            f"「{_name(k)}」和 50 日均線逐項比（四項報酬贏的比例）：{'／'.join(_share(v) for v in s[k]['checks'])} 對 "
            f"{'／'.join(_share(v) for v in s[base]['checks'])}；{n_c:,} 格每年多賺中位數 {_pp(s[k]['c_edge'])} 對 {_pp(s[base]['c_edge'])} 個百分點，"
            f"一般人做法 {_pp(s[k]['p_edge'])} 對 {_pp(s[base]['p_edge'])}；各次熊市避開的跌幅 "
            + "、".join(f"{crash[k][i]['avoid'] * 100:.0f}% 對 {crash[base][i]['avoid'] * 100:.0f}%" for i in done)
            + f"；和 50 日均線持有狀態不同的日子占 {differ(k) * 100:.1f}%。")
    notes = close_call + [
        grid_note,
        (f"還在進行的熊市（{', '.join(_d(close.index[bears[i]['top']]) + ' 起' for i in open_)}）沒有算進中位數，因為低點還沒確定；"
         "第一表照樣列出各規則在這一段什麼時候下車。") if open_ else "目前沒有還在進行的大熊市。",
        (f"大熊市只有 {len(done)} 次走完（2013 年 4 月有一次一週內跌掉七成，任何看日線的順勢規則都來不及），"
         "樣本非常少，中位數差幾個百分點不代表真的比較好。"),
        "候選做法是在已經知道 50 日均線在這幾次熊市怎麼下車、怎麼上車之後才寫下的，雖然參數在看結果前就定好、沒有再調，仍有事後設計的成分。",
        "相鄰的起算月共用大部分歷史，24 個收盤時間也只差幾個小時，所以「幾個裡幾個」不是互相獨立的證據。",
        (f"成本很粗略：照課本 {COST * 100:.2f}%、一般人做法 {P_COST * 100:.2f}%，沒有滑價、稅和現金利息；熊市等效價和被洗次數是用"
         f"UTC 0 點收盤、不扣成本算的。一般人做法每個收盤時間只模擬 {SIMS} 次（和 main_signal_check 一樣，三條對照規則用同一組亂數）。"),
        (f"不偷看未來的檢查：把 {cuts} 個切點之後的收盤價全部隨機打亂，再算一次所有規則，切點以前的持有狀態"
         + ("完全沒變。" if bad == 0 else f"有 {bad} 條規則變了，這次結果不可信。")),
        "過去的結果不保證未來會一樣。這裡只比較幾條固定規則過去在熊市怎麼進出、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "crash_timing",
        "title": "大熊市能不能更早下車、更早上車",
        "question": ("50 日均線的優勢來自避開大熊市的中段。有沒有規則能比它更早下車、更早買回，又不會被洗更多次——"
                     "換起算日、收盤時間、一般人做法和其他幣一起考，是否勝過 50 日均線？"),
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
