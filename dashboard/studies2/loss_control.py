"""Historical study "loss_control": for the same protection against bitcoin's big falls, which method gave up the
least return - holding less, a stop-loss with a re-entry rule, trend-following, volatility sizing, buying puts or a
collar?

Methods (bitcoin, daily closes, decided at the close of day t and held during day t+1, 0.10% of the amount traded,
no leverage, idle cash earns nothing unless stated):
1. Fixed 25 / 40 / 50 / 60 / 75% bitcoin, put back to the share at the close before the first day of each month
   (vol_sizing's fixed grid).
2. Stop-loss: out when the close is X% below the highest close since entry (X = 10..30); back in when (a) the close
   is above the exit close, (b) 30 days have passed, (c) the close is the highest of the last 20, (d) the close is
   above its 50-day average.
3. Volatility trailing stop: out when the close is 3 x (20-day average absolute daily change) below the highest close
   since entry; the same four re-entry rules.
4. Trend: trend_ensemble's 17-rule blend, the blend x 40% volatility sizing, close above the 50-day average, 55-day
   high in / 20-day low out (all reset to target every close, as trend_ensemble does).
5. Puts (MODEL prices, 2021-03 on): hold bitcoin and buy a put 10 / 20 / 30% below the price at every monthly roll, a
   three-month put 10% below rolled quarterly, and the 10% put bought only when the blend holds less than half.
   Black-Scholes with zero interest, volatility = Deribit DVOL shifted by the stored skew exactly as option_selling
   prices strikes away from the price; the buyer pays the model price plus 5% spread plus Deribit's stated fee and
   pays the delivery fee when the put finishes in the money (option_selling's constants).
6. Collar: the 10%-below put paid for by selling a call whose premium (after spread and fee) equals the put's cost.
7. Volatility sizing alone at a 40% target (vol_sizing's version: 30-day volatility, traded when 10% off target).

Equal protection: every method is also mixed with cash at a constant fraction k (k x the method's position, no
leverage) and, for target deepest losses, the largest k whose deepest loss is no worse than the target is used.

Sources: trend_ensemble's daily closes (blockchain.info before 2017-08, Binance after), Deribit DVOL (public, no
key), the stored skew history (read only), the stored Binance hourly bitcoin candles of seasonality.py (read only,
missing hours fetched into memory by clock_robustness._hourly) and the 3-month Treasury bill rate the way
robust_trend gets it. This module keeps no store of its own and writes nothing. Descriptive and in-sample.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import option_selling as osl
from dashboard.studies2 import trend_ensemble as te
from dashboard.studies2 import vol_sizing as vs

ID = "loss_control"
COST = 0.001
YEAR = 365
SIM_START = "2014-01-01"
MAIN_START = "2018-01-01"
OPT_START = "2021-04-01"
FIXED = (0.25, 0.40, 0.50, 0.60, 0.75)
STOPS = (0.10, 0.15, 0.20, 0.25, 0.30)
ATR_DAYS, ATR_MULT = 20, 3.0
REENTRY = ("a", "b", "c", "d")
RE_NAME = {"a": "回到賣出價之上", "b": "等 30 天", "c": "創 20 日新高", "d": "站上 50 日均線"}
WAIT = 30
TARGETS = (-30.0, -40.0, -50.0, -60.0)
KS = np.round(np.arange(0.05, 1.0001, 0.01), 2)
FALL = 0.30
CLOCKS = (0, 6, 12, 18)
PUT_GAPS = (0.10, 0.20, 0.30)
COLLAR_PUT = 0.10

_pct, _num, _d, _ok = te._pct, te._num, te._d, te._ok

TREND = {"blend": "17 條順勢規則混合", "bv40": "混合 × 波動調整 40%", "ma50": "收盤高於 50 日均線",
         "c55": "55 日新高進／20 日新低出"}
OPTS = {"p10": "每月買低 10% 的賣權", "p20": "每月買低 20% 的賣權", "p30": "每月買低 30% 的賣權",
        "q10": "每季買低 10% 的三個月賣權", "g10": "混合看多不到一半才買低 10% 賣權", "collar": "領口（買低 10% 賣權、賣買權付錢）"}


def _skey(x: float, re: str) -> str:
    return f"s{int(round(x * 100))}{re}"


def _name(key: str) -> str:
    if key == "hold":
        return "一直抱著"
    if key.startswith("f"):
        return f"固定 {key[1:]}% 比特幣"
    if key.startswith("s"):
        return f"停損 {key[1:-1]}%＋{RE_NAME[key[-1]]}回補"
    if key.startswith("atr"):
        return f"波動停損（3 倍日均波動）＋{RE_NAME[key[-1]]}回補"
    if key == "v40":
        return "只依波動調整（目標 40%）"
    return TREND.get(key) or OPTS[key]


# ---------- data ----------

def _dvol_daily(index: pd.DatetimeIndex) -> tuple[np.ndarray, dict]:
    """DVOL (fraction) at the moment each daily close is taken (00:00 UTC of the next day); NaN when unknown."""
    dvol = osl._dvol()
    out = np.full(len(index), np.nan)
    for i, ts in enumerate(index):
        d = osl._num(ts.date())
        if d + 1 in dvol:
            out[i] = dvol[d + 1][0]
        elif d in dvol:
            out[i] = dvol[d][1]
    return out, dvol


def _slope_daily(index: pd.DatetimeIndex) -> tuple[np.ndarray, bool]:
    """Skew slope for option_selling's _vol_at, the last stored reading at or before each close (no later reading)."""
    skew = osl._skew()
    if len(skew) < 20:
        return np.zeros(len(index)), False
    xs = np.array(sorted(skew))
    ys = np.array([-skew[x][0] / 100 / 0.2 for x in xs])
    days = np.array([osl._num(ts.date()) + 1 for ts in index])
    pos = np.searchsorted(xs, days, side="right") - 1
    return np.where(pos >= 0, ys[np.clip(pos, 0, None)], 0.0), True


# ---------- the rules ----------

def _stop_state(close: np.ndarray, gap: float | None, re: str, atr: np.ndarray | None = None) -> tuple:
    """1/0 after every close (decided with data up to that close). Starts in bitcoin. Returns (state, peak, level,
    exit price) arrays; level is the stop level while in."""
    n = len(close)
    s = pd.Series(close)
    ma50 = s.rolling(50, min_periods=50).mean().values
    high20 = s.rolling(20, min_periods=20).max().values
    state, peak_a, level_a, exit_a = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    inside, peak, exit_px, exit_i = True, close[0], np.nan, 0
    for i in range(n):
        c = close[i]
        if inside:
            peak = max(peak, c)
            if gap is not None:
                level = peak * (1 - gap)
            else:
                level = peak - ATR_MULT * atr[i] if math.isfinite(atr[i]) else -np.inf
            if c <= level:
                inside, exit_px, exit_i = False, c, i
            else:
                level_a[i] = level
        else:
            if re == "a":
                back = c > exit_px
            elif re == "b":
                back = i - exit_i >= WAIT
            elif re == "c":
                back = math.isfinite(high20[i]) and c >= high20[i]
            else:
                back = math.isfinite(ma50[i]) and c > ma50[i]
            if back:
                inside, peak = True, c
                level = peak * (1 - gap) if gap is not None else (
                    peak - ATR_MULT * atr[i] if math.isfinite(atr[i]) else -np.inf)
                level_a[i] = level
        state[i] = 1.0 if inside else 0.0
        peak_a[i] = peak
        exit_a[i] = exit_px
    return state, peak_a, level_a, exit_a


def _atr(close: np.ndarray) -> np.ndarray:
    return pd.Series(close).diff().abs().rolling(ATR_DAYS, min_periods=ATR_DAYS).mean().values


def _wants(close: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Share each timing method wants after every close (before the one-day delay)."""
    target = te._targets(close)[0]
    signal = te._signals(close)[0]
    vol30 = close.pct_change().rolling(30, min_periods=30).std(ddof=1) * math.sqrt(YEAR)
    c = close.values
    atr = _atr(c)
    out = pd.DataFrame(index=close.index)
    extra = {}
    for x in STOPS:
        for re in REENTRY:
            st = _stop_state(c, x, re)
            out[_skey(x, re)] = st[0]
            extra[_skey(x, re)] = st
    for re in REENTRY:
        st = _stop_state(c, None, re, atr)
        out[f"atr{re}"] = st[0]
        extra[f"atr{re}"] = st
    out["blend"] = target["blend"]
    out["bv40"] = target["vol40"]
    out["ma50"] = signal["A50"]
    from dashboard.studies2 import clock_robustness as cr
    out["c55"] = cr._breakout(close.reset_index(drop=True), 55, 20)
    out["v40"] = vs._size(vol30, 0.40)
    return out, extra


# ---------- options ----------

def _rolls(index: pd.DatetimeIndex, months: tuple) -> list[int]:
    """Positions of the closes taken at 00:00 UTC of the last Friday of each listed month (the Thursday candle)."""
    first, last = index[0].date(), index[-1].date()
    out, y, m = [], first.year, first.month
    while date(y, m, 1) <= last + timedelta(days=100):
        if m in months:
            nxt = date(y + (m == 12), m % 12 + 1, 1)
            day = nxt - timedelta(days=1)
            while day.weekday() != 4:
                day -= timedelta(days=1)
            out.append((pd.Timestamp(day - timedelta(days=1)) - index[0]).days)
        y, m = y + (m == 12), m % 12 + 1
    return out


def _buy_cost(spot, strike, years, vol, slope, scale, call: bool) -> tuple[float, float]:
    """(model value, what the buyer pays) for one option."""
    v = osl.price(spot, strike, years, osl._vol_at(vol, slope, strike / spot, scale), call)
    return v, v * (1 + osl.SPREAD) + min(osl.FEE * spot, osl.FEE_CAP * v)


def _option_run(close: np.ndarray, vol: np.ndarray, slope: np.ndarray, rolls: list[int], mode: str, gap: float,
                gate: np.ndarray | None = None) -> dict:
    """Daily account value (1 at the first roll, NaN before) of bitcoin plus a put (mode "put") or a collar."""
    n = len(close)
    eq = np.full(n, np.nan)
    rolls = [r for r in rolls if r >= 0]
    rolls = [r for r in rolls if r < n and math.isfinite(vol[r])] + [r for r in rolls if r >= n][:1]
    if len(rolls) < 3:
        return {"eq": eq, "paid": [], "call": [], "start": None, "rolls": 0, "hit": 0}
    level, units_prev = 1.0, 0.0
    eq[rolls[0]] = 1.0
    paid, calls, hits = [], [], 0
    for a, b in zip(rolls[:-1], rolls[1:]):
        if a >= n - 1:
            break
        s0, days = close[a], b - a
        years, scale = days / YEAR, math.sqrt(30 / days)
        buy = mode == "collar" or gate is None or (math.isfinite(gate[a]) and gate[a] < 0.5)
        idx = np.arange(a + 1, min(b, n - 1) + 1)
        path = close[idx]
        left = (b - idx) / YEAR
        trade_cost = 0.0
        if buy and math.isfinite(vol[a]):
            kp = s0 * (1 - gap)
            _, put_cost = _buy_cost(s0, kp, years, vol[a], slope[a], scale, False)
            if mode == "collar":
                lo, hi = s0, 4 * s0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    cv = osl.price(s0, mid, years, osl._vol_at(vol[a], slope[a], mid / s0, scale), True)
                    got = cv * (1 - osl.SPREAD) - min(osl.FEE * s0, osl.FEE_CAP * cv)
                    lo, hi = (mid, hi) if got > put_cost else (lo, mid)
                kc = (lo + hi) / 2
                calls.append(kc / s0 - 1)
                units = level / s0
            else:
                kc = None
                units = level / (s0 + put_cost)
                paid.append((a, put_cost / s0))
            trade_cost = COST * abs(units - units_prev) * s0
            pm = osl.price(path, kp, left, osl._vol_at(vol[idx], slope[idx], kp / path, scale), False)
            cm = osl.price(path, kc, left, osl._vol_at(vol[idx], slope[idx], kc / path, scale), True) \
                if kc is not None else 0.0
            if b <= n - 1:  # the option expires inside the data: settle with the delivery fee
                pin, cin = max(kp - close[b], 0.0), max(close[b] - kc, 0.0) if kc is not None else 0.0
                hits += pin > 0
                pm = np.array(pm, dtype=float)
                cm = np.array(cm if kc is not None else np.zeros(len(idx)), dtype=float)
                pm[-1] = pin - min(osl.DELIVERY_FEE * close[b], osl.FEE_CAP * pin)
                cm[-1] = cin + min(osl.DELIVERY_FEE * close[b], osl.FEE_CAP * cin)
            value = units * (path + pm - cm) - trade_cost
        else:
            units = level / s0
            trade_cost = COST * abs(units - units_prev) * s0
            value = units * path - trade_cost
        eq[idx] = value
        level = float(eq[idx[-1]])
        units_prev = units  # bitcoin units held into the next roll (settlement is in dollars and re-invested)
        if b > n - 1:
            break
    return {"eq": eq, "paid": paid, "call": calls, "start": rolls[0], "rolls": len(paid) or len(calls), "hit": hits}


def _scale_roll(eq: np.ndarray, rolls: list[int], ks: np.ndarray) -> np.ndarray:
    """k x the option account + (1-k) cash, put back to k at every roll. Returns (days, len(ks)) equity."""
    n = len(eq)
    out = np.full((n, len(ks)), np.nan)
    first = next((r for r in rolls if 0 <= r < n and math.isfinite(eq[r])), None)
    if first is None:
        return out
    level = np.ones(len(ks))
    cuts = [r for r in rolls if first <= r < n] + [n - 1]
    out[first] = 1.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        rel = eq[a + 1:b + 1] / eq[a]
        out[a + 1:b + 1] = level * ((1 - ks) + ks * rel[:, None])
        level = out[b]
    return out


# ---------- measuring ----------

def _measure(net: np.ndarray, index: pd.DatetimeIndex, start: str) -> dict:
    keep = index >= pd.Timestamp(start)
    x = np.nan_to_num(np.asarray(net, dtype=float)[keep])
    idx = index[keep]
    s = pd.Series(x, index=idx)
    m = te._measure(s, pd.Series(0.0, index=idx), pd.Series(0.0, index=idx))
    eq = np.concatenate([[1.0], np.cumprod(1 + x)])
    m["worst30"] = float((eq[30:] / eq[:-30] - 1).min() * 100) if len(eq) > 30 else None
    m["worst_val"] = m["years"][m["worst"]]
    return m


def _frontier_pick(stats: dict, target: float) -> dict | None:
    """Largest k whose deepest loss is no worse than target; stats has arrays ret/deepest over KS."""
    ok = np.where(stats["deepest"] >= target)[0]
    if not len(ok):
        return None
    j = ok[-1]
    return {"k": float(KS[j]), "ret": float(stats["ret"][j]), "deepest": float(stats["deepest"][j])}


def _compute(close: pd.Series, vol: np.ndarray | None = None, slope: np.ndarray | None = None,
             rate: pd.Series | None = None) -> dict:
    """Everything that depends on the data; study() only formats it. Kept separate so look-ahead can be tested."""
    wants, extra = _wants(close)
    keep = close.index >= pd.Timestamp(SIM_START)
    index = close.index[keep]
    change = close.pct_change().values[keep]
    held_want = wants.shift(1)[keep]
    timing = list(wants.columns)
    band_of = {k: (vs.BAND if k == "v40" else 0.0) for k in timing}
    rf = None
    if rate is not None:
        rf = rate.reindex(rate.index.union(index)).ffill().reindex(index).fillna(0.0).values / 100 / YEAR

    # every timing method at every k (k = 1 is the method itself)
    cols, bands, labels = [], [], []
    for key in timing:
        base = np.nan_to_num(held_want[key].values, nan=0.0)
        for k in KS:
            cols.append(base * k)
            bands.append(band_of[key])
            labels.append((key, float(k)))
    target = np.column_stack(cols)
    bands = np.array(bands)
    runs = {"plain": vs._sim(target, change, band=bands)}
    if rf is not None:
        runs["rf"] = vs._sim(target, change, band=bands, rf=rf)
    fixed = {"plain": vs._fixed_grid(index, change)}
    if rf is not None:
        fixed["rf"] = vs._fixed_grid(index, change, rf=rf)
    col = {lab: i for i, lab in enumerate(labels)}

    def net_of(key: str, which: str = "plain") -> tuple[np.ndarray, np.ndarray]:
        if key == "hold":
            return np.nan_to_num(change), np.ones(len(change))
        if key.startswith("f"):
            j = int(np.argmin(np.abs(vs.SHARES - int(key[1:]) / 100)))
            return fixed[which][0][:, j], fixed[which][2][:, j]
        j = col[(key, 1.0)]
        return runs[which][0][:, j], runs[which][2][:, j]

    keys = ["hold"] + [f"f{int(x * 100)}" for x in FIXED] + timing
    nets = {k: net_of(k)[0] for k in keys}
    out = {"index": index, "close": close, "wants": wants, "extra": extra, "nets": nets, "keys": keys,
           "has_rate": rf is not None}
    periods = {"main": MAIN_START, "opt": OPT_START}
    out["m"] = {p: {k: _measure(nets[k], index, s) for k in keys} for p, s in periods.items()}
    if rf is not None:
        out["m_rf"] = {k: _measure(net_of(k, "rf")[0], index, MAIN_START) for k in keys}
        out["rate_mean"] = float(rf[index >= pd.Timestamp(MAIN_START)].mean() * YEAR * 100)

    # options (2021 on)
    opt = {}
    if vol is not None:
        c = close.values[keep]
        v, sl = vol[keep], slope[keep]
        monthly = _rolls(index, tuple(range(1, 13)))
        quarterly = _rolls(index, (3, 6, 9, 12))
        start_i = int(np.argmax(np.isfinite(v))) if np.isfinite(v).any() else len(v)
        monthly = [r for r in monthly if r >= start_i]
        quarterly = [r for r in quarterly if r >= start_i]
        gate = wants["blend"].values[keep]
        specs = {"p10": (monthly, "put", 0.10, None), "p20": (monthly, "put", 0.20, None),
                 "p30": (monthly, "put", 0.30, None), "q10": (quarterly, "put", 0.10, None),
                 "g10": (monthly, "put", 0.10, gate), "collar": (monthly, "collar", COLLAR_PUT, None)}
        for key, (rolls, mode, gap, g) in specs.items():
            run = _option_run(c, v, sl, rolls, mode, gap, g)
            eq = run["eq"]
            net = np.zeros(len(eq))
            ok = np.isfinite(eq)
            net[1:] = np.where(ok[1:] & ok[:-1], eq[1:] / np.where(ok[:-1], eq[:-1], 1.0) - 1, 0.0)
            run["net"] = net
            run["rolls_list"] = rolls
            opt[key] = run
            nets[key] = net
        opt_first = min((r["start"] for r in opt.values() if r["start"] is not None), default=None)
        out["opt_ok"] = opt_first is not None and index[opt_first] < pd.Timestamp(OPT_START)
        if out["opt_ok"]:
            out["m"]["opt"].update({k: _measure(opt[k]["net"], index, OPT_START) for k in opt})
            yrs = (index[-1] - pd.Timestamp(OPT_START)).days / YEAR
            for k, r in opt.items():
                cut = [p for a, p in r["paid"] if index[a] >= pd.Timestamp(OPT_START)]
                r["paid_year"] = float(sum(cut) * 100 / yrs) if cut else 0.0
    out["opt"] = opt

    # frontier: for each period and family the best k-scaled setting at each target
    def kstats(which: str, start: str) -> dict:
        rows = index >= pd.Timestamp(start)
        st = vs._stats(runs[which][0][rows])
        fs = vs._stats(fixed[which][0][rows])
        per = {}
        for key in timing:
            j = [col[(key, float(k))] for k in KS]
            per[key] = {"ret": st["ret"][j], "deepest": st["deepest"][j]}
        per["fixed"] = fs
        return per

    fam = {"fixed": ["fixed"], "stop": [k for k in timing if k.startswith(("s", "atr"))],
           "trend": list(TREND), "vol": ["v40"], "opt": list(opt) if out.get("opt_ok") else []}
    frontier = {}
    for which in runs:
        for pname, start in periods.items():
            per = kstats(which, start)
            if pname == "opt" and out.get("opt_ok") and which == "plain":
                rows = index >= pd.Timestamp(start)
                for key, r in opt.items():
                    eqk = _scale_roll(r["eq"], r["rolls_list"], KS)[rows]
                    eqk = eqk / eqk[0]
                    netk = eqk[1:] / eqk[:-1] - 1
                    per[key] = vs._stats(np.vstack([np.zeros(len(KS)), netk]))
            res = {}
            for t in TARGETS:
                res[t] = {}
                for f, members in fam.items():
                    if pname == "main" and f == "opt" or (which == "rf" and f == "opt"):
                        continue
                    best = None
                    for key in members:
                        if key == "fixed":
                            ok = np.where(per["fixed"]["deepest"] >= t)[0]
                            if len(ok):
                                j = ok[-1]
                                cand = {"key": f"fixed", "k": float(vs.SHARES[j]), "ret": float(per["fixed"]["ret"][j]),
                                        "deepest": float(per["fixed"]["deepest"][j])}
                            else:
                                cand = None
                        else:
                            pick = _frontier_pick(per[key], t)
                            cand = dict(pick, key=key) if pick else None
                        if cand and (best is None or cand["ret"] > best["ret"]):
                            best = cand
                    res[t][f] = best
            frontier[(which, pname)] = res
    out["frontier"] = frontier

    # falls of 30%+ and the biggest rallies (2014 on)
    out["swings"] = _swings(close[keep])
    return out


def _swings(c: pd.Series) -> dict:
    """Zigzag with a 30% reversal both ways: [(peak position, trough position, size)], rallies, and whether the last
    fall is still open."""
    v = c.values
    falls, rallies = [], []
    mode, hi, hi_i, lo, lo_i = "up", v[0], 0, v[0], 0
    last_trough = 0
    for i in range(1, len(v)):
        x = v[i]
        if mode == "up":
            if x > hi:
                hi, hi_i = x, i
            elif x <= hi * (1 - FALL):
                rallies.append((last_trough, hi_i, hi / v[last_trough] - 1))
                mode, lo, lo_i = "down", x, i
        else:
            if x < lo:
                lo, lo_i = x, i
            elif x >= lo * (1 + FALL):
                falls.append((hi_i, lo_i, lo / hi - 1, False))
                last_trough = lo_i
                mode, hi, hi_i = "up", x, i
    if mode == "down":
        falls.append((hi_i, lo_i, lo / hi - 1, True))
    return {"falls": falls, "rallies": [x for x in rallies if x[0] > 0]}  # the first leg does not start at a low


# ---------- clock robustness ----------

def _clocks() -> tuple[dict | None, str | None]:
    """Deepest loss and yearly return (2018 on) of the 20% stop with 20-day-high re-entry and of the blend when the
    daily close is taken at 00, 06, 12 and 18 UTC (clock_robustness's hourly grid and engine)."""
    try:
        from dashboard.studies2 import clock_robustness as cr
        rows, problem = cr._hourly("BTCUSDT")
        if len(rows) < cr.MIN_HOURS:
            return None, "1 小時線不夠"
        price, b0, _ = cr._grid(rows)
        pre = cr._prehistory(b0)
        m = cr._Market(price, b0, pre)
        out = {}
        for h in CLOCKS:
            closes = np.concatenate([m.pre, m.price[h::24]])
            n0 = len(m.pre)
            stop = _stop_state(closes, 0.20, "c")[0][n0:]
            sig = te._signals(pd.Series(closes))[0]
            blend = sig.mean(axis=1).where(sig.notna().all(axis=1)).fillna(0.0).values[n0:]
            out[h] = {"stop": m.run(h, stop), "blend": m.run(h, blend)}
        return out, problem
    except Exception as exc:  # noqa: BLE001 - reported as a note
        return None, type(exc).__name__


# ---------- the study ----------

def study() -> dict:
    close = te._daily_close()
    problems = []
    try:
        vol, dvol = _dvol_daily(close.index)
        slope, has_skew = _slope_daily(close.index)
    except Exception as exc:  # noqa: BLE001 - the option rows are left out
        vol, dvol, slope, has_skew = None, None, None, False
        problems.append(f"這次拿不到 Deribit DVOL（{type(exc).__name__}），賣權和領口的部分沒有算。")
    rate, _, rate_problem = vs._tbill()
    if rate_problem:
        problems.append(rate_problem)
    r = _compute(close, vol, slope, rate)
    clocks, clock_problem = _clocks()
    index, m, last = r["index"], r["m"], r["index"][-1]
    main, optm = m["main"], m["opt"]
    hold = main["hold"]
    opt_ok = r.get("opt_ok", False)

    def under(x: dict) -> str:
        return f"{x['under']:,} 天" + ("（還沒回到）" if x["under_open"] else "")

    def given(x: dict, base: dict) -> str:
        return "—" if x is base else f"{base['ret'] - x['ret']:+.1f} 點"

    # ----- table 2: all methods 2018 on -----
    order2 = ["hold", "f25", "f40", "f50", "f60", "f75", "s20a", "s20b", "s20c", "s20d", "atrc", "blend", "bv40",
              "ma50", "c55", "v40"]
    rows2 = []
    for k in order2:
        x = main[k]
        rows2.append([_name(k), _pct(x["ret"]), _pct(x["deepest"]), f"{x['worst']} {_pct(x['worst_val'])}",
                      _pct(x["worst30"]), under(x), _num(x["ratio"]), given(x, hold)])

    # stop grid as compact text
    grid_lines = []
    for x in STOPS:
        cells = [f"{RE_NAME[re]} {_pct(main[_skey(x, re)]['ret'])}／{_pct(main[_skey(x, re)]['deepest'])}"
                 for re in REENTRY]
        grid_lines.append(f"停損 {int(x * 100)}%：" + "、".join(cells))
    grid_lines.append("波動停損（3 倍）：" + "、".join(
        f"{RE_NAME[re]} {_pct(main[f'atr{re}']['ret'])}／{_pct(main[f'atr{re}']['deepest'])}" for re in REENTRY))
    stop_keys = [k for k in r["keys"] if k.startswith(("s", "atr"))]
    best_stop = max(stop_keys, key=lambda k: main[k]["ret"])
    stop_beat_blend = sum(1 for k in stop_keys if main[k]["ret"] > main["blend"]["ret"])
    stop_beat_hold = sum(1 for k in stop_keys if main[k]["ret"] > hold["ret"])

    # ----- table 1: frontier -----
    fam_names = {"fixed": "少拿一點", "stop": "停損＋回補", "trend": "順勢", "vol": "波動調整", "opt": "賣權／領口"}

    def cell(b: dict | None) -> str:
        if not b:
            return "做不到"
        if b["key"] == "fixed":
            return f"固定 {b['k'] * 100:.0f}%：{_pct(b['ret'])}"
        short = _name(b["key"]).replace("回補", "").replace("（3 倍日均波動）", "")
        share = "" if b["k"] >= 0.995 else f"×{b['k'] * 100:.0f}%"
        return f"{short}{share}：{_pct(b['ret'])}"

    rows1 = []
    for pname, label in (("main", f"{MAIN_START[:4]} 到最新"), ("opt", f"{OPT_START[:7]} 到最新")):
        res = r["frontier"][("plain", pname)]
        for t in TARGETS:
            row = res[t]
            fams = [f for f in ("fixed", "stop", "trend", "vol", "opt")]
            cells = [cell(row.get(f)) if (f != "opt" or pname == "opt") else "—（2021 年前沒有資料）" for f in fams]
            avail = {f: b for f, b in row.items() if b}
            win = max(avail, key=lambda f: avail[f]["ret"]) if avail else None
            rows1.append([label, f"{t:.0f}%"] + cells + [fam_names[win] if win else "—"])

    # ----- table 3: falls one by one -----
    sw = r["swings"]
    falls = sw["falls"]
    eqs = {k: np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(r["nets"][k]))]) for k in
           ("hold", "f50", "s20c", "blend", "bv40")}
    if opt_ok:
        e = r["opt"]["p10"]["eq"]
        eqs["p10"] = np.concatenate([[np.nan], e])
    fall_keys = ("f50", "s20c", "blend", "bv40", "p10")

    def leg(k: str, a: int, b: int) -> float | None:
        if k not in eqs:
            return None
        e = eqs[k]
        x, y = e[a + 1], e[b + 1]
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return (y / x - 1) * 100

    shown = sorted(falls, key=lambda f: f[2])[:13] if len(falls) > 13 else falls
    shown = sorted(shown, key=lambda f: f[0])
    rows3 = []
    less = {k: 0 for k in fall_keys}
    count_p = 0
    for a, b, size, open_ in falls:
        for k in fall_keys:
            v = leg(k, a, b)
            if v is not None and v > size * 50:  # lost less than half of bitcoin's fall
                less[k] += 1
        if leg("p10", a, b) is not None:
            count_p += 1
    for a, b, size, open_ in shown:
        rows3.append([f"{_d(index[a])} → {_d(index[b])}" + ("（還在跌）" if open_ else ""), _pct(size * 100)]
                     + [_pct(leg(k, a, b)) for k in fall_keys])
    rallies = sorted(sw["rallies"], key=lambda x: -x[2])[:3]
    rally_kept = {k: [] for k in fall_keys}
    for a, b, g in sorted(rallies, key=lambda x: x[0]):
        cells = []
        for k in fall_keys:
            v = leg(k, a, b)
            if v is not None:
                rally_kept[k].append(v / (g * 100))
            cells.append("—" if v is None else f"{_pct(v)}（{v / (g * 100) * 100:.0f}%）")
        rows3.append([f"大漲 {_d(index[a])} → {_d(index[b])}", _pct(g * 100)] + cells)

    # ----- table 4: options era -----
    order4 = ["hold", "f50", "s20c", "blend", "bv40", "v40"] + (list(OPTS) if opt_ok else [])
    ohold = optm["hold"]
    rows4 = []
    for k in order4:
        x = optm[k]
        extra_cell = "—"
        if k in OPTS:
            o = r["opt"][k]
            if k == "collar":
                extra_cell = f"賣出的買權平均高 {np.mean(o['call']) * 100:.0f}%" if o["call"] else "—"
            else:
                extra_cell = f"每年 {o['paid_year']:.1f}%"
        rows4.append([_name(k), _pct(x["ret"]), _pct(x["deepest"]), f"{x['worst']} {_pct(x['worst_val'])}",
                      _pct(x["worst30"]), _num(x["ratio"]), given(x, ohold), extra_cell])

    span_main = f"{MAIN_START[:7]} 到 {_d(last)}"
    span_opt = f"{OPT_START[:7]} 到 {_d(last)}"
    tables = [
        {"caption": ("同樣的保護，各種方法能留下多少每年報酬（要把最深虧損控制在 X 左右，各種方法各能留下多少每年報酬）。"
                     "每種方法都可以只拿一部分錢去做、其餘放現金（例如「×60%」＝ 60% 的錢照這個方法做），"
                     "格子裡是每一類方法裡、最深虧損沒有超過目標的設定中每年報酬最高的那一個；"
                     "「做不到」＝這類方法任何設定都壓不到這麼淺（不借錢）。設定是看完整段歷史才挑出來的，數字偏樂觀；"
                     "停損這一類有 24 種組合可挑、順勢只有 4 種、選擇權 6 種，可挑的越多越容易挑到運氣好的那一個。"),
         "columns": ["期間", "目標最深虧損", "少拿一點（固定比例）", "停損＋回補", "順勢", "波動調整", "賣權保險／領口",
                     "留下最多的一類"],
         "rows": rows1},
        {"caption": (f"所有方法並排（{span_main}）。當天收盤決定、隔天照做，每次買賣扣 {COST * 100:.2f}%，現金沒有利息。"
                     "停損＝跌到離進場後最高收盤某個幅度就全部賣；回補＝賣掉之後什麼條件買回。其他停損幅度和回補方式的組合見說明。"
                     "「比一直抱著少賺」是每年報酬少了幾個百分點（負的＝反而多賺）。"),
         "columns": ["做法", "每年報酬", "最深虧損", "最差的日曆年", "最差的 30 天", "最久沒回到高點", "報酬÷波動",
                     "比一直抱著少賺"],
         "rows": rows2},
        {"caption": (f"比特幣 2014 年以來每一次跌 30% 以上（收盤價，從高點到低點；之後要反彈 30% 才算這次跌完），"
                     f"共 {len(falls)} 次" + (f"，這裡列最大的 {len(shown)} 次" if len(shown) < len(falls) else "")
                     + "，以及同一段日子各方法的帳面變化；最下面是漲最多的三段，括號是留下了比特幣漲幅的幾成。"
                     "賣權那欄只有 2021-03 以後有資料（模型估價）。"),
         "columns": ["期間（高點 → 低點）", "比特幣", "固定 50%", "停損 20%＋創 20 日新高回補", "17 條混合", "混合×波動 40%",
                     "每月買低 10% 賣權"],
         "rows": rows3},
    ]
    if opt_ok:
        tables.append(
            {"caption": (f"有選擇權資料的這段（{span_opt}）全部方法並排。賣權保險＝付一筆錢，買「跌破某價位可以照原價賣出」的權利；"
                         "領口＝買保險的錢用賣出上漲空間（賣一個買權）來付。選擇權價格都是用 Deribit 波動率指數算出的模型估價，"
                         "不是實際成交價。最後一欄是每年花在賣權上的錢（占當時比特幣價值），領口則是賣出的買權平均比現價高多少。"),
             "columns": ["做法", "每年報酬", "最深虧損", "最差的日曆年", "最差的 30 天", "報酬÷波動", "比一直抱著少賺",
                         "每年保險費／買權位置"],
             "rows": rows4})

    # ----- today -----
    w = r["wants"].iloc[-1]
    st = r["extra"]["s20c"]
    c_last = float(close.iloc[-1])
    if st[0][-1] == 1:
        stop_now = f"持有中；進場後最高收盤 {st[1][-1]:,.0f} 美元，收盤跌到 {st[2][-1]:,.0f} 美元以下就賣"
    else:
        high20 = float(close.iloc[-20:].max())
        stop_now = f"空手中；上次賣在 {st[3][-1]:,.0f} 美元，收盤創 20 日新高（目前要到 {high20:,.0f} 美元以上）才買回"
    sa = r["extra"]["atrc"]
    atr_now = (f"持有中，收盤跌到 {sa[2][-1]:,.0f} 美元以下就賣" if sa[0][-1] == 1 else
               f"空手中，上次賣在 {sa[3][-1]:,.0f} 美元")
    now = [
        {"label": f"最後一根收完的日線（{_d(last)}）", "value": f"收盤 {c_last:,.0f} 美元"},
        {"label": "停損 20%＋創 20 日新高回補", "value": stop_now},
        {"label": "波動停損（3 倍日均波動）＋創 20 日新高回補", "value": atr_now},
        {"label": "17 條順勢規則混合：照規則下一天持有", "value": f"{w['blend'] * 100:.0f}%"},
        {"label": "混合 × 波動調整 40%：照規則下一天持有", "value": f"{w['bv40'] * 100:.0f}%"},
        {"label": "收盤高於 50 日均線／55 日新高進 20 日新低出",
         "value": f"{'持有' if w['ma50'] == 1 else '空手'}／{'持有' if w['c55'] == 1 else '空手'}"},
        {"label": "只依波動調整（目標 40%）", "value": f"{w['v40'] * 100:.0f}%"},
    ]
    put_now = None
    if dvol:
        dl = max(dvol)
        sigma = dvol[dl][1]
        sl_now = float(slope[-1]) if slope is not None else 0.0
        val, pay = _buy_cost(c_last, c_last * 0.9, 30 / YEAR, sigma, sl_now, 1.0, False)
        put_now = pay / c_last * 100
        now.append({"label": f"一個月、約定價低 10% 的賣權，模型估買進成本（DVOL {sigma * 100:.0f}%，{osl._ym(dl)}）",
                    "value": f"現價的 {put_now:.2f}%（約 {pay:,.0f} 美元；一年 12 次約 {put_now * 12:.0f}%）"})

    # ----- verdict -----
    fr = r["frontier"][("plain", "main")][-40.0]
    avail = {f: b for f, b in fr.items() if b}
    win = max(avail, key=lambda f: avail[f]["ret"])
    others = "、".join(f"{fam_names[f]} {_pct(avail[f]['ret'])}" for f in ("fixed", "stop", "vol") if f in avail and f != win)
    b, s = main["blend"], main["s20c"]
    v1 = (f"{MAIN_START[:4]} 年以來一直抱著每年 {_pct(hold['ret'])}、最深虧 {_pct(hold['deepest'])}；"
          f"要把最深虧損壓到 −40% 左右，{fam_names[win]}留下最多（{cell(avail[win])}），{others}。")
    deeper = "，比一直抱著還深" if s["deepest"] < hold["deepest"] else ""
    v2 = (f"停損加回補 {len(stop_keys)} 種組合裡只有 {stop_beat_blend} 種報酬高過 17 條混合（{_pct(b['ret'])}、最深 {_pct(b['deepest'])}），"
          f"停損 20%＋創 20 日新高回補是 {_pct(s['ret'])}、最深 {_pct(s['deepest'])}{deeper}，結果全看回補方式。")
    if opt_ok:
        p, ph, co = optm["p10"], optm["hold"], optm["collar"]
        same = "和持有差不多" if abs(p["deepest"] - ph["deepest"]) < 5 else f"持有是 {_pct(ph['deepest'])}"
        v3 = (f"{OPT_START[:7]} 以來每月買低 10% 賣權（模型估價）一年約花 {r['opt']['p10']['paid_year']:.0f}% 保費，"
              f"每年 {_pct(p['ret'])}、最深 {_pct(p['deepest'])}，{same}；領口是 {_pct(co['ret'])}、{_pct(co['deepest'])}，"
              f"同期混合 {_pct(optm['blend']['ret'])}、{_pct(optm['blend']['deepest'])}。")
    else:
        v3 = "這次拿不到選擇權資料，沒有算賣權保險。"
    verdict = v1 + v2 + v3

    # ----- notes -----
    n_falls = len(falls)
    notes = [
        "名詞：停損＝跌到離進場後最高收盤某個幅度就全部賣掉；回補＝賣掉之後什麼條件再買回（回到賣出價之上、等滿 30 天、"
        "收盤創 20 日新高、收盤站上 50 日均線）；波動停損＝離最高收盤超過「3 × 最近 20 天每天平均漲跌金額」就賣；"
        "賣權保險＝付一筆錢，買「跌破某價位可以照原價賣出」的權利；領口＝買保險的錢用賣出上漲空間（賣一個買權）來付。",
        f"停損幅度 × 回補方式的全部組合（{span_main}，每年報酬／最深虧損）：" + "；".join(grid_lines)
        + f"。{len(stop_keys)} 種裡 {stop_beat_hold} 種每年報酬高過一直抱著、{stop_beat_blend} 種高過 17 條混合；"
          f"報酬最高的是{_name(best_stop)}（{_pct(main[best_stop]['ret'])}／{_pct(main[best_stop]['deepest'])}）。",
        "賣權和領口的價格都是模型估價，不是實際成交價：用 Deribit 的 DVOL（30 天期波動率指數）套 Black-Scholes（利率當 0），"
        "離現價遠的約定價照本站存下的偏斜歷史修正（只用當時已有的最近一筆），買進時比模型價多付 5% 價差、再加 Deribit 公布的手續費，"
        "到期有價值時另扣交割費——和「抱著比特幣賣買權」那項統計同一套假設。只有 2021-03 以後有這些資料，所以選擇權只能比 "
        f"{OPT_START[:7]} 以後這一段，這段只有一次大空頭（2022 年）。每月的換約日是每月最後一個週五 00:00（UTC）的收盤。",
        "停損的結果很看回補方式和每天用哪個時間的收盤價：同一個停損幅度，換一種回補方式，每年報酬可以差十幾個百分點（見上一條）。",
    ]
    if clocks:
        def clock_cells(k: str) -> str:
            return "、".join(f"{h:02d}:00 {_pct(clocks[h][k]['ret'])}／{_pct(clocks[h][k]['deepest'])}" for h in CLOCKS)
        notes.append(f"換收盤時間（UTC，{MAIN_START[:4]} 年以來，每年報酬／最深虧損）：停損 20%＋創 20 日新高回補 "
                     + clock_cells("stop") + "；17 條混合 " + clock_cells("blend") + "。")
    else:
        notes.append(f"這次沒有算換收盤時間的結果（{clock_problem}），表裡只有 UTC 0 點收盤；之前的研究發現單一規則換個收盤時間結果就可能差很多。")
    if r["has_rate"]:
        mr = r["m_rf"]
        fr_rf = r["frontier"][("rf", "main")]
        best_rf = []
        for t in TARGETS:
            av = {f: x for f, x in fr_rf[t].items() if x}
            if av:
                wf = max(av, key=lambda f: av[f]["ret"])
                best_rf.append(f"{t:.0f}% → {fam_names[wf]}（{cell(av[wf])}）")
        notes.append(f"現金有利息的版本（美國三個月國庫券，{MAIN_START[:4]} 年以來平均一年 {r['rate_mean']:.1f}%）："
                     + "、".join(f"{_name(k)} {_pct(mr[k]['ret'])}" for k in ("f50", "s20c", "blend", "bv40", "v40"))
                     + "；同樣保護下留下最多的一類：" + "；".join(best_rf) + "。實際放在交易所的美元不一定領得到這個利率。")
    notes += [
        f"大跌的次數很少：2014 年以來比特幣跌 30% 以上只有 {n_falls} 次（從高點算、反彈 30% 才算結束），"
        f"其中 {n_falls} 次裡固定 50% 有 {less['f50']} 次、停損 20%＋創 20 日新高回補 {less['s20c']} 次、17 條混合 {less['blend']} 次、"
        f"混合×波動 40% {less['bv40']} 次虧不到比特幣跌幅的一半"
        + (f"，每月買低 10% 賣權在有資料的 {count_p} 次裡 {less['p10']} 次" if opt_ok else "") + "。"
        "只測了比特幣一種資產，幾次大跌彼此也不獨立，方法之間的小差距不能當成確定的結論。",
        "各方法的設定（停損幅度、回補方式、目標比例）都是看過整段歷史才挑出來比較的，表裡的最佳數字偏樂觀，事先未必挑得到。",
        f"成本只用一個粗略的數字：每次買賣扣交易金額的 {COST * 100:.2f}%；選擇權另有上述價差與手續費假設。沒有算稅。"
        "一直抱著沒有扣成本。持有比例不是 0 或 100% 的做法每天收盤調回目標比例（只依波動調整那一列是差超過一成才調），"
        "固定比例每月 1 日調回。",
        "「最深虧損」是從帳上最高點往下掉最多的幅度；「最差的 30 天」是任何連續 30 天裡最大的跌幅；「最久沒回到高點」是從一個高點到"
        "重新回到那個高點最長的天數；「報酬÷波動」越高表示每承受一分起伏換到的報酬越多。今年還沒過完也算一年。",
        "價格來源：2017-08 以前是 blockchain.info 的每日價格，之後是幣安現貨日線（UTC 0 點收盤）；新高、停損都只看收盤價。",
        "過去的結果不保證未來會一樣。這裡只是整理各種方法過去做了什麼、現在各自是什麼狀態，不是買賣建議。",
    ]
    notes += problems
    if clock_problem and clocks:
        notes.append(f"1 小時線：{clock_problem}")

    sources = [
        {"name": "blockchain.info 每日比特幣價格（2017 年以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
        {"name": "幣安現貨 BTCUSDT 日線與 1 小時線", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        {"name": "Deribit DVOL 比特幣波動率指數（日線）",
         "url": "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=1D"},
        {"name": "Deribit 選擇權歷史成交（本站偏斜歷史的來源）", "url": "https://history.deribit.com/"},
        {"name": "Deribit 手續費說明", "url": "https://www.deribit.com/kb/fees"},
        {"name": "美國三個月國庫券利率（FRED DTB3）", "url": "https://fred.stlouisfed.org/series/DTB3"},
    ]
    return {
        "id": ID,
        "title": "怎樣躲大跌最划算：少拿、停損、順勢、買保險",
        "question": "同樣要躲掉比特幣的大跌，少拿一點、停損加回補、順勢、買賣權保險、領口，哪一種犧牲的報酬最少？",
        "asof": _d(last),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": sources,
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
