"""Historical study "rally_timing": after a big fall, can a rule get back into bitcoin closer to the low than the
50-day average does, without being fooled by dead-cat bounces?

Every rule below was written down before any result was looked at (no tuning). Decisions use closes up to and
including the decision close and are held from the next day; 0.10% of the amount traded.

Early entries (each is added to the 50-day-average rule: hold when the 50-day rule holds OR an early entry is open;
an early entry that has not yet been handed over to the 50-day rule is closed when a close falls below the 20-day
average; once the 50-day rule holds, its own exit applies):
  f1a  a close crosses above the 20-day average while some close of the last 90 days was 40%+ below the all-time high
  f1b  a close is the highest of the last 20 closes while some close of the last 90 days was 40%+ below its 1-year high
  f2   after such a 40% fall: a swing low is followed by a higher swing low (zigzag on closes with a fixed 10% reversal,
       built only from past closes) and a close crosses above the swing high between them
  f3   after such a 40% fall: 20-day realised volatility was in the lowest fifth of the past year at some day of the
       last 30 days, and a close is the highest of the last 20 closes
  f4   (bitcoin only) within 90 days after a capitulation marker - fear & greed <= 15, 7-day average funding <= -0.01%
       per 8 hours, a hash-ribbon recovery signal, or MVRV < 1 (all used one day late) - a close crosses above the
       20-day average
  f5   the 50-day average stops falling (5-day change >= 0) and the close is above it; held until a close below the
       50-day average (this one replaces the 50-day entry instead of being added to it)
  p*   half at the early rule, half at the 50-day rule.
Benchmarks: close above the 50-day average (ma50), 55-day high in / 20-day low out (b5520), at least 9 of the 17 rules
long (thresh), holding.

Measures: (1) for each bitcoin bear market of 50%+ since 2013 (peak -> low, ended when the price has doubled from the
low), the first entry on or after the low: price vs the low and share of the next rally caught = (next top - entry) /
(next top - low); (2) false starts = entries closed at a loss within 30 days; (3) main_signal_check's fair grid and its
pre-stated ranking rule, reusing its code (same start dates, clocks, ordinary-person simulations and seeds).
A rule counts as an improvement over ma50 only if (a) median share of rally caught is larger, (b) its ranking score is
not worse and (c) its median deepest loss over the start x clock cells is not deeper.

Sources (public, key-less, through the dashboard's helpers): blockchain.info and Binance daily/hourly closes, Yahoo for
MSTR, alternative.me fear & greed, Binance funding history, blockchain.info (CoinMetrics fallback) hash rate,
CoinMetrics community MVRV.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import funding_positioning as fp
from dashboard.studies2 import hash_ribbons as hr
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import popular_rules as pr
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_ensemble as te

COST = msc.COST
YEAR = 365
DEEP = 0.40  # "a big fall": 40% below the high
DEEP_DAYS = 90  # ... at some close of the last 90 days
SWING = 0.10  # zigzag reversal
VOL_FIFTH = 0.20
SQUEEZE_DAYS = 30
CAP_DAYS = 90
FG_LOW, FUND_LOW, MVRV_LOW = 15.0, -0.01, 1.0
SLOPE_DAYS = 5
FALSE_DAYS = 30
BEAR = 0.50  # bear market: a fall of 50% or more ...
BEAR_END = 1.00  # ... that ends when the price has doubled from the low
SINCE = "2013-01-01"
STORE = common.STORE / "rally_timing_markers.json"

BENCH = ("ma50", "b5520", "thresh")
FAST = ("f1a", "f1b", "f2", "f3", "f4", "f5")
PART = ("p1a", "p1b", "p2", "p3", "p4", "p5")
CANDS = BENCH + FAST + PART
MARKER = ("f4", "p4")  # need bitcoin-only data
HALF = dict(zip(PART, FAST))
SHORT = {"ma50": "50 日均線", "b5520": "55／20", "thresh": "17 條過半",
         "f1a": "1a 跌深站上 20 日均線", "f1b": "1b 跌深收 20 日新高", "f2": "2 更高的低點", "f3": "3 波動縮小後突破",
         "f4": "4 恐慌訊號＋站上 20 日均線", "f5": "5 50 日均線止跌",
         "p1a": "一半 1a＋一半 50 日", "p1b": "一半 1b＋一半 50 日", "p2": "一半 2＋一半 50 日",
         "p3": "一半 3＋一半 50 日", "p4": "一半 4＋一半 50 日", "p5": "一半 5＋一半 50 日"}
NAMES = {
    "ma50": "收盤高於 50 日均線才持有（主訊號）",
    "b5520": "55 日新高進、20 日新低出",
    "thresh": "17 條規則至少 9 條看多就全部持有",
    "f1a": "1a：比歷史高點低 40% 以上之後，收盤站上 20 日均線就先上車",
    "f1b": "1b：比一年高點低 40% 以上之後，收盤創 20 日新高就先上車",
    "f2": "2：大跌後出現更高的低點，再收過中間那個反彈高點就先上車",
    "f3": "3：大跌後波動縮到一年裡最低的兩成，再收 20 日新高就先上車",
    "f4": "4：恐慌訊號出現後 90 天內，收盤站上 20 日均線就先上車（只有比特幣）",
    "f5": "5：50 日均線不再往下（5 天變化 ≥ 0）而且收在均線上才上車",
}
for _p, _f in HALF.items():
    NAMES[_p] = f"一半照 {_f[1:]} 先上車、一半等 50 日均線"

_pct, _pp, _d, _share, _of = cr._pct, cr._pp, cr._d, msc._share, msc._of
_ok = msc._ok


# ---------- markers (bitcoin only) ----------

def _load_store() -> dict:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - first run or damaged file
        return {}


def _as_series(days: dict) -> pd.Series:
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in days.items()}).sort_index()
    return s[np.isfinite(s.values)]


def _fetch_fng() -> dict:
    s = pr._fear_greed()
    return {k.strftime("%Y-%m-%d"): float(v) for k, v in s.items()}


def _fetch_funding() -> dict:
    """7-day average of Binance BTCUSDT funding (% per 8 hours) at the end of each UTC day."""
    s = fp._funding_history()
    day = s.groupby(s.index.normalize()).mean().asfreq("D")
    avg7 = day.rolling(7, min_periods=5).mean().dropna()
    return {k.strftime("%Y-%m-%d"): float(v) for k, v in avg7.items()}


def _fetch_hash() -> dict:
    try:
        s = hr._hash_blockchain()
    except Exception:  # noqa: BLE001 - the same series from the other public source
        s = hr._hash_coinmetrics()
    return {k.strftime("%Y-%m-%d"): float(v) for k, v in s.items()}


def _fetch_mvrv() -> dict:
    out = pr._fetch_mvrv()
    if len(out) < 3000:
        raise RuntimeError(f"MVRV 只回傳 {len(out)} 筆")
    return out


def _markers(close: pd.Series) -> tuple[pd.DataFrame | None, list[str], dict]:
    """Daily capitulation markers, each already moved one day later (the value of day D is used from day D+1 on).
    A failed download falls back to this module's own stored copy (and for MVRV to popular_rules' stored copy, read
    only); a marker that cannot be had at all is left out and said so."""
    store, problems, raw, names = _load_store(), [], {}, {
        "fng": "恐懼與貪婪指數", "funding": "資金費率", "hash": "算力", "mvrv": "MVRV"}
    changed = False
    for key, fetch in (("fng", _fetch_fng), ("funding", _fetch_funding), ("hash", _fetch_hash), ("mvrv", _fetch_mvrv)):
        try:
            days = fetch()
            store[key] = dict(sorted(days.items()))
            changed = True
        except Exception as exc:  # noqa: BLE001
            days = store.get(key)
            if not days and key == "mvrv":
                try:
                    days = json.loads((common.STORE / "popular_rules_mvrv.json").read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    days = None
            if days:
                problems.append(f"{names[key]}這次抓不到（{type(exc).__name__}），改用之前存下來的資料（到 {max(days)}）。")
            else:
                problems.append(f"{names[key]}這次抓不到（{type(exc).__name__}），這個恐慌訊號這次沒有算進去。")
        if days:
            raw[key] = _as_series(days)
    if changed:
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            STORE.write_text(json.dumps(store), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    if not raw:
        return None, problems, {}
    idx = close.index
    cols = {}
    if "fng" in raw:
        cols["fng"] = (raw["fng"].reindex(idx) <= FG_LOW)
    if "funding" in raw:
        cols["funding"] = (raw["funding"].reindex(idx) <= FUND_LOW)
    if "mvrv" in raw:
        cols["mvrv"] = (raw["mvrv"].reindex(idx).ffill(limit=5) < MVRV_LOW)
    if "hash" in raw:
        ev = hr._signals(close, raw["hash"])["signal"]
        cols["hash"] = ev.reindex(idx).fillna(False).astype(bool)
    marks = pd.DataFrame(cols, index=idx).fillna(False).astype(bool).shift(1, fill_value=False)
    marks["cap"] = marks.any(axis=1)
    return marks, problems, raw


# ---------- rules ----------

def _zigzag(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each close: the older (L1) and newer (L2) of the last two confirmed swing lows and the swing high between
    them, of a zigzag on closes with a fixed 10% reversal. A swing is confirmed only when the price has already moved
    10% away from it, so every value uses past closes only."""
    n = len(c)
    l1, l2, hh = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    mode, hi, lo = 0, np.nan, np.nan
    last_l, last_h, prev_l, between = np.nan, np.nan, np.nan, np.nan
    for i in range(n):
        x = c[i]
        if not math.isfinite(x):
            continue
        if not math.isfinite(hi):
            hi = lo = x
        if mode >= 0:
            hi = max(hi, x)
        if mode <= 0:
            lo = min(lo, x)
        if mode >= 0 and x <= hi * (1 - SWING):
            last_h, mode, lo = hi, -1, x
        elif mode <= 0 and x >= lo * (1 + SWING):
            prev_l, between, last_l = last_l, last_h, lo
            mode, hi = 1, x
        l1[i], l2[i], hh[i] = prev_l, last_l, between
    return l1, l2, hh


def _latch(ready: np.ndarray, base: np.ndarray, trigger: np.ndarray, c: np.ndarray, m20: np.ndarray) -> np.ndarray:
    """1 when the 50-day rule holds; otherwise an early entry opened by `trigger` is held until a close below the
    20-day average (or until the 50-day rule takes over)."""
    out = np.full(len(c), np.nan)
    latch = False
    for i in range(len(c)):
        if not ready[i]:
            latch = False
            continue
        if base[i] > 0.5:
            out[i], latch = 1.0, False
        elif latch:
            if c[i] < m20[i]:
                out[i], latch = 0.0, False
            else:
                out[i] = 1.0
        elif trigger[i]:
            out[i], latch = 1.0, True
        else:
            out[i] = 0.0
    return out


def _turn(ready: np.ndarray, c: np.ndarray, m50: np.ndarray, slope: np.ndarray) -> np.ndarray:
    out = np.full(len(c), np.nan)
    hold = False
    for i in range(len(c)):
        if not ready[i]:
            hold = False
            continue
        if hold:
            hold = c[i] > m50[i]
        else:
            hold = c[i] > m50[i] and slope[i] >= 0
        out[i] = float(hold)
    return out


def _event(cond: pd.Series) -> np.ndarray:
    cond = cond.fillna(False).astype(bool)
    return (cond & ~cond.shift(1, fill_value=False)).values


def _rules(close: pd.Series, year: int = YEAR, cap: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What each candidate wants to hold after each close, and the pieces used (for today's state).
    cap: True on days with a capitulation marker (already lagged), on the same index; None = no marker data."""
    k = year / YEAR
    w90, w30 = max(int(round(DEEP_DAYS * k)), 1), max(int(round(SQUEEZE_DAYS * k)), 1)
    bench = msc._targets(close, year)
    c = close.values.astype(float)
    m20 = close.rolling(20, min_periods=20).mean()
    m50 = close.rolling(50, min_periods=50).mean()
    ready = m50.notna().values & m20.notna().values
    base = bench["ma50"].fillna(0.0).values
    ath = close.cummax()
    hi1 = close.rolling(year, min_periods=1).max()
    deep_ath = (close <= ath * (1 - DEEP)).astype(float).rolling(w90, min_periods=1).max() > 0
    deep_1y = (close <= hi1 * (1 - DEEP)).astype(float).rolling(w90, min_periods=1).max() > 0
    high20 = close >= close.rolling(20, min_periods=20).max()
    above20 = close > m20
    l1, l2, hh = _zigzag(c)
    higher = pd.Series((l2 > l1) & (c > hh), index=close.index)
    vol = close.pct_change().rolling(20, min_periods=20).std(ddof=1)
    fifth = vol.rolling(year, min_periods=120).quantile(VOL_FIFTH)
    quiet = (vol <= fifth) & fifth.notna()
    squeeze = quiet.astype(float).rolling(w30, min_periods=1).max() > 0
    slope = (m50 - m50.shift(SLOPE_DAYS)).values
    out = pd.DataFrame({"ma50": bench["ma50"], "b5520": bench["b5520"], "thresh": bench["thresh"]}, index=close.index)
    out["f1a"] = _latch(ready, base, _event(deep_ath & above20), c, m20.values)
    out["f1b"] = _latch(ready, base, _event(deep_1y & high20), c, m20.values)
    out["f2"] = _latch(ready, base, _event(deep_1y & higher), c, m20.values)
    out["f3"] = _latch(ready, base, _event(deep_1y & squeeze & high20), c, m20.values)
    if cap is not None:
        recent = cap.reindex(close.index).fillna(False).astype(float).rolling(CAP_DAYS, min_periods=1).max() > 0
        out["f4"] = _latch(ready, base, _event(recent & above20), c, m20.values)
    else:
        recent = pd.Series(False, index=close.index)
        out["f4"] = np.nan
    out["f5"] = _turn(ready, c, m50.values, slope)
    for p, f in HALF.items():
        out[p] = 0.5 * out[f] + 0.5 * out["ma50"]
    feat = pd.DataFrame({"m20": m20, "m50": m50, "ath": ath, "hi1": hi1, "deep_ath": deep_ath, "deep_1y": deep_1y,
                         "l1": l1, "l2": l2, "hh": hh, "vol": vol, "fifth": fifth, "squeeze": squeeze,
                         "slope": slope, "cap_recent": recent}, index=close.index)
    return out, feat


# ---------- fair grid (main_signal_check's code, more candidates) ----------

def _seed(k: str) -> int:
    return msc.CANDS.index(k) if k in msc.CANDS else 100 + CANDS.index(k)


def _daily(close: pd.Series, year: int, first, last, cap, cands) -> dict:
    target, _ = _rules(close, year, cap)
    change = close.pct_change().fillna(0.0).values
    valid = target[list(cands)].notna().all(axis=1)
    begin = max(close.index[int(np.argmax(valid.values))] + pd.Timedelta(days=1), pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels, "target": target}
    hold = msc._suffix(msc._equity(change), starts, year)
    out["hold"] = (hold[0][:, 0], hold[1][:, 0])
    for kk in cands:
        net, _ = te._run(target[kk].shift(1).values, change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, year)
        out[kk] = (r[:, 0], dd[:, 0])
    return out


def _clock(price: np.ndarray, b0: pd.Timestamp, pre: np.ndarray, cap: pd.Series | None) -> dict:
    m = cr._Market(price, b0, pre)
    eng = rt._Engine(m)
    days, npre = m.days, len(pre)
    starts, labels = msc._month_starts(m.dates, msc.CLOCK_STARTS[0], msc.CLOCK_STARTS[1])
    hold_r, hold_d = msc._suffix(msc._equity(m.change), starts, YEAR)
    hold_r, hold_d = hold_r[:, 0], hold_d[:, 0]
    cands = [k for k in CANDS if cap is not None or k not in MARKER]
    text = {k: (np.zeros((len(starts), 24)), np.zeros((len(starts), 24))) for k in cands}
    person = {k: (np.zeros((len(starts), 24, msc.SIMS)), np.zeros((len(starts), 24, msc.SIMS))) for k in cands}
    pre_dates = pd.date_range(end=b0 - pd.Timedelta(days=2), periods=npre, freq="D")
    for h in range(24):
        series = price[h::24]
        dates = pre_dates.append(pd.date_range(b0 - pd.Timedelta(days=1), periods=len(series), freq="D"))
        closes = pd.Series(np.concatenate([pre, series]), index=dates)
        # a close at clock h of day j uses the markers of the day before the last finished UTC day (as the daily run)
        target, _ = _rules(closes, YEAR, cap)
        target = target.fillna(0.0)
        for k in cands:
            w = target[k].values[npre:][:days].astype(float)
            r, dd = msc._suffix(np.concatenate([[1.0], eng._mid(h, w, COST, None)]), starts, YEAR)
            text[k][0][:, h], text[k][1][:, h] = r[:, 0], dd[:, 0]
            b, wl = rt._later(h, w, days)
            rng = np.random.default_rng(msc.SEED + 1000 * _seed(k) + h)
            pos = rt._missed(wl, msc.P_MISS, rng, msc.SIMS)
            mid = eng._mid(b, pos, msc.P_COST, None)
            r, dd = msc._suffix(np.vstack([np.ones((1, msc.SIMS)), mid]), starts, YEAR)
            person[k][0][:, h, :], person[k][1][:, h, :] = r, dd
    return {"starts": labels, "hold": (hold_r, hold_d), "text": text, "person": person, "cands": cands}


def _summary(daily: dict, clock: dict, other: dict) -> dict:
    """main_signal_check._summary for this module's candidates. A candidate without other-asset results (the
    capitulation ones) is scored on the three bitcoin checks only."""
    hr_, hd = clock["hold"]
    s = {}
    for k in clock["cands"]:
        dr, dd = daily[k]
        edge_d = dr - daily["hold"][0]
        tr, td = clock["text"][k]
        edge = tr - hr_[:, None]
        pr_, pd_ = clock["person"][k]
        pedge = pr_ - hr_[:, None, None]
        assets = [a for a in other if k in other[a]]
        n_other = sum(len(other[a]["starts"]) for a in assets)
        x = {"d_n": len(dr), "d_beat": int((edge_d > 0).sum()), "d_shallow": int((dd > daily["hold"][1]).sum()),
             "c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
             "c_p10": float(np.percentile(edge, 10)), "c_shallow": int((td > hd[:, None]).sum()),
             "c_deep_med": float(np.median(td)), "c_deep_worst": float(td.min()),
             "p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge)),
             "p_deep_med": float(np.median(pd_)),
             "o_n": n_other, "o_beat": sum(int((other[a][k][0] > other[a]["hold"][0]).sum()) for a in assets),
             "o_shallow": sum(int((other[a][k][1] > other[a]["hold"][1]).sum()) for a in assets),
             "o_each": {a: (len(other[a]["starts"]), int((other[a][k][0] > other[a]["hold"][0]).sum()),
                            int((other[a][k][1] > other[a]["hold"][1]).sum())) for a in assets}}
        x["checks"] = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"], x["p_beat"] / x["p_n"],
                       (x["o_beat"] / x["o_n"]) if x["o_n"] else None]
        x["low"] = min(v for v in x["checks"] if v is not None)
        x["loss_ok"] = min(x["d_shallow"] / x["d_n"], x["c_shallow"] / x["c_n"])
        x["eligible"] = x["loss_ok"] >= msc.LOSS_BAR
        s[k] = x
    order = sorted(s, key=lambda k: (not s[k]["eligible"], -s[k]["low"], -s[k]["c_edge"]))
    return {"s": s, "order": order, "hold_deep_med": float(np.median(hd))}


# ---------- rallies and false starts (bitcoin, dashboard daily closes) ----------

def _bears(close: pd.Series) -> list[dict]:
    """Bear markets of 50%+ since 2013: peak, low (lowest close before the price doubles from it), next top (highest
    close from the low to the next bear's peak, or to today)."""
    c = close[close.index >= pd.Timestamp(SINCE)]
    v, idx = c.values, c.index
    out, peak, low, falling = [], 0, 0, False
    for i in range(1, len(v)):
        if not falling:
            if v[i] > v[peak]:
                peak = i
            elif v[i] <= v[peak] * (1 - BEAR):
                falling, low = True, i
        else:
            if v[i] < v[low]:
                low = i
            elif v[i] >= v[low] * (1 + BEAR_END):
                out.append([peak, low])
                falling, peak = False, i
    ongoing = (peak, low) if falling else None
    bears = []
    for j, (p, lo) in enumerate(out):
        end = out[j + 1][0] if j + 1 < len(out) else (ongoing[0] if ongoing else len(v) - 1)
        top = lo + int(np.argmax(v[lo:end + 1]))
        bears.append({"peak": idx[p], "peak_px": float(v[p]), "low": idx[lo], "low_px": float(v[lo]),
                      "top": idx[top], "top_px": float(v[top])})
    return bears


def _entry(close: pd.Series, target: pd.Series, bear: dict) -> dict | None:
    """First entry on or after the low (the start of the holding run in effect on the first held day)."""
    t = target.fillna(0.0)
    after = t[(t.index >= bear["low"]) & (t.index <= bear["top"])]
    held = after[after > 0.001]
    if held.empty:
        return None
    day = held.index[0]
    span = bear["top_px"] - bear["low_px"]
    if day == bear["low"]:  # never left during the fall: it was already in at the low
        return {"day": day, "px": bear["low_px"], "above": 0.0, "caught": 100.0, "held": True}
    px = float(close.loc[day])
    return {"day": day, "px": px, "above": (px / bear["low_px"] - 1) * 100,
            "caught": (bear["top_px"] - px) / span * 100 if span > 0 else None}


def _false_starts(close: pd.Series, target: pd.Series) -> dict:
    """Entries (from cash to holding anything) closed back to cash within 30 days at a loss, from SINCE on."""
    change = close.pct_change().fillna(0.0).values
    pos = target.shift(1).fillna(0.0).values
    net, _ = te._run(pos, change, COST)
    t = target.fillna(0.0).values
    idx = close.index
    first = int(idx.searchsorted(pd.Timestamp(SINCE)))
    trades, bad = 0, []
    i = max(first, 1)
    while i < len(t):
        if t[i] > 0.001 and t[i - 1] <= 0.001:
            j = i
            while j + 1 < len(t) and t[j + 1] > 0.001:
                j += 1
            if j + 1 >= len(t):
                break  # still open
            trades += 1
            gain = float(np.prod(1 + net[i + 1:min(j + 3, len(net))])) - 1  # held days and the exit day's cost
            if j + 1 - i <= FALSE_DAYS and gain < 0:
                bad.append((idx[i], gain * 100))
            i = j + 1
        else:
            i += 1
    years = (len(t) - first) / YEAR
    worst = min(bad, key=lambda x: x[1]) if bad else None
    return {"trades": trades, "n": len(bad), "per_year": len(bad) / years, "trades_year": trades / years,
            "cost": float(sum(g for _, g in bad)), "worst": worst}


# ---------- the whole computation ----------

def _compute() -> dict:
    close = te._daily_close()
    marks, problems, raw = _markers(close)
    cap = marks["cap"] if marks is not None else None
    cands_btc = [k for k in CANDS if cap is not None or k not in MARKER]
    daily = _daily(close, YEAR, msc.DAILY_STARTS[0], msc.DAILY_STARTS[1], cap, cands_btc)
    rows, problem = cr._hourly("BTCUSDT")
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    if problem:
        problems.insert(0, problem)
    price, b0, _ = cr._grid(rows)
    pre = cr._prehistory(b0)
    if not len(pre):
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    clock = _clock(price, b0, pre, cap)
    other_data, more = msc._load_other()
    problems += more
    three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
    plain = [k for k in CANDS if k not in MARKER]
    other = {}
    for key, (c, since, year) in other_data.items():
        try:
            other[key] = _daily(c, year, since or c.index[0], three, None, plain)
            if not other[key]["starts"]:
                raise RuntimeError("沒有起算日")
        except Exception as exc:  # noqa: BLE001
            other.pop(key, None)
            problems.append(f"{key} 算不出來（{type(exc).__name__}），沒有算進去。")
    target, feat = _rules(close, YEAR, cap)
    bears = _bears(close)
    entries = {k: [(_entry(close, target[k], b) if k in cands_btc else None) for b in bears]
               for k in BENCH + FAST}
    falses = {k: _false_starts(close, target[k]) for k in cands_btc}
    return {"close": close, "marks": marks, "raw": raw, "problems": problems, "daily": daily, "clock": clock,
            "other": other, "target": target, "feat": feat, "bears": bears, "entries": entries, "falses": falses,
            "cands": cands_btc}


# ---------- today ----------

def _now(r: dict) -> list[dict]:
    close, t, f = r["close"], r["target"].iloc[-1], r["feat"].iloc[-1]
    c = close.values
    last = float(c[-1])

    def lvl(x: float) -> str:
        return f"{x:,.0f} 美元（離最後收盤 {(x / last - 1) * 100:+.1f}%）"

    def word(v) -> str:
        if not _ok(v):
            return "—"
        return "全部持有" if v >= 0.999 else "全部現金" if v <= 0.001 else f"持有 {v * 100:.0f}%"

    ma20_next = float(c[-19:].sum() / 19)
    ma50_next = float(c[-49:].sum() / 49)
    high20 = float(c[-19:].max())
    out = [{"label": f"最後一根收完的日線（{_d(close.index[-1])}，UTC 0 點收盤）",
            "value": (f"比特幣 {last:,.0f} 美元；比歷史最高收盤 {f['ath']:,.0f} 低 {(1 - last / f['ath']) * 100:.0f}%，"
                      f"比一年內最高收盤 {f['hi1']:,.0f} 低 {(1 - last / f['hi1']) * 100:.0f}%")}]
    out.append({"label": SHORT["ma50"], "value": f"{word(t['ma50'])}；收盤 ≥ {lvl(ma50_next)}會轉成持有"
                if t["ma50"] < 0.5 else f"{word(t['ma50'])}；收盤 ≤ {lvl(ma50_next)}會轉成現金"})
    b_long = t["b5520"] >= 0.5
    out.append({"label": SHORT["b5520"], "value": f"{word(t['b5520'])}；收盤"
                + (f" ≤ {lvl(float(c[-19:].min()))}會轉成現金" if b_long else f" ≥ {lvl(float(c[-54:].max()))}會轉成持有")})
    try:
        th = [x for x in msc._now(close, msc._targets(close)) if x["label"] == msc.SHORT["thresh"]]
        if th:
            out.append({"label": SHORT["thresh"], "value": th[0]["value"]})
    except Exception:  # noqa: BLE001 - the state line is optional
        pass
    deep_a, deep_y = bool(f["deep_ath"]), bool(f["deep_1y"])
    above20 = last > f["m20"]
    if t["f1a"] >= 0.5:
        v1a = f"{word(t['f1a'])}；收盤跌破 20 日均線（約 {lvl(ma20_next)}）而 50 日均線還沒接手時會轉成現金"
    elif not deep_a:
        v1a = "全部現金；最近 90 天沒有任何收盤比歷史高點低 40% 以上，這條沒有啟動"
    elif above20:
        v1a = "全部現金；已在 20 日均線上方但不是「剛站上」，要先收在 20 日均線下、再重新站上才會先上車"
    else:
        v1a = f"全部現金；已符合「比歷史高點低 40%」，收盤 ≥ {lvl(ma20_next)}（站上 20 日均線）會先上車"
    out.append({"label": SHORT["f1a"], "value": v1a})
    if t["f1b"] >= 0.5:
        v1b = f"{word(t['f1b'])}；收盤跌破 20 日均線（約 {lvl(ma20_next)}）而 50 日均線還沒接手時會轉成現金"
    elif not deep_y:
        v1b = "全部現金；最近 90 天沒有收盤比一年高點低 40% 以上，這條沒有啟動"
    else:
        v1b = f"全部現金；已符合「比一年高點低 40%」，收盤 ≥ {lvl(high20)}（20 日新高）會先上車"
    out.append({"label": SHORT["f1b"], "value": v1b})
    if t["f2"] >= 0.5:
        v2 = f"{word(t['f2'])}；收盤跌破 20 日均線（約 {lvl(ma20_next)}）而 50 日均線還沒接手時會轉成現金"
    elif not deep_y:
        v2 = "全部現金；最近 90 天沒有 40% 以上的大跌，這條沒有啟動"
    elif _ok(f["l1"]) and _ok(f["l2"]) and f["l2"] > f["l1"] and _ok(f["hh"]):
        v2 = (f"全部現金；最近兩個低點 {f['l1']:,.0f} → {f['l2']:,.0f} 已經墊高，收盤 ≥ {lvl(float(f['hh']))}"
              "（兩低點中間的反彈高點）會先上車")
    else:
        lo2 = f"{f['l2']:,.0f}" if _ok(f["l2"]) else "—"
        v2 = (f"全部現金；還沒有「更高的低點」（最近確認的低點 {lo2} 美元）。要先出現一個比它高的新低點"
              "（從那個低點反彈 10% 才算確認），再收過中間的反彈高點")
    out.append({"label": SHORT["f2"], "value": v2})
    if t["f3"] >= 0.5:
        v3 = f"{word(t['f3'])}；收盤跌破 20 日均線（約 {lvl(ma20_next)}）而 50 日均線還沒接手時會轉成現金"
    elif not deep_y:
        v3 = "全部現金；最近 90 天沒有 40% 以上的大跌，這條沒有啟動"
    else:
        pct = float((r["feat"]["vol"].iloc[-YEAR:] <= f["vol"]).mean() * 100) if _ok(f["vol"]) else None
        state = "最近 30 天內有縮到一年最低兩成" if bool(f["squeeze"]) else "最近 30 天沒有縮到一年最低兩成"
        v3 = (f"全部現金；20 日波動現在排在一年裡由低到高的 {pct:.0f}% 位置，{state}"
              + (f"，收盤 ≥ {lvl(high20)}（20 日新高）會先上車" if bool(f["squeeze"]) else "，要先縮到那裡、再收 20 日新高"))
    out.append({"label": SHORT["f3"], "value": v3})
    marks = r["marks"]
    if marks is not None and "f4" in r["cands"]:
        recent = marks.iloc[-CAP_DAYS:]
        names = {"fng": "恐懼指數 ≤ 15", "funding": "7 天平均資金費率 ≤ −0.01%", "hash": "礦工投降結束訊號",
                 "mvrv": "MVRV < 1"}
        fired = [f"{names[col]}（最近一次 {_d(recent.index[recent[col].values][-1] - pd.Timedelta(days=1))}）"
                 for col in names if col in recent and recent[col].any()]
        if t["f4"] >= 0.5:
            v4 = f"{word(t['f4'])}；收盤跌破 20 日均線（約 {lvl(ma20_next)}）而 50 日均線還沒接手時會轉成現金"
        elif not fired:
            v4 = "全部現金；最近 90 天沒有出現任何恐慌訊號，這條沒有啟動"
        elif above20:
            v4 = ("全部現金；最近 90 天有：" + "、".join(fired)
                  + "；已在 20 日均線上方但不是「剛站上」，要先收在 20 日均線下、再重新站上才會先上車")
        else:
            v4 = "全部現金；最近 90 天有：" + "、".join(fired) + f"；收盤 ≥ {lvl(ma20_next)}（站上 20 日均線）會先上車"
        out.append({"label": SHORT["f4"], "value": v4})
    slope = f["slope"]
    sl = f"50 日均線 5 天變化 {slope / f['m50'] * 100:+.1f}%" if _ok(slope) and _ok(f["m50"]) else "—"
    if t["f5"] >= 0.5:
        v5 = f"{word(t['f5'])}；{sl}；收盤 ≤ {lvl(ma50_next)}會轉成現金"
    else:
        v5 = f"全部現金；{sl}，要等它不再往下、而且收盤高於 50 日均線（約 {lvl(ma50_next)}）"
    out.append({"label": SHORT["f5"], "value": v5})
    out.append({"label": "這幾格的意思", "value": "只描述各規則在最後一根日線收盤後停在哪個狀態、要什麼條件才會改變，不是買賣建議"})
    return out


# ---------- the study ----------

def _median(values) -> float | None:
    v = [x for x in values if _ok(x)]
    return float(np.median(v)) if v else None


def study() -> dict:
    r = _compute()
    sm = _summary(r["daily"], r["clock"], r["other"])
    s, bears, entries, falses = sm["s"], r["bears"], r["entries"], r["falses"]
    close, cands = r["close"], r["cands"]
    n_d, n_c = len(r["daily"]["starts"]), len(r["clock"]["starts"]) * 24

    # rally shares (partials: half each)
    share, above = {}, {}
    for k in cands:
        if k in HALF:
            a, b = entries[HALF[k]], entries["ma50"]
            share[k] = [0.5 * (x["caught"] + y["caught"]) if x and y else None for x, y in zip(a, b)]
            above[k] = [0.5 * (x["above"] + y["above"]) if x and y else None for x, y in zip(a, b)]
        else:
            share[k] = [x["caught"] if x else None for x in entries[k]]
            above[k] = [x["above"] if x else None for x in entries[k]]
    med_share = {k: _median(share[k]) for k in cands}
    med_above = {k: _median(above[k]) for k in cands}
    gap = {k: _median([(b - a) if _ok(a) and _ok(b) else None for a, b in zip(above[k], above["ma50"])])
           for k in cands}  # how much closer to the low than the 50-day rule, percentage points, median over bears

    base = s["ma50"]
    rank = {k: i for i, k in enumerate(sm["order"], 1)}
    verdicts = {}
    for k in cands:
        if k not in FAST + PART:
            continue
        x = s[k]
        a = _ok(med_share[k]) and med_share[k] > med_share["ma50"] + 1e-9
        b = (x["eligible"] or not base["eligible"]) and x["low"] >= base["low"] - 1e-12
        c = x["c_deep_med"] >= base["c_deep_med"] - 1e-9
        verdicts[k] = (a, b, c)
    winners = [k for k in verdicts if all(verdicts[k])]
    closest = sorted(verdicts, key=lambda k: (-sum(verdicts[k]), -(s[k]["low"] - base["low"]),
                                              -((med_share[k] or 0) - (med_share["ma50"] or 0))))

    # table 1: rally by rally
    rows1 = [["低點 → 下一個高點"] + [f"{_d(b['low'])} {b['low_px']:,.0f} → {_d(b['top'])[:7]} {b['top_px']:,.0f}"
                                  for b in bears] + ["中位數"]]
    for k in BENCH + FAST:
        if k not in cands:
            continue
        cells = []
        for e in entries[k]:
            cells.append("—" if e is None else "低點時沒有賣出（一路持有）｜算吃到 100%" if e.get("held") else
                         f"{_d(e['day'])} 上車，比低點高 {e['above']:.0f}%｜吃到 {e['caught']:.0f}%")
        rows1.append([NAMES[k]] + cells + [f"比低點高 {med_above[k]:.0f}%｜吃到 {med_share[k]:.0f}%"
                                           if _ok(med_share[k]) else "—"])
    if len(rows1[0]) > 8:  # keep at most 8 columns: the oldest bears go to the notes
        cut = len(rows1[0]) - 8
        rows1 = [[row[0]] + row[1 + cut:] for row in rows1]

    # table 2: share caught vs false starts
    rows2 = []
    for k in cands:
        fs = falses[k]
        worst = f"{_d(fs['worst'][0])}：{fs['worst'][1]:+.1f}%" if fs["worst"] else "—"
        rows2.append([NAMES[k], f"{med_share[k]:.1f}%" if _ok(med_share[k]) else "—",
                      (f"+{med_above[k]:.0f}%｜{gap[k]:+.0f} 點" if _ok(gap[k]) else f"+{med_above[k]:.0f}%")
                      if _ok(med_above[k]) else "—",
                      f"{fs['trades_year']:.1f} 次", f"{fs['n']} 次（一年 {fs['per_year']:.1f} 次）",
                      f"{fs['cost']:+.0f}%", worst])

    # table 3: fair grid
    rows3 = []
    for k in sm["order"]:
        x = s[k]
        ck = x["checks"]
        v = verdicts.get(k)
        mark = "（基準）" if v is None else "".join("✓" if ok else "✗" for ok in v)
        rows3.append([NAMES[k], _share(ck[0]), _share(ck[1]), _share(ck[2]),
                      _share(ck[3]) if ck[3] is not None else "沒有資料",
                      f"{_share(x['low'])}（第 {rank[k]}）" + ("" if x["eligible"] else "，虧損條件未達"),
                      f"{_pct(x['c_deep_med'])}｜{_pp(x['c_edge'])}", mark])

    # table 4: other coins
    o = r["other"]
    assets = list(o)
    plain = [k for k in cands if k not in MARKER]
    rows4 = [["起算日範圍（幾個）"] + [f"{_d(o[a]['starts'][0])[:7]} 到 {_d(o[a]['starts'][-1])[:7]}（{len(o[a]['starts'])} 個）"
                                for a in assets] + [f"共 {sum(len(o[a]['starts']) for a in assets)} 個"]]
    for k in plain:
        x = s[k]
        if len(rows4) >= 16:
            break
        rows4.append([NAMES[k]] + [f"{x['o_each'][a][1]}｜{x['o_each'][a][2]}" if a in x["o_each"] else "—"
                                   for a in assets] + [f"{x['o_beat']}｜{x['o_shallow']}"])

    tables = [
        {"caption": ("比特幣每一次跌 50% 以上的空頭（從高點跌一半以上，價格從低點翻倍才算結束），各規則在低點之後第一次上車的日期、"
                     "上車價比低點高多少，以及「吃到」下一段上漲的幾成 =（下一個高點 − 上車價）÷（下一個高點 − 低點）。用儀表板 UTC 0 點日線，"
                     "收盤決定、照收盤價算。第一次上車後如果很快又被洗出去，那算在下一張表的假反彈裡；這裡只看第一次。"),
         "columns": ["規則"] + rows1[0][1:],
         "rows": rows1[1:]},
        {"caption": (f"從 {SINCE} 到 {_d(close.index[-1])}（比特幣日線，成本 {COST * 100:.2f}%）：吃到的漲幅、上車價比低點高多少（都是各次空頭的中位數；"
                     f"一半一半的做法取兩半的平均；「比 50 日均線低幾個百分點」是每次空頭兩者上車價離低點的差，正數 = 比 50 日均線更接近低點），以及「假反彈」= 上車後 {FALSE_DAYS} 天內就照規則賣出而且賠錢的次數、"
                     "這些次數的賠錢加總（每次的 % 直接相加）和最差一次。每年交易次數是全部已結束的進出次數。"),
         "columns": ["規則", "吃到下一段上漲（中位數）", "上車價比低點高｜比 50 日均線低幾個百分點（中位數）", "一年進出幾次",
                     f"假反彈（{FALSE_DAYS} 天內賠錢出場）", "假反彈賠錢加總", "最差的一次假反彈"],
         "rows": rows2},
        {"caption": (f"main_signal_check 的公平考試，原封不動：比特幣 {n_d} 個起算月、{len(r['clock']['starts'])} 個起算月 × 24 個收盤時間"
                     f"（{n_c:,} 格）、照一般人方式做（晚 10 小時、成本 0.25%、一成進場漏掉，每格 {msc.SIMS} 次）、其他幣，各算「報酬贏過一直抱著」"
                     "的比例；排名看四項裡最低的那項（前提：最深虧損比一直抱著淺的比例至少 95%）。最後一欄是三個條件：吃到的漲幅中位數比 50 日均線多、"
                     "四項最低不比 50 日均線差、最深虧損中位數不比 50 日均線深，✓ 是過、✗ 是沒過。恐慌訊號那兩條沒有其他幣的資料，只用三項比。"),
         "columns": ["規則", f"起算月（{n_d}）", f"起算×時間（{n_c:,}）", "一般人照做", "其他幣", "四項最低（排名）",
                     "最深虧損中位數｜每年多賺（中位數）", "三個條件"],
         "rows": rows3},
        {"caption": ("換到其他幣和 MSTR（只用價格的規則；照課本做、成本 0.10%，從每條規則都有值的第一個月到三年前，每個月 1 日起算）。"
                     "每格是「報酬贏過一直抱著的起算日個數｜最深虧損比一直抱著淺的個數」，分母是第一列。"),
         "columns": ["規則"] + assets + ["合計"],
         "rows": rows4},
    ]

    # verdict
    m50 = entries["ma50"]
    late = _median([e["above"] for e in m50 if e])
    b55 = _median([e["above"] for e in entries["b5520"] if e])
    early_best = max((k for k in FAST if k in cands and _ok(gap[k])), key=lambda k: (gap[k], med_share[k] or 0))
    fs_m = falses["ma50"]["per_year"]
    first = (f"2013 年以來 {len(bears)} 次跌一半以上的空頭，50 日均線在低點後第一次上車的價格中位數比低點高 {late:.0f}%"
             f"（55／20 高 {b55:.0f}%），大約吃到下一段上漲的 {med_share['ma50']:.0f}%。")
    fe = falses[early_best]
    second = (f"更早上車的做法裡，「{SHORT[early_best]}」最接近低點，上車價每次中位數比 50 日均線低 {gap[early_best]:.0f} 個百分點、"
              f"吃到 {med_share[early_best]:.0f}%，但假反彈一年 {fe['per_year']:.1f} 次（50 日均線 {fs_m:.1f} 次）。")
    if winners:
        w = winners[0]
        third = (f"照事先定的三個條件（吃到更多、公平考試分數不輸、最深虧損不更深），{len(winners)} 種做法勝過 50 日均線，最好的是「{SHORT[w]}」"
                 f"（四項最低 {_share(s[w]['low'])}，50 日均線 {_share(base['low'])}）。")
    else:
        cl = closest[0]
        miss = "、".join(n for n, ok in zip(("吃到更多漲幅", "公平考試分數", "最深虧損"), verdicts[cl]) if not ok)
        third = (f"照事先定的三個條件（吃到更多、公平考試分數不輸、最深虧損不更深），沒有一種做法勝過 50 日均線；最接近的是「{SHORT[cl]}」，差在{miss}"
                 f"（四項最低 {_share(s[cl]['low'])}，50 日均線 {_share(base['low'])}）。")
    verdict = first + second + third

    dropped = ""
    if len(bears) > 6:
        dropped = "表一欄位不夠，最早的 " + "、".join(f"{_d(b['low'])}" for b in bears[:len(bears) - 6]) + " 那次只算進中位數。"
    raw = r["raw"]
    spans = {k: f"{_d(v.index[0])} 起" for k, v in raw.items()}
    notes = [
        f"空頭只有 {len(bears)} 次（低點：" + "、".join(f"{_d(b['low'])} {b['low_px']:,.0f}" for b in bears)
        + "），每一次的樣子都不一樣，中位數只代表這幾次；目前這一段從 "
        + f"{_d(close[close == r['feat']['ath'].iloc[-1]].index[0])} 的高點下來還沒跌到一半，沒有算進去。" + dropped,
        "這些更早上車的規則，是在已經知道「50 日均線大跌後總是晚上車」之後才寫下來的；雖然寫好之後沒有再調任何數字（40%、20 日、10%、"
        "一年最低兩成、90 天都是事先定的），事後才知道的成分還在。",
        "提早上車的部位，在 50 日均線還沒接手前，收盤跌破 20 日均線就賣出；50 日均線接手後照 50 日均線進出。"
        "第 5 條（50 日均線止跌）照題目寫法是「均線不再往下而且收在均線上」，這只會和 50 日均線同一天或更晚上車，它其實是在測「少一點假反彈」，不是更早上車。",
        "恐慌訊號的資料長短不一：" + "、".join(f"{ {'fng': '恐懼與貪婪指數', 'funding': '幣安資金費率', 'hash': '算力', 'mvrv': 'MVRV'}[k]}{v}"
                                      for k, v in spans.items())
        + "。2018 年以前只有算力和 MVRV 能觸發；MVRV 和算力是事後整理的數值，可能被修正過。所有恐慌訊號都晚一天才用。",
        "上車價用決定那天的收盤價；「吃到幾成」只看第一次上車，之後如果被洗出去、再上車，差價沒有算在這裡（算在假反彈裡）。"
        "一半一半的做法，吃到的比例是兩半各自的平均。",
        "公平考試沿用 main_signal_check 的程式、起算日、收盤時間、模擬次數和亂數種子，所以 50 日均線、55／20、17 條過半的數字和那份研究一樣，可以直接比。"
        "相鄰的起算日共用大部分歷史、24 個收盤時間只差幾小時，這些「幾個裡幾個」不是互相獨立的證據。",
        "成本很粗略：照課本做 0.10%、一般人 0.25%，沒有算滑價、稅和現金利息。",
        "不偷看未來的檢查（寫這份研究時做的）：把 5 個切點之後的比特幣收盤價隨機打亂、恐慌訊號隨機改掉，再算一次所有規則，"
        "切點以前每一天的持有狀態完全沒變；以太幣上把切點之後的價格砍半也一樣。",
        "過去的結果不保證未來會一樣。這裡只比較固定規則過去做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]
    return {
        "id": "rally_timing",
        "title": "大跌後能不能更早上車",
        "question": "比特幣大跌之後，能不能比 50 日均線更接近低點上車，又不常被假反彈（漲一下又跌回去）騙進去？",
        "asof": _d(close.index[-1]),
        "verdict": verdict,
        "now": _now(r),
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "blockchain.info 每日比特幣價格（2017-08 以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
            {"name": "Yahoo Finance：MSTR 調整後日收盤", "url": "https://finance.yahoo.com/quote/MSTR/"},
            {"name": "alternative.me 恐懼與貪婪指數", "url": "https://alternative.me/crypto/fear-and-greed-index/"},
            {"name": "幣安 BTCUSDT 永續合約資金費率歷史",
             "url": "https://www.binance.com/zh-TC/futures/funding-history/perpetual/funding-fee-history"},
            {"name": "blockchain.info 全網算力（備用：CoinMetrics 社群版）", "url": "https://www.blockchain.com/explorer/charts/hash-rate"},
            {"name": "CoinMetrics 社群版 API（MVRV）", "url": "https://charts.coinmetrics.io/crypto-data/"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
