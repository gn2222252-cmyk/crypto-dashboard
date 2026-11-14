"""Historical studies for bitcoin, recomputed on every refresh and written to cache/_studies.json.

These are not indicators (collect() returns no rows). Each study answers one "what happened before" question with
the whole public history, so the page can show the evidence next to today's reading:

- cycles: halving -> peak -> low timetable of every cycle, and how long each bear market stayed in low zones
- oversold: what followed every time the daily RSI first fell below 25
- washout: what followed "fear and negative funding" days compared with "fear but positive funding" days
- fib: do pullbacks end at Fibonacci retracement levels more often than at five random lines
- structure: the last swing highs and lows (are they rising or falling)
- levels: moving averages and model prices, as price levels
- leverage: a daily 2x fund (MSTU) against its underlying share (MSTR): prices, volatility, decay, and how
  different mixes ended over every historical window

Sources (public, no key): blockchain.info daily price, Binance spot klines and funding history, alternative.me
fear and greed index, Yahoo Finance chart endpoint for MSTR and MSTU. Everything is descriptive and in-sample:
few independent cases, overlapping windows, no trading costs. A study that fails keeps its previous result.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from dashboard import common
from dashboard.collectors import classic
from dashboard.collectors import price as px

PATH = common.CACHE / "_studies.json"
HALVINGS = ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")
FIB = (0.236, 0.382, 0.5, 0.618, 0.786)
MSTR_SINCE = "2020-08-11"  # the day the company announced its first bitcoin purchase
FUND_COST = (0.0105 + 0.05) / 252  # a 2x fund's fee plus financing of the borrowed half, per trading day (rough)
_memo: dict = {}


# ---------- data ----------

def _close() -> pd.Series:
    """Closed daily candles since 2010, DatetimeIndex."""
    if "close" not in _memo:
        series = classic._long_close()
        _memo["close"] = pd.Series(series.values, index=pd.to_datetime(series.index))
    return _memo["close"]


def _ohlc() -> pd.DataFrame:
    """Binance daily high, low, close since 2017 (closed candles only)."""
    if "ohlc" not in _memo:
        rows, end = [], None
        for _ in range(5):
            params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000}
            if end is not None:
                params["endTime"] = end
            page = px._binance("/api/v3/klines", params)
            if not page:
                break
            rows = page + rows
            end = page[0][0] - 1
            if len(page) < 1000:
                break
        frame = pd.DataFrame([[r[0], float(r[2]), float(r[3]), float(r[4])] for r in rows],
                             columns=["ms", "high", "low", "close"]).drop_duplicates("ms").sort_values("ms")
        frame.index = pd.to_datetime(frame["ms"], unit="ms")
        _memo["ohlc"] = frame[["high", "low", "close"]].iloc[:-1]
    return _memo["ohlc"]


def _funding() -> pd.Series:
    """Binance BTCUSDT perpetual funding, summed per day and annualised (%), since 2019."""
    if "funding" not in _memo:
        out, start = [], int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
        for _ in range(12):
            page = common.get_json("https://fapi.binance.com/fapi/v1/fundingRate",
                                   {"symbol": "BTCUSDT", "limit": 1000, "startTime": start}, timeout=30)
            if not page:
                break
            out += page
            start = page[-1]["fundingTime"] + 1
            if len(page) < 1000:
                break
        series = pd.Series([float(p["fundingRate"]) for p in out],
                           index=pd.to_datetime([p["fundingTime"] for p in out], unit="ms"))
        _memo["funding"] = series.resample("1D").sum() * 365 * 100
    return _memo["funding"]


def _fng() -> pd.Series:
    if "fng" not in _memo:
        data = common.get_json("https://api.alternative.me/fng/", {"limit": 0}, timeout=30)["data"]
        _memo["fng"] = pd.Series({pd.to_datetime(int(d["timestamp"]), unit="s").normalize(): float(d["value"])
                                  for d in data}).sort_index()
    return _memo["fng"]


def _yahoo(symbol: str) -> pd.Series:
    """Adjusted daily closes from Yahoo Finance's public chart endpoint."""
    reply = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                         params={"range": "10y", "interval": "1d"}, headers=common.HEADERS, timeout=30)
    reply.raise_for_status()
    time.sleep(common.PAUSE)
    result = reply.json()["chart"]["result"][0]
    series = pd.Series(result["indicators"]["adjclose"][0]["adjclose"],
                       index=pd.to_datetime(result["timestamp"], unit="s").normalize()).dropna()
    if len(series) < 200:
        raise RuntimeError(f"Yahoo {symbol} 只回傳 {len(series)} 筆")
    return series


def _day(stamp) -> str:
    return pd.Timestamp(stamp).strftime("%Y-%m-%d")


def _f(value, digits: int = 4):
    """A JSON-safe rounded number (None for NaN)."""
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


# ---------- swing points ----------

def pivots(high: pd.Series, low: pd.Series, th: float) -> list[tuple]:
    """Zigzag: a swing ends when price reverses by th from its extreme. A bar can not both set the extreme and
    confirm the reversal away from it (the order inside one day is unknown). Returns [(date, price, 'H'|'L')]."""
    out, mode = [], None
    hi, hi_d, lo, lo_d = high.iloc[0], high.index[0], low.iloc[0], low.index[0]
    for d, h, l in zip(high.index[1:], high.values[1:], low.values[1:]):
        if mode in (None, "up"):
            if h > hi:
                hi, hi_d = h, d
            if hi_d != d and l <= hi * (1 - th):
                if mode is None:
                    out.append((lo_d, lo, "L"))
                out.append((hi_d, hi, "H"))
                mode, lo, lo_d = "down", l, d
                continue
        if mode in (None, "down"):
            if l < lo:
                lo, lo_d = l, d
            if lo_d != d and h >= lo * (1 + th):
                if mode is None:
                    out.append((hi_d, hi, "H"))
                out.append((lo_d, lo, "L"))
                mode, hi, hi_d = "up", h, d
    return out


# ---------- studies ----------

def cycles() -> dict:
    close = _close()
    ma2y = close.rolling(730).mean()
    cost = np.exp(np.log(close).rolling(200).mean())
    fitted = pd.Series(10 ** (5.84 * np.log10(classic._age(close.index.strftime("%Y-%m-%d"))) - 17.01),
                       index=close.index)
    ahr = (close / cost) * (close / fitted)
    last = close.index[-1]
    rows = []
    for i, halving in enumerate(HALVINGS):
        start = pd.Timestamp(halving)
        current = i + 1 == len(HALVINGS)
        # the cycle high came 372-546 days after each halving; look 800 days ahead so that the run-up into the
        # next halving is not mistaken for this cycle's high, and 600 days after the high for the low
        peak_d = close[start:min(start + pd.Timedelta(days=800), last)].idxmax()
        after = close[peak_d:min(peak_d + pd.Timedelta(days=600), last)]
        low_d = after.idxmin()
        window = close[peak_d:min(peak_d + pd.Timedelta(days=1100), last)]
        higher = close[low_d:][close[low_d:] > close[peak_d]]
        rows.append({"halving": halving, "current": current,
                     "peak_date": _day(peak_d), "peak": _f(close[peak_d], 0),
                     "days_to_peak": int((peak_d - start).days),
                     "low_date": _day(low_d), "low": _f(close[low_d], 0),
                     "days_peak_to_low": int((low_d - peak_d).days),
                     "days_halving_to_low": int((low_d - start).days),
                     "drawdown": _f((close[low_d] / close[peak_d] - 1) * 100, 1),
                     "days_below_2y": int((window < ma2y[window.index]).sum()),
                     "days_ahr_low": int((ahr[window.index] < 0.45).sum()),
                     "days_below_half": int((window < 0.5 * close[peak_d]).sum()),
                     "days_to_new_high": int((higher.index[0] - peak_d).days) if len(higher) else None,
                     "window_days": int((window.index[-1] - peak_d).days)})
    past = [r for r in rows if not r["current"]]
    cur = rows[-1]
    peak_d = pd.Timestamp(cur["peak_date"])
    lo, hi = min(r["days_peak_to_low"] for r in past), max(r["days_peak_to_low"] for r in past)
    return {"asof": _day(last), "rows": rows,
            "now": {"days_since_halving": int((last - pd.Timestamp(HALVINGS[-1])).days),
                    "days_since_peak": int((last - peak_d).days),
                    "drawdown_now": _f((close.iloc[-1] / close[peak_d] - 1) * 100, 1), "close": _f(close.iloc[-1], 0)},
            "window": {"min_days": lo, "max_days": hi, "from": _day(peak_d + pd.Timedelta(days=lo)),
                       "to": _day(peak_d + pd.Timedelta(days=hi)),
                       "passed": bool(last > peak_d + pd.Timedelta(days=hi))}}


def oversold(threshold: float = 25) -> dict:
    close = _close()
    rsi = px._rsi(close)
    days = close.index[(rsi < threshold) & (close.index >= "2017-01-01")]
    groups = [[days[0]]]
    for d in days[1:]:
        if (d - groups[-1][-1]).days <= 20:
            groups[-1].append(d)
        else:
            groups.append([d])
    rows = []
    for group in groups:
        d0 = group[0]
        seg = close[d0:].iloc[:91]
        rel = seg / seg.iloc[0] - 1

        def at(n):
            return _f(rel.iloc[n] * 100, 1) if len(rel) > n else None
        done = len(seg) >= 91
        hit = rel[rel <= -0.10]
        if len(hit):
            kind = "direct" if float(rel[:hit.index[0]].max()) < 0.05 else "bounce_first"
        else:
            kind = "no_drop" if done else "open"
        rows.append({"date": _day(d0), "close": _f(close[d0], 0), "rsi_min": _f(rsi[group[0]:group[-1]].min(), 1),
                     "r7": at(7), "r30": at(30), "r90": at(90), "low90": _f(rel.iloc[1:].min() * 100, 1) if len(rel) > 1 else None,
                     "kind": kind, "done": done})
    full = [r for r in rows if r["done"]]

    def stat(key):
        values = [r[key] for r in full if r[key] is not None]
        return {"up": sum(v > 0 for v in values), "n": len(values), "median": _f(np.median(values), 1)}
    return {"asof": _day(close.index[-1]), "threshold": threshold, "rsi_now": _f(rsi.iloc[-1], 1), "episodes": rows,
            "summary": {"n": len(full), "r7": stat("r7"), "r30": stat("r30"), "r90": stat("r90"),
                        "fell10": sum(1 for r in full if r["low90"] is not None and r["low90"] <= -10),
                        "low_median": _f(np.median([r["low90"] for r in full]), 1),
                        "kinds": {k: sum(1 for r in full if r["kind"] == k) for k in ("direct", "bounce_first", "no_drop")}}}


def washout() -> dict:
    close, fund, fng = _close(), _funding(), _fng()
    frame = pd.DataFrame({"c": close}).join(fund.rename("fund")).join(fng.rename("fng"))
    frame["fund7"] = frame["fund"].rolling(7).mean()
    frame["dd"] = frame["c"] / frame["c"].rolling(180, min_periods=60).max() - 1
    for h in (30, 90):
        frame[f"r{h}"] = frame["c"].shift(-h) / frame["c"] - 1
    frame["again"] = frame["c"][::-1].rolling(90, min_periods=90).min()[::-1].shift(-1) / frame["c"] - 1
    base = frame[frame["dd"] <= -0.20].dropna(subset=["fund7", "fng"])
    done = base.dropna(subset=["r90", "again"])
    states = (("washout", "很怕，而且做多的人已經在付錢給做空的人", (done["fund7"] < 0) & (done["fng"] <= 20)),
              ("fear_only", "很怕，但做多的人還在付錢", (done["fund7"] >= 0) & (done["fng"] <= 20)),
              ("other", "不算很怕", done["fng"] > 20))
    rows = []
    for key, label, mask in states:
        part = done[mask]
        if part.empty:
            continue
        runs = int((part.index.to_series().diff().dt.days > 14).sum() + 1)
        rows.append({"key": key, "label": label, "days": int(len(part)), "runs": runs,
                     "up30": _f((part["r30"] > 0).mean() * 100, 0), "med30": _f(part["r30"].median() * 100, 1),
                     "up90": _f((part["r90"] > 0).mean() * 100, 0), "med90": _f(part["r90"].median() * 100, 1),
                     "again10": _f((part["again"] <= -0.10).mean() * 100, 0)})
    last = frame.dropna(subset=["fund7"]).iloc[-1]
    fng_now = float(fng.iloc[-1])
    in_low = bool(last["dd"] <= -0.20)
    state = None if not in_low else "washout" if last["fund7"] < 0 and fng_now <= 20 else \
        "fear_only" if fng_now <= 20 else "other"
    # swing lows: fall of 25% or more, then a rise of 25% or more
    lows, mode = [], "up"
    series = frame["c"][frame.index >= "2019-09-10"]
    hi, hi_d, lo, lo_d = series.iloc[0], series.index[0], None, None
    for d, c in series.items():
        if mode == "up":
            if c > hi:
                hi, hi_d = c, d
            elif c <= hi * 0.75:
                mode, lo, lo_d = "down", c, d
        elif c < lo:
            lo, lo_d = c, d
        elif c >= lo * 1.25:
            window = frame.loc[lo_d - pd.Timedelta(days=6): lo_d + pd.Timedelta(days=1)]
            f7, g = float(window["fund7"].min()), float(window["fng"].min())
            lows.append({"date": _day(lo_d), "close": _f(lo, 0), "drop": _f((lo / hi - 1) * 100, 0),
                         "fund7_min": _f(f7, 1), "fng_min": _f(g, 0), "washout": bool(f7 < 0 and g <= 20)})
            mode, hi, hi_d = "up", c, d
    return {"asof": _day(frame.index[-1]), "rows": rows,
            "now": {"fund7": _f(last["fund7"], 1), "fng": _f(fng_now, 0), "dd180": _f(last["dd"] * 100, 1),
                    "in_low_zone": in_low, "state": state},
            "lows": lows, "lows_washout": sum(1 for r in lows if r["washout"]),
            "sample_from": _day(base.index[0]), "sample_days": int(len(done))}


def _retracements(pv: list[tuple]) -> np.ndarray:
    out = []
    for (_, a, _), (_, b, _), (_, c, _) in zip(pv, pv[1:], pv[2:]):
        r = (b - c) / (b - a)
        if 0 < r < 1:
            out.append(r)
    return np.array(out)


def fib(tol: float = 0.02, sets: int = 1500) -> dict:
    rng = np.random.default_rng(7)
    close, ohlc = _close(), _ohlc()
    random_sets = []
    while len(random_sets) < sets:
        levels = np.sort(rng.uniform(0.15, 0.9, len(FIB)))
        if np.min(np.diff(levels)) >= 2 * tol:
            random_sets.append(levels)

    def rate(r, levels):
        return float(np.mean(np.min(np.abs(r[:, None] - np.asarray(levels)[None, :]), axis=1) <= tol))
    rows, example = [], None
    for label, high, low in (("收盤價（2010 年起）", close, close), ("最高最低價（2017 年起）", ohlc["high"], ohlc["low"])):
        for th in (0.10, 0.15, 0.20):
            r = _retracements(pivots(high, low, th))
            if len(r) < 20:
                continue
            hit = rate(r, FIB)
            placebo = np.array([rate(r, levels) for levels in random_sets])
            rows.append({"data": label, "threshold": int(th * 100), "n": int(len(r)), "fib": _f(hit * 100, 0),
                         "random": _f(placebo.mean() * 100, 0), "rank": _f((placebo < hit).mean() * 100, 0),
                         "median": _f(np.median(r), 2)})
            if label.startswith("最高") and th == 0.15:
                counts, _ = np.histogram(r, bins=np.arange(0, 1.01, 0.1))
                example = {"n": int(len(r)), "hist": [int(c) for c in counts],
                           "near": [{"level": lv, "n": int(np.sum(np.abs(r - lv) <= tol)), "fib": lv in FIB}
                                    for lv in (0.382, 0.45, 0.5, 0.55, 0.618, 0.7, 0.786, 0.85)]}
    return {"asof": _day(close.index[-1]), "tol": tol, "levels": list(FIB), "random_sets": sets, "rows": rows,
            "passed": sum(1 for r in rows if r["rank"] is not None and r["rank"] >= 95), "example": example}


def structure() -> dict:
    ohlc = _ohlc()
    pv = pivots(ohlc["high"], ohlc["low"], 0.15)
    highs = [p for p in pv if p[2] == "H"]
    lows = [p for p in pv if p[2] == "L"]
    last_close = float(ohlc["close"].iloc[-1])
    # the swing that is still forming: the extreme since the last confirmed pivot
    since = ohlc.loc[pv[-1][0]:]
    forming = {"kind": "L" if pv[-1][2] == "H" else "H"}
    if forming["kind"] == "L":
        forming.update(date=_day(since["low"].idxmin()), price=_f(since["low"].min(), 0))
    else:
        forming.update(date=_day(since["high"].idxmax()), price=_f(since["high"].max(), 0))
    return {"asof": _day(ohlc.index[-1]), "threshold": 15, "close": _f(last_close, 0),
            "pivots": [{"date": _day(d), "price": _f(p, 0), "kind": k} for d, p, k in pv[-9:]],
            "forming": forming,
            "higher_highs": bool(len(highs) >= 2 and highs[-1][1] > highs[-2][1]),
            "higher_lows": bool(len(lows) >= 2 and lows[-1][1] > lows[-2][1]),
            "last_high": _f(highs[-1][1], 0), "prev_high": _f(highs[-2][1], 0),
            "last_low": _f(lows[-1][1], 0), "prev_low": _f(lows[-2][1], 0),
            "last_high_date": _day(highs[-1][0]), "last_low_date": _day(lows[-1][0])}


def levels() -> dict:
    close = _close()
    fit = classic._fit()
    model = float(fit["model"].iloc[-1])
    weekly = close.resample("W-SUN").last().dropna()
    if weekly.index[-1] > close.index[-1]:
        weekly = weekly.iloc[:-1]
    peak_d = close[HALVINGS[-1]:].idxmax()
    prev_start = pd.Timestamp(HALVINGS[-2])
    prev_peak = float(close[prev_start:prev_start + pd.Timedelta(days=800)].max())
    pre_halving = float(close[pd.Timestamp(HALVINGS[-1]) - pd.Timedelta(days=365):HALVINGS[-1]].max())
    low_after = close[peak_d:]
    out = [("ma200d", "200 日均線", close.rolling(200).mean().iloc[-1]),
           ("ma111d", "111 日均線", close.rolling(111).mean().iloc[-1]),
           ("ma350d", "350 日均線", close.rolling(350).mean().iloc[-1]),
           ("ma20w", "20 週均線", weekly.rolling(20).mean().iloc[-1]),
           ("ema21w", "21 週指數均線", weekly.ewm(span=21, adjust=False).mean().iloc[-1]),
           ("ma200w", "200 週均線", close.rolling(1400).mean().iloc[-1]),
           ("ma2y", "2 年均線", close.rolling(730).mean().iloc[-1]),
           ("cost200", "200 日定投成本", float(np.exp(np.log(close).rolling(200).mean().iloc[-1]))),
           ("power_mid", "冪律模型中線", model),
           ("power_floor", "冪律模型歷史下緣", model * 10 ** fit["lo"]),
           ("rainbow_1", "彩虹圖最底帶上緣", model * 10 ** fit["edges"][1]),
           ("prev_cycle_high", "上一輪牛市最高收盤", prev_peak),
           ("pre_halving_high", "上次減半前的高點", pre_halving),
           ("cycle_high", "這一輪最高收盤", float(close[peak_d])),
           ("cycle_low", "這一輪高點後的最低收盤", float(low_after.min()))]
    return {"asof": _day(close.index[-1]), "close": _f(close.iloc[-1], 0),
            "cycle_low_date": _day(low_after.idxmin()),
            "levels": [{"key": k, "label": label, "price": _f(v, 0)} for k, label, v in out if np.isfinite(v)]}


def leverage() -> dict:
    mstr, mstu = _yahoo("MSTR"), _yahoo("MSTU")
    mstr = mstr[mstr.index >= MSTR_SINCE]
    ret = mstr.pct_change().fillna(0)
    sim = (1 + (2 * ret - FUND_COST).clip(lower=-0.999)).cumprod()  # a simulated daily 2x fund over the whole history
    btc = _close().reindex(mstr.index)
    pair = pd.DataFrame({"m": ret, "b": btc.pct_change()}).dropna().tail(250)
    both = pd.DataFrame({"u": mstu, "m": mstr, "s": sim}).dropna()
    last = both.index[-1]
    # the real fund against twice the share, from a few anchor dates
    decay = []
    for label, days in (("近 1 個月", 21), ("近 3 個月", 63), ("近 6 個月", 126), ("近 1 年", 252)):
        if len(both) > days:
            a, b = both.iloc[-1 - days], both.iloc[-1]
            decay.append({"label": label, "from": _day(both.index[-1 - days]), "share": _f((b.m / a.m - 1) * 100, 1),
                          "fund": _f((b.u / a.u - 1) * 100, 1), "double": _f(2 * (b.m / a.m - 1) * 100, 1),
                          "sim": _f((b.s / a.s - 1) * 100, 1)})
    # same share price, different fund price: earlier days when the share closed within 2% of today's close
    now_m, now_u = float(both.m.iloc[-1]), float(both.u.iloc[-1])
    earlier = both[(abs(both.m / now_m - 1) <= 0.02) & (both.index < last - pd.Timedelta(days=45))]
    same = [{"date": _day(d), "share": _f(row.m, 2), "fund": _f(row.u, 2), "change": _f((now_u / row.u - 1) * 100, 1)}
            for d, row in earlier.iloc[::max(1, len(earlier) // 4)].tail(5).iterrows()]
    # every historical window: how portfolios with a given share of the 2x fund ended, as multiples of the start
    dd = _close() / _close().rolling(180).max() - 1
    low_zone = [d for d in mstr.index if d in dd.index and dd[d] <= -0.20]
    table = []
    for group, starts in (("low", low_zone), ("all", list(mstr.index))):
        index = mstr.index.get_indexer(starts)
        for horizon in (126, 252, 504):
            ok = index[index + horizon < len(mstr)]
            if len(ok) < 30:
                continue
            p = mstr.values[ok + horizon] / mstr.values[ok]
            u = sim.values[ok + horizon] / sim.values[ok]
            for share in (0, 0.1, 0.25, 0.5, 0.75, 1.0):
                v = share * u + (1 - share) * p
                table.append({"group": group, "days": horizon, "share": int(share * 100), "n": int(len(v)),
                              "up": _f((v > 1).mean() * 100, 0), "median": _f(np.median(v), 2),
                              "worst10": _f(np.percentile(v, 10), 2), "best10": _f(np.percentile(v, 90), 2),
                              "half": _f((v < 0.5).mean() * 100, 0)})
    vol = ret.rolling(252).std() * np.sqrt(252)
    return {"asof": _day(last), "share": "MSTR", "fund": "MSTU", "share_price": _f(now_m, 2), "fund_price": _f(now_u, 2),
            "share_from": MSTR_SINCE, "fund_from": _day(mstu.index[0]),
            "vol_90d": _f(ret.tail(90).std() * np.sqrt(252) * 100, 0), "vol_1y": _f(vol.iloc[-1] * 100, 0),
            "vol_min": _f(vol.min() * 100, 0), "vol_max": _f(vol.max() * 100, 0),
            "beta": _f(np.polyfit(pair.b, pair.m, 1)[0], 2), "corr": _f(pair.b.corr(pair.m), 2),
            "share_high": _f(mstr.max(), 2), "share_from_high": _f((now_m / mstr.max() - 1) * 100, 0),
            "fund_from_high": _f((now_u / mstu.max() - 1) * 100, 0),
            "decay": decay, "same_price": same, "table": table,
            "sim_check": [{"from": d["from"], "real": d["fund"], "sim": d["sim"]} for d in decay]}


STUDIES = (("cycles", cycles), ("oversold", oversold), ("washout", washout), ("fib", fib), ("structure", structure),
           ("levels", levels), ("leverage", leverage))


def collect() -> list[dict]:
    try:
        previous = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - first run or damaged file
        previous = {}
    payload = {"collected_utc": common.now_utc(), "errors": {}}
    for name, build in STUDIES:
        try:
            payload[name] = build()
            payload[name]["computed_utc"] = payload["collected_utc"]
        except Exception as exc:  # noqa: BLE001 - one study failing keeps its previous result
            payload["errors"][name] = f"{type(exc).__name__}: {exc}"[:200]
            if isinstance(previous.get(name), dict):
                payload[name] = previous[name]
            print(f"  統計 {name} 這次沒有更新：{payload['errors'][name]}")
    common.CACHE.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    return []


if __name__ == "__main__":
    collect()
    data = json.loads(PATH.read_text(encoding="utf-8"))
    for key, value in data.items():
        print(key, "->", json.dumps(value, ensure_ascii=False, default=str)[:400])
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), "done")
