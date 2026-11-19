"""Historical study "stablecoin_supply": does stablecoin supply growth (USDT, USDC, all stablecoins) LEAD bitcoin, or
does it only FOLLOW the price? And which mechanism is behind the link, if any?

Claims (practitioners; none shows an out-of-sample test): CryptoQuant and Bitcoin Magazine Pro chart stablecoin
supply growth / "stablecoin buying power" and say rising supply precedes bitcoin rallies; Cointelegraph's write-ups of
large USDT mints say big mints come before price moves. The opposite mechanism is just as plausible: price rises
first, traders bring in dollars to chase it, and supply follows - the way the ETF flows followed price in etf_loop.
Related earlier study: flow_filters (a 7-day-growth-above-median filter on a breakout rule); this study is separate.

Pre-registered design (written before any result was seen; not changed afterwards):
(1) Who moves first: lead/lag cross-correlation of weekly (and daily) supply growth vs BTC returns in both
    directions, and a Granger-style test (4 weekly lags) with a block bootstrap, both directions; also an expanding-
    window out-of-sample check of both directions.
(2) Forward 30/90-day BTC returns by trailing-year quintile of 30-day supply growth, split 2019-21 / 2022-23 /
    2024-now.
(3) Rules on top of ma50: S1 = ma50 AND 30-day supply growth > 0; S2 = ma50, but enter early when there is a supply
    burst (7-day growth in its trailing-year top 20%) while price is below ma50 (catching the start of a rally).
    Fair exam: main_signal_check's grid where data allows (monthly starts x 24 clocks), ordinary-person execution
    (0.25%, 10% missed entries), other coins ETH/SOL/BNB/XRP, ranking score = worst of the four "beats holding"
    shares, placebo = supply series circularly shifted >= 180 days (200 draws), median deepest loss.
Mechanisms (description, same style as etf_loop): price -> supply (next-30-day supply growth by quintile of the past
30-day BTC return), each direction controlling for the other, mint bursts versus a price-only twin (BTC 7-day return
in its trailing-year top 20%), and burns (30-day supply contraction) as a mark of weakness.
Pass test (stated now): predicted sign in all three periods AND ranking score >= ma50's AND median deepest loss no
worse AND placebo share <= 10%.

Details fixed with the design (before results):
  * Data: DefiLlama stablecoins API (free, key-less): /stablecoincharts/all for all stablecoins (sum of every peg in
    totalCirculatingUSD), ?stablecoin=1 (USDT) and ?stablecoin=2 (USDC). A point stamped day D is that day's supply.
    DefiLlama's coverage before mid-2019 is incomplete (USDT on Omni is missing: 2018 shows tens of millions, not
    billions) and chains are added later in single jumps, so: growth is measured from 2019-07-01 only, and a daily log
    change above +15% or below -15% is treated as a coverage jump and set to zero growth (2020-01-03 and 2020-04-03
    are such days in the total). Growth = log change; 7-day / 30-day growth = sums of the cleaned daily log changes.
  * Point in time: supply of day D is used only from the daily close of the candle labelled D+1 (00:00 UTC of D+2),
    i.e. one full day after it is final. On the 24-clock grid a close on UTC date E uses the daily state decided at
    the UTC close that ended day E-1 (supply up to E-2), as in blockspace. Forward returns start at the decision close.
  * Trailing-year quintile: share of the 365 prior values lower than today's (at least 330), in fifths (ls_ratio's
    method). Burst = 7-day growth above the 80% quantile of its 365 prior values (at least 330).
  * (1) Weekly = consecutive 7-day blocks of daily BTC log returns and of the known (lagged) supply growth. Lead
    correlation = corr(this week's supply growth, BTC return of the next 4 weeks); follow correlation = corr(this
    week's BTC return, supply growth of the next 4 weeks). 90% intervals: circular blocks of 8 weeks, 1,000 draws;
    one-sided p from 200 circular shifts (>= 26 weeks). Cross-correlation table lags -8..+8 weeks with a 95% band from
    the shifts. Granger: y_w on 4 own lags plus 4 lags of x; statistic = added R2; p = share of 1,000 draws in which x
    is replaced by a circular block resample of itself (8 weeks) with an added R2 at least as large. Out of sample:
    the same two models refitted every week on all earlier weeks (at least 104), OOS R2 of the full model against the
    own-lags-only model. Total supply is the main series; USDT and USDC are reported the same way.
  * (2) Main effect = slope of the forward 30-day BTC return on the quintile number (1..5) of 30-day total supply
    growth; predicted sign positive. 90-day reported too. 90% intervals with circular blocks (30 days for 30-day,
    120 days for 90-day returns), 1,000 draws.
  * (3) S1 sign condition = the (2) 30-day slope positive in all three periods. S2 sign condition = among days with
    the price below ma50, mean next-30-day return on burst days minus on non-burst days, positive in all three periods.
    ma50 = close above its 50-day average (main_signal_check / blockspace definition and seeds). Grid: monthly starts
    from the first month after both rules have values to main_signal_check's last start (2023-12), UTC closes
    (textbook 0.10%) and the same months x 24 closing hours (textbook, and ordinary person: 10 hours later, 0.25%,
    10% of entries missed, 200 simulations per clock). ma50 is scored on exactly the same starts. Other coins: their
    own 50-day average, BTC-market stablecoin signal as a filter (the same total-supply flags), monthly starts from the
    first month the flags exist to three years before today. Placebo: the total-supply daily growth series rolled
    circularly by 180 ... (n - 180) days, 200 draws, flags and rules rebuilt; to keep the run within minutes the placebo
    score is the ranking score without the ordinary-person check, and the real rule is scored the same way for that
    comparison (ls_ratio's choice). Median deepest loss = median over the start x clock textbook cells.
  * Mechanisms: price -> supply = slope of next-30-day total supply growth on the quintile of the past 30-day BTC
    return; "controlling for the other" = least squares with both quintile numbers. Mint-burst events and price-twin
    events are declustered (a new event needs 30 days without the flag); excess = mean forward return after events
    minus the mean over all days with a known flag. Burns = 30-day growth < 0.
  * Look-ahead check: prices and supply after three cut dates scrambled -> no state up to the cut may change.
Only public key-less endpoints through common.get_json (fixed user agent). Not advice.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from dashboard import common
from dashboard.studies2 import clock_robustness as cr
from dashboard.studies2 import crash_timing as ct
from dashboard.studies2 import main_signal_check as msc
from dashboard.studies2 import robust_trend as rt
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

LLAMA = "https://stablecoins.llama.fi/stablecoincharts/all"
LLAMA_PAGE = "https://defillama.com/stablecoins"
SERIES = (("all", None), ("usdt", 1), ("usdc", 2))
CN = {"all": "全部穩定幣", "usdt": "USDT（泰達幣）", "usdc": "USDC"}
START = "2019-07-01"
JUMP = 0.15
WIN, WIN_MIN, TOP_Q = 365, 330, 0.80
COST, P_COST, P_MISS, SIMS, SEED = msc.COST, msc.P_COST, msc.P_MISS, msc.SIMS, msc.SEED
YEAR = 365
PERIODS = (("2019–2021", "2019-01-01", "2021-12-31"), ("2022–2023", "2022-01-01", "2023-12-31"),
           ("2024 到今天", "2024-01-01", "2100-01-01"))
BOOT, PLACEBO, PL_GAP = 1000, 200, 180
BLOCK = {30: 30, 90: 120}
WBLOCK, WLAGS, WAHEAD, WSHIFT, MIN_TRAIN = 8, 4, 4, 26, 104
DECLUSTER = 30
KEYS = ("ma50", "s1", "s2")
TESTED = ("s1", "s2")
KI = {"ma50": 1, "s1": 75, "s2": 76}  # ma50 uses main_signal_check's random draws
NAMES = {"ma50": "50 日均線（基準）", "s1": "S1：50 日均線，而且 30 天穩定幣供給成長 > 0 才持有",
         "s2": "S2：50 日均線，但價格在均線下時遇到「增發爆量」就提早進場"}
SHORT = {"ma50": "50 日均線", "s1": "S1", "s2": "S2"}
OTHER = (("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP"))
CUTS = ("2021-06-30", "2023-03-31", "2025-06-30")
DAY = pd.Timedelta(days=1)

_d = cr._d


def _ok(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


def _p(v) -> str:
    """fraction -> signed percent"""
    return f"{float(v) * 100:+.1f}%" if _ok(v) else "—"


def _pt(v) -> str:
    return f"{float(v) * 100:+.1f} 點" if _ok(v) else "—"


def _share(v) -> str:
    return f"{float(v) * 100:.0f}%" if _ok(v) else "—"


def _c(v) -> str:
    return f"{float(v):+.2f}" if _ok(v) else "—"


def _ci(lo, hi, f) -> str:
    return f"［{f(lo)}，{f(hi)}］" if _ok(lo) and _ok(hi) else "［—］"


def _of(n: int, m: int) -> str:
    return f"{n:,} 個裡 {m:,} 個"


# ---------- data ----------

def _fetch() -> dict:
    out = {}
    for key, sid in SERIES:
        reply = common.get_json(LLAMA, {"stablecoin": sid} if sid else None, timeout=60)
        pts = {}
        for p in reply:
            tot = sum(float(v) for v in (p.get("totalCirculatingUSD") or {}).values() if v is not None)
            if tot > 0:
                pts[pd.Timestamp(int(p["date"]), unit="s").strftime("%Y-%m-%d")] = tot
        if len(pts) < 1500:
            raise RuntimeError(f"DefiLlama {key} 只有 {len(pts)} 天")
        out[key] = pts
    return out


def _load() -> tuple[pd.DataFrame, list[str]]:
    path = common.STORE / "stablecoin_supply_llama.json"
    problems = []
    try:
        got = _fetch()
        try:
            common.STORE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(got), encoding="utf-8")
        except Exception:  # noqa: BLE001 - the store is only a fallback
            pass
    except Exception as exc:  # noqa: BLE001 - a refusal is reported, never worked around
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"DefiLlama 穩定幣資料取不到（{type(exc).__name__}），也沒有存下來的資料") from exc
        problems.append(f"DefiLlama 這次取不到（{type(exc).__name__}），用之前存下來的資料。")
    raw = pd.DataFrame({k: pd.Series(v, dtype=float) for k, v in got.items()})
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    raw = raw[raw.index < pd.Timestamp(common.today_utc())]  # today's point is still moving
    return raw.asfreq("D").ffill(), problems


def _growth(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleaned daily log growth from START (coverage jumps set to zero)."""
    lv = raw[raw.index >= pd.Timestamp(START)]
    g = np.log(lv.where(lv > 0)).diff().iloc[1:]
    jumps = {c: [_d(t) for t in g.index[(g[c].abs() > JUMP).values]] for c in g.columns}
    g = g.where(g.abs() <= JUMP, 0.0).fillna(0.0)
    return g, jumps


# ---------- signals ----------

def _pos(s: pd.Series) -> pd.Series:
    """Share of the 365 prior values lower than today's (NaN without 330 of them)."""
    v = s.values.astype(float)
    out = np.full(len(v), np.nan)
    for i in range(WIN_MIN, len(v)):
        if not math.isfinite(v[i]):
            continue
        w = v[max(0, i - WIN):i]
        w = w[np.isfinite(w)]
        if len(w) >= WIN_MIN:
            out[i] = float((w < v[i]).mean())
    return pd.Series(out, index=s.index)


def _quint(pos: pd.Series) -> pd.Series:
    return (np.floor(pos * 5).clip(0, 4) + 1).where(pos.notna())


def _top(s: pd.Series) -> pd.Series:
    q = s.shift(1).rolling(WIN, min_periods=WIN_MIN).quantile(TOP_Q)
    return (s > q).astype(float).where(q.notna() & s.notna())


def _sig(g: pd.Series, quint: bool = True) -> pd.DataFrame:
    """On supply dates: 7/30-day growth, burst flag, growth flag, quintile of 30-day growth."""
    g7 = g.rolling(7, min_periods=7).sum()
    g30 = g.rolling(30, min_periods=30).sum()
    out = pd.DataFrame({"g7": g7, "g30": g30, "burst": _top(g7), "grow": (g30 > 0).astype(float).where(g30.notna())})
    if quint:
        out["q30"] = _quint(_pos(g30))
    return out


def _at(series: pd.Series, dates, lag: int) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(dates) - lag * DAY).values.astype(float)


def _ma(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return ct._gt(c, ct._mean(c, 50))


def _rules(ma: np.ndarray, grow: np.ndarray, burst: np.ndarray) -> dict:
    s1 = np.where(np.isnan(ma) | np.isnan(grow), np.nan, np.nan_to_num(ma) * np.nan_to_num(grow))
    s2 = np.where(np.isnan(ma) | np.isnan(burst), np.nan, np.maximum(np.nan_to_num(ma), np.nan_to_num(burst)))
    return {"ma50": ma, "s1": s1, "s2": s2}


def _states(close: pd.Series, sg: pd.DataFrame) -> dict:
    """Daily decision states on the BTC UTC close labels (supply of day D used from the candle labelled D+1)."""
    grow, burst = _at(sg["grow"], close.index, 1), _at(sg["burst"], close.index, 1)
    out = _rules(_ma(close.values), grow, burst)
    out["grow"], out["burst"] = grow, burst
    return out


# ---------- fair grid ----------

def _grid_daily(close: pd.Series, pos: dict, first, last, keys=KEYS) -> dict:
    starts, labels = msc._month_starts(close.index, first, last)
    change = close.pct_change().fillna(0.0).values
    hr, hd = msc._suffix(msc._equity(change), starts, YEAR)
    out = {"starts": labels, "hold": (hr[:, 0], hd[:, 0])}
    for k in keys:
        net, _ = te._run(np.concatenate([[np.nan], pos[k][:-1]]), change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, YEAR)
        out[k] = (r[:, 0], dd[:, 0])
    return out


class _Clock:
    """The 24 closing hours of main_signal_check's hourly grid, ma50 per hour, from the given start months."""

    def __init__(self, first, last):
        rows, self.problem = cr._hourly("BTCUSDT")
        if len(rows) < cr.MIN_HOURS:
            raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
        price, b0, _ = cr._grid(rows)
        pre = cr._prehistory(b0)
        if not len(pre):
            raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
        self.m = cr._Market(price, b0, pre)
        self.eng = rt._Engine(self.m)
        self.days = self.m.days
        self.starts, self.labels = msc._month_starts(self.m.dates, first, last)
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hr, self.hd = hr[:, 0], hd[:, 0]
        npre = len(pre)
        self.ma = [np.nan_to_num(_ma(np.concatenate([pre, price[h::24]]))[npre:][:self.days], nan=0.0)
                   for h in range(24)]

    def flags(self, daily_index: pd.DatetimeIndex, st: dict) -> dict:
        """A close on UTC date E uses the daily state decided at the close that ended day E-1."""
        return {k: np.nan_to_num(_at(pd.Series(st[k], index=daily_index), self.m.dates, 1), nan=0.0)
                for k in ("grow", "burst")}

    def want(self, k: str, h: int, fl: dict) -> np.ndarray:
        if k == "ma50":
            return self.ma[h]
        if k == "s1":
            return self.ma[h] * fl["grow"]
        return np.maximum(self.ma[h], fl["burst"])

    def run(self, k: str, fl: dict, person: bool = True) -> dict:
        n = len(self.starts)
        tr, td = np.zeros((n, 24)), np.zeros((n, 24))
        pr = np.zeros((n, 24, SIMS)) if person else None
        for h in range(24):
            w = self.want(k, h, fl)
            r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.starts, YEAR)
            tr[:, h], td[:, h] = r[:, 0], dd[:, 0]
            if person:
                b, wl = rt._later(h, w, self.days)
                rng = np.random.default_rng(SEED + 1000 * KI[k] + h)
                pos = rt._missed(wl, P_MISS, rng, SIMS)
                r, _dd = msc._suffix(np.vstack([np.ones((1, SIMS)), self.eng._mid(b, pos, P_COST, None)]),
                                     self.starts, YEAR)
                pr[:, h, :] = r
        edge = tr - self.hr[:, None]
        out = {"c_n": edge.size, "c_beat": int((edge > 0).sum()), "c_edge": float(np.median(edge)),
               "c_shallow": int((td > self.hd[:, None]).sum()), "c_deep_med": float(np.median(td))}
        if person:
            pedge = pr - self.hr[:, None, None]
            out.update({"p_n": pedge.size, "p_beat": int((pedge > 0).sum()), "p_edge": float(np.median(pedge))})
        return out


def _other_pos(c: pd.Series, btc_index: pd.DatetimeIndex, st: dict) -> dict:
    """Other coins: own 50-day average, BTC-market stablecoin flags (same UTC calendar, same decision close)."""
    grow = _at(pd.Series(st["grow"], index=btc_index), c.index, 0)
    burst = _at(pd.Series(st["burst"], index=btc_index), c.index, 0)
    return _rules(_ma(c.values), grow, burst)


def _first_known(index: pd.DatetimeIndex, pos: dict, keys=KEYS) -> pd.Timestamp:
    known = np.ones(len(index), dtype=bool)
    for k in keys:
        known &= ~np.isnan(pos[k])
    if not known.any():
        raise RuntimeError("規則沒有值")
    return (index[int(np.argmax(known))] + pd.offsets.MonthBegin(1)).normalize()


def _exam(close, pos, clock, fl, others, opos, first, keys=KEYS, person=True) -> dict:
    d = _grid_daily(close, pos, first, msc.DAILY_STARTS[1], keys)
    out = {}
    for k in keys:
        x = clock.run(k, fl, person)
        dr, dd = d[k]
        x["d_n"], x["d_beat"] = len(dr), int((dr > d["hold"][0]).sum())
        x["d_shallow"] = int((dd > d["hold"][1]).sum())
        x["d_edge"] = float(np.median(dr - d["hold"][0]))
        o_n = o_beat = 0
        x["o_each"] = {}
        for a, (c, fa) in others.items():
            g = _grid_daily(c, opos[a], fa, others_last(), (k,))
            n_a, b_a = len(g[k][0]), int((g[k][0] > g["hold"][0]).sum())
            x["o_each"][a] = (n_a, b_a)
            o_n, o_beat = o_n + n_a, o_beat + b_a
        x["o_n"], x["o_beat"] = o_n, o_beat
        checks = [x["d_beat"] / x["d_n"], x["c_beat"] / x["c_n"]]
        if person:
            checks.append(x["p_beat"] / x["p_n"])
        checks.append(o_beat / o_n if o_n else 0.0)
        x["checks"], x["low"] = checks, min(checks)
        out[k] = x
    out["n_daily"] = len(d["starts"])
    out["first_label"] = d["starts"][0] if d["starts"] else None
    return out


def others_last() -> pd.Timestamp:
    return pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)


# ---------- statistics ----------

def _blocks(n: int, length: int, draws: int, rng) -> np.ndarray:
    nb = -(-n // length)
    st = rng.integers(0, n, (draws, nb))
    return ((st[:, :, None] + np.arange(length)) % n).reshape(draws, -1)[:, :n]


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 60 or np.var(x) < 1e-12:
        return None
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))


def _slope_ci(x: np.ndarray, y: np.ndarray, block: int, rng) -> tuple:
    est = _slope(x, y)
    if est is None:
        return None, None, None, len(x)
    idx = _blocks(len(x), block, BOOT, rng)
    xs, ys = x[idx], y[idx]
    xm, ym = xs.mean(1, keepdims=True), ys.mean(1, keepdims=True)
    var = ((xs - xm) ** 2).mean(1)
    good = var > 1e-12
    b = ((xs - xm) * (ys - ym)).mean(1)[good] / var[good]
    return est, float(np.percentile(b, 5)), float(np.percentile(b, 95)), len(x)


def _diff_ci(flag: np.ndarray, y: np.ndarray, block: int, rng) -> tuple:
    """mean(y | flag) - mean(y | not flag) with a circular block interval."""
    if flag.sum() < 10 or (~flag).sum() < 10:
        return None, None, None, int(flag.sum())
    est = float(y[flag].mean() - y[~flag].mean())
    idx = _blocks(len(y), block, BOOT, rng)
    f, v = flag[idx].astype(float), y[idx]
    n1, n0 = f.sum(1), (1 - f).sum(1)
    ok = (n1 > 0) & (n0 > 0)
    d = ((f * v).sum(1)[ok] / n1[ok]) - (((1 - f) * v).sum(1)[ok] / n0[ok])
    return est, float(np.percentile(d, 5)), float(np.percentile(d, 95)), int(flag.sum())


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _runs(m: np.ndarray) -> int:
    m = np.asarray(m, dtype=bool)
    return int(m[0]) + int((m[1:] & ~m[:-1]).sum()) if len(m) else 0


def _events(flag: np.ndarray) -> np.ndarray:
    """First flag day after DECLUSTER days without it."""
    f = np.nan_to_num(flag, nan=0.0) > 0.5
    ev = np.zeros(len(f), dtype=bool)
    last = -10 ** 9
    for i in np.flatnonzero(f):
        if i - last > DECLUSTER:
            ev[i] = True
        last = i
    return ev


# ---------- (1) who moves first: weekly ----------

def _weekly(close: pd.Series, g: pd.Series) -> pd.DataFrame:
    """Consecutive 7-day blocks of BTC log returns and known supply growth (supply of D counted on label D+1)."""
    r = np.log(close).diff()
    gk = pd.Series(_at(g, close.index, 1), index=close.index)
    df = pd.DataFrame({"b": r, "s": gk}).dropna()
    df = df[df.index >= pd.Timestamp(START) + 2 * DAY]
    n = len(df) // 7 * 7
    df = df.iloc[len(df) - n:]
    w = df.groupby(np.arange(n) // 7).sum()
    w.index = df.index[6::7]
    return w


def _ahead(v: np.ndarray, k: int) -> np.ndarray:
    """Sum of the next k values (NaN where fewer than k remain)."""
    out = np.zeros(len(v))
    for j in range(1, k + 1):
        nxt = np.full(len(v), np.nan)
        nxt[:-j] = v[j:]
        out = out + nxt
    return out


def _lead(x: np.ndarray, y: np.ndarray, rng) -> dict:
    """corr(x_w, y over the next WAHEAD weeks), block interval and circular-shift p (one-sided, positive)."""
    ya = _ahead(y, WAHEAD)
    m = np.isfinite(x) & np.isfinite(ya)
    xv, yv = x[m], ya[m]
    est = _corr(xv, yv)
    idx = _blocks(len(xv), WBLOCK, BOOT, rng)
    bs = [_corr(xv[i], yv[i]) for i in idx]
    bs = np.array([b for b in bs if b is not None])
    sh = []
    n = len(x)
    for _ in range(PLACEBO):
        s = int(rng.integers(WSHIFT, n - WSHIFT))
        xs = np.roll(x, s)
        sh.append(_corr(xs[m], yv) or 0.0)
    sh = np.array(sh)
    return {"est": est, "lo": float(np.percentile(bs, 5)), "hi": float(np.percentile(bs, 95)), "n": int(m.sum()),
            "p": float(np.mean(sh >= est - 1e-12)) if est is not None else None}


def _xcorr(s: np.ndarray, b: np.ndarray, rng) -> dict:
    """corr(s_w, b_{w+k}) for k = -8..8 and a 95% band of |corr| from circular shifts of s."""
    out = {}
    for k in range(-8, 9):
        if k >= 0:
            out[k] = _corr(s[:len(s) - k], b[k:])
        else:
            out[k] = _corr(s[-k:], b[:len(b) + k])
    sh = []
    for _ in range(PLACEBO):
        ss = np.roll(s, int(rng.integers(WSHIFT, len(s) - WSHIFT)))
        sh.append(abs(_corr(ss, b) or 0.0))
    out["band"] = float(np.percentile(sh, 95))
    return out


def _lags(v: np.ndarray, p: int) -> np.ndarray:
    return np.column_stack([v[p - j:len(v) - j] for j in range(1, p + 1)])


def _r2(y, X) -> float:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    return 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum())


def _granger(y: np.ndarray, x: np.ndarray, rng) -> dict:
    p = WLAGS
    Y = y[p:]
    one = np.ones((len(Y), 1))
    Xr = np.hstack([one, _lags(y, p)])
    Xf = np.hstack([Xr, _lags(x, p)])
    r2r = _r2(Y, Xr)
    beta = np.linalg.lstsq(Xf, Y, rcond=None)[0]
    e = Y - Xf @ beta
    add = 1 - float(e @ e) / float(((Y - Y.mean()) ** 2).sum()) - r2r
    csum = float(beta[1 + p:].sum())
    null = []
    for idx in _blocks(len(x), WBLOCK, BOOT, rng):
        xb = x[idx]
        null.append(_r2(Y, np.hstack([Xr, _lags(xb, p)])) - r2r)
    # out of sample: expanding refit every week
    sse_f = sse_r = 0.0
    n_oos = 0
    for t in range(MIN_TRAIN, len(Y)):
        bf = np.linalg.lstsq(Xf[:t], Y[:t], rcond=None)[0]
        br = np.linalg.lstsq(Xr[:t], Y[:t], rcond=None)[0]
        sse_f += float((Y[t] - Xf[t] @ bf) ** 2)
        sse_r += float((Y[t] - Xr[t] @ br) ** 2)
        n_oos += 1
    return {"add": add, "p": float(np.mean(np.array(null) >= add - 1e-15)), "csum": csum, "n": len(Y),
            "oos": (1 - sse_f / sse_r) if sse_r > 0 else None, "n_oos": n_oos}


def _who_first(close: pd.Series, g: pd.DataFrame, rng) -> dict:
    out = {}
    for c in g.columns:
        w = _weekly(close, g[c])
        s, b = w["s"].values, w["b"].values
        res = {"weeks": len(w), "from": w.index[0], "lead": _lead(s, b, rng), "follow": _lead(b, s, rng),
               "xc": _xcorr(s, b, rng), "g_sb": _granger(b, s, rng), "g_bs": _granger(s, b, rng)}
        r = np.log(close).diff()
        dd = pd.DataFrame({"b": r, "s": _at(g[c], close.index, 1)}, index=close.index).dropna()
        dd = dd[dd.index >= pd.Timestamp(START) + 2 * DAY]
        ds, db = dd["s"].values, dd["b"].values
        res["daily"] = {k: (_corr(ds[:len(ds) - k], db[k:]) if k >= 0 else _corr(ds[-k:], db[:len(db) + k]))
                        for k in (-30, -7, -1, 0, 1, 7, 30)}
        res["daily30"] = (_corr(ds, _ahead(db, 30)), _corr(db, _ahead(ds, 30)))
        res["per"] = {}
        for name, a, z in PERIODS:
            m = (w.index >= pd.Timestamp(a)) & (w.index <= pd.Timestamp(z))
            sv, bv = s[m], b[m]
            res["per"][name] = (_corr(sv, _ahead(bv, WAHEAD)), _corr(bv, _ahead(sv, WAHEAD)), int(m.sum()))
        out[c] = res
    return out


# ---------- (2) and mechanisms: daily ----------

def _frame(close: pd.Series, sigs: dict, st: dict) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    for c, sg in sigs.items():
        df[f"q_{c}"] = _at(sg["q30"], close.index, 1)
        df[f"g30_{c}"] = _at(sg["g30"], close.index, 1)
    df["burst"], df["grow"], df["ma"] = st["burst"], st["grow"], st["ma50"]
    for h in (30, 90):
        df[f"f{h}"] = close.shift(-h) / close - 1
    past = close / close.shift(30) - 1
    df["qb"] = _quint(_pos(past))
    df["fsup"] = df["g30_all"].shift(-30)  # total supply growth over the next 30 days, known 30 days later
    b7 = close / close.shift(7) - 1
    df["b7top"] = _top(b7)
    return df[df.index >= pd.Timestamp(START)]


def _sub(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]


def _quint_tests(df: pd.DataFrame, rng) -> dict:
    out = {}
    for c in ("all", "usdt", "usdc"):
        res = {}
        for h in (30, 90):
            ok = df[f"q_{c}"].notna() & df[f"f{h}"].notna()
            x, y = df[f"q_{c}"][ok].values, df[f"f{h}"][ok].values
            res[h] = {"full": _slope_ci(x, y, BLOCK[h], rng),
                      "rows": [float(y[x == i].mean()) if (x == i).any() else None for i in range(1, 6)],
                      "days": [int((x == i).sum()) for i in range(1, 6)], "mean": float(y.mean())}
            for name, a, b in PERIODS:
                s = _sub(df, a, b)
                ok2 = s[f"q_{c}"].notna() & s[f"f{h}"].notna()
                xs, ys = s[f"q_{c}"][ok2].values, s[f"f{h}"][ok2].values
                res[h][name] = {"slope": _slope(xs, ys), "n": len(xs),
                                "rows": [float(ys[xs == i].mean()) if (xs == i).any() else None for i in range(1, 6)]}
        out[c] = res
    return out


def _partial(y: np.ndarray, a: np.ndarray, b: np.ndarray, block: int, rng) -> tuple:
    """Coefficient of a in y ~ 1 + a + b, with a circular block interval."""
    X = np.column_stack([np.ones(len(y)), a, b])
    est = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    bs = []
    for idx in _blocks(len(y), block, BOOT, rng):
        bs.append(float(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][1]))
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95)), len(y)


def _mechanisms(df: pd.DataFrame, rng) -> dict:
    out = {}
    # price -> supply
    ok = df["qb"].notna() & df["fsup"].notna()
    x, y = df["qb"][ok].values, df["fsup"][ok].values
    out["p2s"] = {"full": _slope_ci(x, y, 30, rng),
                  "rows": [float(y[x == i].mean()) if (x == i).any() else None for i in range(1, 6)],
                  "per": {n: _slope(_sub(df[ok], a, b)["qb"].values, _sub(df[ok], a, b)["fsup"].values)
                          for n, a, b in PERIODS}}
    ok = df["qb"].notna() & df["q_all"].notna() & df["fsup"].notna() & df["f30"].notna()
    s = df[ok]
    out["part_s2p"] = _partial(s["f30"].values, s["q_all"].values, s["qb"].values, 30, rng)
    out["part_p2s"] = _partial(s["fsup"].values, s["qb"].values, s["q_all"].values, 30, rng)
    out["part_per"] = {}
    for n, a, b in PERIODS:
        t = _sub(s, a, b)
        if len(t) > 60:
            X = np.column_stack([np.ones(len(t)), t["q_all"].values, t["qb"].values])
            out["part_per"][n] = (float(np.linalg.lstsq(X, t["f30"].values, rcond=None)[0][1]),
                                  float(np.linalg.lstsq(X[:, [0, 2, 1]], t["fsup"].values, rcond=None)[0][1]))
    # mint bursts vs price twin
    base = df[df["burst"].notna()]
    ev_s = _events(base["burst"].values)
    ev_p = _events(base["b7top"].values)
    out["burst"] = {}
    for key, ev in (("supply", ev_s), ("price", ev_p)):
        r = {"n": int(ev.sum())}
        for h in (30, 90):
            f = base[f"f{h}"].values
            m = ev & np.isfinite(f)
            r[h] = float(f[m].mean() - np.nanmean(f)) if m.any() else None
            r[f"up{h}"] = float((f[m] > np.nanmedian(f)).mean()) if m.any() else None
            r[f"n{h}"] = int(m.sum())
        r["per"] = {}
        for n, a, b in PERIODS:
            mm = (base.index >= pd.Timestamp(a)) & (base.index <= pd.Timestamp(b))
            f = base["f90"].values
            m = ev & mm & np.isfinite(f)
            r["per"][n] = (float(f[m].mean() - np.nanmean(f[mm])) if m.any() else None, int(m.sum()))
        out["burst"][key] = r
    out["burst_dates"] = [_d(t) for t in base.index[ev_s]]
    # S2's own effect: burst vs no burst among days below ma50
    out["s2eff"] = {}
    below = df[(df["ma"] == 0) & df["burst"].notna()]
    for h in (30, 90):
        f = below[f"f{h}"].values
        m = np.isfinite(f)
        out["s2eff"][h] = _diff_ci(below["burst"].values[m] > 0.5, f[m], BLOCK[h], rng)
    out["s2per"] = {}
    for n, a, b in PERIODS:
        t = _sub(below, a, b)
        f = t["f30"].values
        m = np.isfinite(f)
        fl = t["burst"].values[m] > 0.5
        out["s2per"][n] = (float(f[m][fl].mean() - f[m][~fl].mean()) if fl.any() and (~fl).any() else None,
                           int(fl.sum()), _runs(fl))
    # burns
    g = df[df["grow"].notna()]
    out["burn"] = {}
    for h in (30, 90):
        f = g[f"f{h}"].values
        m = np.isfinite(f)
        out["burn"][h] = _diff_ci(g["grow"].values[m] < 0.5, f[m], BLOCK[h], rng)
    out["burn_days"], out["burn_runs"] = int((g["grow"] < 0.5).sum()), _runs((g["grow"] < 0.5).values)
    out["burn_per"] = {}
    for n, a, b in PERIODS:
        t = _sub(g, a, b)
        f = t["f90"].values
        m = np.isfinite(f)
        fl = t["grow"].values[m] < 0.5
        out["burn_per"][n] = (float(f[m][fl].mean() - f[m][~fl].mean()) if fl.any() and (~fl).any() else None,
                              int(fl.sum()))
    ma_on = g["ma"] == 1
    out["burn_in_ma"] = float(((g["grow"] < 0.5) & ma_on).sum() / max(1, ma_on.sum()))
    return out


# ---------- look-ahead ----------

def _lookahead(close: pd.Series, g: pd.Series, clock: _Clock) -> tuple[int, int]:
    base = _states(close, _sig(g, quint=False))
    f0 = clock.flags(close.index, base)
    rng = np.random.default_rng(11)
    cuts = bad = 0
    for cut in CUTS:
        t = pd.Timestamp(cut)
        c2 = close.copy()
        c2[c2.index > t] *= np.exp(rng.normal(0, 0.2, int((c2.index > t).sum())))
        g2 = g.copy()
        g2[g2.index >= t] = rng.normal(0, 0.02, int((g2.index >= t).sum()))
        alt = _states(c2, _sig(g2, quint=False))
        m = close.index <= t
        for k in ("ma50", "s1", "s2"):
            cuts += 1
            bad += int(not np.array_equal(np.nan_to_num(base[k][m], nan=-1), np.nan_to_num(alt[k][m], nan=-1)))
        f1 = clock.flags(close.index, alt)
        j = int(clock.m.dates.searchsorted(t + DAY, side="right"))
        for k in ("grow", "burst"):
            cuts += 1
            bad += int(not np.array_equal(f0[k][:j], f1[k][:j]))
    return cuts, bad


# ---------- the computation ----------

def _compute() -> dict:
    t0 = time.time()
    raw, problems = _load()
    g, jumps = _growth(raw)
    close = te._daily_close()
    rng = np.random.default_rng(SEED + 31)
    sigs = {c: _sig(g[c]) for c in g.columns}
    st = _states(close, sigs["all"])
    pos = {k: st[k] for k in KEYS}

    who = _who_first(close, g, rng)
    df = _frame(close, sigs, st)
    quint = _quint_tests(df, rng)
    mech = _mechanisms(df, rng)

    # fair exam
    first = _first_known(close.index, pos)
    clock = _Clock(first, msc.DAILY_STARTS[1])
    if clock.problem:
        problems.insert(0, clock.problem)
    fl = clock.flags(close.index, st)
    others, opos = {}, {}
    for sym, key in OTHER:
        try:
            c = ta._crypto_close(sym)
        except Exception as exc:  # noqa: BLE001 - one coin failing must not stop the others
            problems.append(f"{key} 現貨日線這次取不到（{type(exc).__name__}），沒有算進去。")
            continue
        p = _other_pos(c, close.index, st)
        fa = _first_known(c.index, p)
        if fa < others_last():
            others[key], opos[key] = (c, fa), p
    exam = _exam(close, pos, clock, fl, others, opos, first)
    t_exam = time.time() - t0

    # placebo: total-supply growth rolled circularly, ranking score without the ordinary-person check
    real_t = _exam(close, pos, clock, fl, others, opos, first, TESTED, person=False)
    real_text = {k: real_t[k]["low"] for k in TESTED}
    prng = np.random.default_rng(SEED + 177)
    gv = g["all"]
    plac = {k: [] for k in TESTED}
    for _ in range(PLACEBO):
        s = int(prng.integers(PL_GAP, len(gv) - PL_GAP))
        gs = pd.Series(np.roll(gv.values, s), index=gv.index)
        sp = _states(close, _sig(gs, quint=False))
        pp = {k: sp[k] for k in KEYS}
        flp = clock.flags(close.index, sp)
        opp = {a: _other_pos(c, close.index, sp) for a, (c, _fa) in others.items()}
        e = _exam(close, pp, clock, flp, others, opp, first, TESTED, person=False)
        for k in TESTED:
            plac[k].append(e[k]["low"])
    placebo = {k: float(np.mean(np.array(v) >= real_text[k] - 1e-12)) for k, v in plac.items()}
    placebo_med = {k: float(np.median(v)) for k, v in plac.items()}

    first_i = int(close.index.searchsorted(first))
    held = {k: float(np.nanmean(np.nan_to_num(pos[k][first_i:]))) for k in KEYS}
    diff_days = {k: int(np.nansum(np.nan_to_num(pos[k][first_i:]) != np.nan_to_num(pos["ma50"][first_i:])))
                 for k in TESTED}
    look = _lookahead(close, g["all"], clock)
    return {"raw": raw, "g": g, "jumps": jumps, "close": close, "sigs": sigs, "st": st, "pos": pos, "who": who,
            "quint": quint, "mech": mech, "exam": exam, "first": first, "clock": clock, "others": list(others),
            "real_text": real_text, "placebo": placebo, "placebo_med": placebo_med, "held": held,
            "diff_days": diff_days, "look": look, "problems": problems, "t_exam": t_exam, "t_all": time.time() - t0}


# ---------- the study ----------

def _sign(v) -> int:
    return 0 if not _ok(v) or v == 0 else (1 if v > 0 else -1)


def study() -> dict:
    r = _compute()
    close, who, q, mech, ex = r["close"], r["who"], r["quint"], r["mech"], r["exam"]
    pnames = [p[0] for p in PERIODS]
    m = ex["ma50"]

    # ----- pass test -----
    s1_per = [q["all"][30][p]["slope"] for p in pnames]
    s2_per = [mech["s2per"][p][0] for p in pnames]
    sign_ok = {"s1": all(_sign(v) == 1 for v in s1_per), "s2": all(_sign(v) == 1 for v in s2_per)}
    rule_ok = {}
    for k in TESTED:
        x = ex[k]
        rule_ok[k] = {"sign": sign_ok[k], "score": x["low"] >= m["low"] - 1e-12,
                      "deep": x["c_deep_med"] >= m["c_deep_med"] - 1e-12, "placebo": r["placebo"][k] <= 0.10}
        rule_ok[k]["all"] = all(rule_ok[k].values())
    passed = [k for k in TESTED if rule_ok[k]["all"]]

    # ----- table 1: who moves first -----
    rows1 = []
    for c in ("all", "usdt", "usdc"):
        w = who[c]
        ld, fw, gs, gb = w["lead"], w["follow"], w["g_sb"], w["g_bs"]
        rows1.append([f"{CN[c]}（{w['weeks']} 週，{_d(w['from'])} 起）",
                      f"{_c(ld['est'])} {_ci(ld['lo'], ld['hi'], _c)}｜p {ld['p']:.2f}",
                      f"{_c(fw['est'])} {_ci(fw['lo'], fw['hi'], _c)}｜p {fw['p']:.2f}",
                      "／".join(f"{_c(w['per'][p][0])}" for p in pnames),
                      "／".join(f"{_c(w['per'][p][1])}" for p in pnames),
                      f"多解釋 {gs['add'] * 100:.1f}%（p {gs['p']:.2f}）｜樣本外 {_p(gs['oos'])}",
                      f"多解釋 {gb['add'] * 100:.1f}%（p {gb['p']:.2f}）｜樣本外 {_p(gb['oos'])}"])

    # ----- table 2: cross-correlation by lag -----
    rows2 = []
    for k in range(-8, 9):
        label = ("同一週" if k == 0 else (f"供給先、比特幣晚 {k} 週" if k > 0 else f"比特幣先、供給晚 {-k} 週"))
        rows2.append([label] + [_c(who[c]["xc"][k]) for c in ("all", "usdt", "usdc")])
    rows2.append(["打亂後的 95% 範圍（絕對值）"] + [f"±{who[c]['xc']['band']:.2f}" for c in ("all", "usdt", "usdc")])
    for k in (-30, -7, -1, 0, 1, 7, 30):
        label = ("每日：同一天" if k == 0 else (f"每日：供給先、比特幣晚 {k} 天" if k > 0 else f"每日：比特幣先、供給晚 {-k} 天"))
        rows2.append([label] + [_c(who[c]["daily"][k]) for c in ("all", "usdt", "usdc")])
    rows2.append(["每日：這天供給成長 vs 之後 30 天比特幣"] + [_c(who[c]["daily30"][0]) for c in ("all", "usdt", "usdc")])
    rows2.append(["每日：這天比特幣報酬 vs 之後 30 天供給成長"] + [_c(who[c]["daily30"][1]) for c in ("all", "usdt", "usdc")])

    # ----- table 3: quintiles -----
    rows3 = []
    for i in range(5):
        rows3.append([f"第 {i + 1} 組（{'成長最慢' if i == 0 else '成長最快' if i == 4 else '中間'}）",
                      f"{q['all'][30]['days'][i]:,}", _p(q["all"][30]["rows"][i]), _p(q["all"][90]["rows"][i])]
                     + ["／".join(_p(q["all"][90][p]["rows"][i]) for p in pnames)])
    for c in ("all", "usdt", "usdc"):
        f30, f90 = q[c][30]["full"], q[c][90]["full"]
        rows3.append([f"斜率（每高一組）：{CN[c]}", f"{f30[3]:,}",
                      f"{_pt(f30[0])} {_ci(f30[1], f30[2], _pt)}", f"{_pt(f90[0])} {_ci(f90[1], f90[2], _pt)}",
                      "30 天 " + "／".join(_pt(q[c][30][p]["slope"]) for p in pnames)
                      + "；90 天 " + "／".join(_pt(q[c][90][p]["slope"]) for p in pnames)])

    # ----- table 4: mechanisms -----
    p2s, ps, pp = mech["p2s"], mech["part_s2p"], mech["part_p2s"]
    bs, bp = mech["burst"]["supply"], mech["burst"]["price"]
    s2e, bu = mech["s2eff"], mech["burn"]
    rows4 = [
        ["價格先動 → 供給跟上：過去 30 天比特幣漲跌（一年位置分 5 組）每高一組，之後 30 天總供給多成長",
         f"{_pt(p2s['full'][0])} {_ci(p2s['full'][1], p2s['full'][2], _pt)}",
         "／".join(_pt(p2s["per"][p]) for p in pnames),
         "第 1→5 組：" + "、".join(_p(v) for v in p2s["rows"])],
        ["供給 → 價格，扣掉過去價格的影響：供給成長每高一組，之後 30 天比特幣",
         f"{_pt(ps[0])} {_ci(ps[1], ps[2], _pt)}",
         "／".join(_pt(mech["part_per"].get(p, (None, None))[0]) for p in pnames), f"{ps[3]:,} 天"],
        ["價格 → 供給，扣掉供給自己的慣性：過去漲跌每高一組，之後 30 天總供給",
         f"{_pt(pp[0])} {_ci(pp[1], pp[2], _pt)}",
         "／".join(_pt(mech["part_per"].get(p, (None, None))[1]) for p in pnames), f"{pp[3]:,} 天"],
        [f"增發爆量（7 天成長在一年最高 20%，{DECLUSTER} 天內只算第一次）之後 30／90 天，比所有日子平均多",
         f"{_pt(bs[30])}／{_pt(bs[90])}（{bs['n']} 次）",
         "90 天：" + "／".join(f"{_pt(bs['per'][p][0])}（{bs['per'][p][1]}）" for p in pnames),
         f"90 天高於中位數的比例 {_share(bs['up90'])}"],
        ["對照（只看價格）：比特幣 7 天漲幅在一年最高 20% 之後 30／90 天，比所有日子平均多",
         f"{_pt(bp[30])}／{_pt(bp[90])}（{bp['n']} 次）",
         "90 天：" + "／".join(f"{_pt(bp['per'][p][0])}（{bp['per'][p][1]}）" for p in pnames),
         f"90 天高於中位數的比例 {_share(bp['up90'])}"],
        ["S2 的依據：價格在 50 日均線下的日子，爆量那幾天之後 30／90 天比沒爆量的多",
         f"{_pt(s2e[30][0])} {_ci(s2e[30][1], s2e[30][2], _pt)}／{_pt(s2e[90][0])}",
         "30 天：" + "／".join(f"{_pt(mech['s2per'][p][0])}（{mech['s2per'][p][2]} 段）" for p in pnames),
         f"{s2e[30][3]:,} 天"],
        ["縮水（30 天供給減少）的日子之後 30／90 天，比供給增加的日子多",
         f"{_pt(bu[30][0])} {_ci(bu[30][1], bu[30][2], _pt)}／{_pt(bu[90][0])} {_ci(bu[90][1], bu[90][2], _pt)}",
         "90 天：" + "／".join(f"{_pt(mech['burn_per'][p][0])}（{mech['burn_per'][p][1]} 天）" for p in pnames),
         f"{mech['burn_days']:,} 天、{mech['burn_runs']} 段；50 日均線持有的日子裡 {_share(mech['burn_in_ma'])} 在縮水"],
    ]

    # ----- table 5: fair exam -----
    order = sorted(KEYS, key=lambda k: -ex[k]["low"])
    rows5 = []
    for k in KEYS:
        x = ex[k]
        rows5.append([NAMES[k], _of(x["d_n"], x["d_beat"]), f"{_share(x['checks'][1])}｜{cr._pp(x['c_edge'])}",
                      f"{_share(x['checks'][2])}｜{cr._pp(x['p_edge'])}", _of(x["o_n"], x["o_beat"]),
                      f"{_share(x['low'])}（第 {order.index(k) + 1}）｜{cr._pct(x['c_deep_med'])}",
                      f"{_share(r['held'][k])}" + ("" if k == "ma50" else f"｜和均線不同 {r['diff_days'][k]:,} 天"),
                      "—（基準）" if k == "ma50" else _share(r["placebo"][k])])
    rows5.append(["對照：一直抱著不動", "—", "—", "—", "—", f"—｜{cr._pct(float(np.median(r['clock'].hd)))}", "100%", "—"])

    # ----- table 6: pass test -----
    def yes(v: bool) -> str:
        return "過" if v else "沒過"

    rows6 = []
    for k, per, what in (("s1", s1_per, "30 天成長分組的斜率（30 天報酬）"),
                         ("s2", s2_per, "均線下爆量減沒爆量（30 天報酬）")):
        x = ex[k]
        rows6.append([NAMES[k], f"{what}：" + "／".join(_pt(v) for v in per) + f"（{yes(rule_ok[k]['sign'])}）",
                      f"{_share(x['low'])} vs {_share(m['low'])}（{yes(rule_ok[k]['score'])}）",
                      f"{cr._pct(x['c_deep_med'])} vs {cr._pct(m['c_deep_med'])}（{yes(rule_ok[k]['deep'])}）",
                      f"{_share(r['placebo'][k])}（{yes(rule_ok[k]['placebo'])}）",
                      "通過" if rule_ok[k]["all"] else "沒有通過"])

    n_d, n_c = ex["n_daily"], m["c_n"]
    tables = [
        {"caption": ("一、誰先動？每週（連續 7 天）的穩定幣供給成長（只用已公布的，晚一天）和比特幣報酬。「供給領先」＝這週供給成長和之後 4 週比特幣報酬的相關；"
                     "「價格領先」＝這週比特幣報酬和之後 4 週供給成長的相關。方括號是 90% 區間（8 週一段重抽 1,000 次），p＝把整串循環平移 200 次後一樣高或更高的比例。"
                     "格蘭傑式檢定：用過去 4 週的自己再加過去 4 週的對方，多解釋了多少變動（p＝把對方換成重抽的序列 1,000 次後一樣多的比例）；"
                     f"樣本外＝每週只用之前的資料重估（至少 {MIN_TRAIN} 週），加上對方之後預測誤差少了多少（正數才代表真的有幫助）。"),
         "columns": ["哪種穩定幣", "供給領先：相關［90% 區間］｜p", "價格領先：相關［90% 區間］｜p",
                     "供給領先：" + "／".join(pnames), "價格領先：" + "／".join(pnames),
                     "供給 → 比特幣（格蘭傑）", "比特幣 → 供給（格蘭傑）"],
         "rows": rows1},
        {"caption": ("二、逐週錯開的相關（這週供給成長 vs 錯開 k 週的比特幣報酬）。正數的 k 是供給在前，負數是比特幣在前；「打亂後」那一列是隨機平移後"
                     "95% 會落在裡面的範圍。下面幾列是同樣的事改用每日資料（每日的雜訊大，相關自然比較小）。"),
         "columns": ["錯開幾週", CN["all"], CN["usdt"], CN["usdc"]], "rows": rows2},
        {"caption": ("三、把 30 天穩定幣總供給成長放進它過去一年的位置分 5 組（只用當時已知的資料），之後 30／90 天比特幣平均漲跌。"
                     "斜率＝每高一組多（少）幾個百分點，方括號 90% 區間（30 天用 30 天一段、90 天用 120 天一段重抽）。報酬天天重疊，真正獨立的樣本少很多。"
                     "2019–2021 只有 2020 年中以後才有分組（要先累積一年）。"),
         "columns": ["分組", "天數", "之後 30 天平均", "之後 90 天平均", "之後 90 天：" + "／".join(pnames)],
         "rows": rows3},
        {"caption": ("四、機制：是供給推價格，還是價格帶供給？爆量和縮水各代表什麼？「點」是百分點；扣掉對方影響的那兩列是把兩個分組一起放進迴歸。"
                     "增發爆量和只看價格的對照用同樣的「一年最高 20%、30 天內只算第一次」，用來分辨是穩定幣本身的資訊，還是只是價格動能。"),
         "columns": ["看什麼", "全期（90% 區間）", "三段：" + "／".join(pnames), "補充"],
         "rows": rows4},
        {"caption": (f"五、規則和 50 日均線同樣條件考（main_signal_check 的公平考法，但只能從穩定幣訊號有值的月份開始）：比特幣 {n_d} 個起算月"
                     f"（{_d(r['first'])} 到 2023-12，UTC 0 點收盤）、{n_c:,} 格起算月×收盤時間（照課本做，成本 {COST * 100:.2f}%）、"
                     f"同樣格數照一般人方式做（10 小時後才做、成本 {P_COST * 100:.2f}%、一成進場漏掉、每個收盤時間 {SIMS} 次），"
                     f"以及其他幣（{'、'.join(r['others'])}，用各自的 50 日均線，加上同一個穩定幣訊號當大盤濾網）。前四項是報酬贏過同一天起一直抱著的比例"
                     "（後面是每年多賺的中位數）；排名分數＝四項裡最低的；最深虧損＝起算月×收盤時間各格的中位數。"
                     f"安慰劑＝把穩定幣總供給的每日成長整串循環平移 {PLACEBO} 次再算排名分數（為了時間不含一般人那一項，真的規則也用同樣方法算），一樣好或更好的比例。"),
         "columns": ["規則", "比特幣起算月：贏的", f"{n_c:,} 格照課本：贏的｜多賺", "一般人做法：贏的｜多賺", "其他幣起算月：贏的",
                     "排名分數（名次）｜最深虧損中位數", "持有天數比例", "安慰劑：打亂後一樣好的比例"],
         "rows": rows5},
        {"caption": ("六、事先寫下的通過條件逐條對：預測的方向（為正）在三段期間都成立；排名分數不低於 50 日均線；最深虧損中位數不比 50 日均線深；"
                     "安慰劑比例 ≤ 10%。四條都過才算通過。"),
         "columns": ["規則", "方向（三段）", "排名分數 vs 50 日均線", "最深虧損中位數 vs 50 日均線", "安慰劑比例", "結論"],
         "rows": rows6},
    ]

    # ----- today -----
    raw, sg, st, pos = r["raw"], r["sigs"]["all"], r["st"], r["pos"]
    t = close.index[-1]
    use = t - DAY
    last_raw = raw.index[-1]

    def val(s: pd.Series, d):
        v = s.get(d) if d in s.index else None
        return float(v) if _ok(v) else None

    def word(v) -> str:
        return "—" if not _ok(v) else ("持有" if v >= 0.5 else "現金")

    g7, g30, qq = val(sg["g7"], use), val(sg["g30"], use), val(sg["q30"], use)
    now = [
        {"label": f"最後一根收完的日線（{_d(t)}，UTC 0 點收盤）", "value": f"比特幣 {float(close.iloc[-1]):,.0f} 美元"},
        {"label": f"DefiLlama 最新一天（{_d(last_raw)}）的供給",
         "value": "、".join(f"{CN[c]} {raw[c].iloc[-1] / 1e8:,.0f} 億美元" for c in ("all", "usdt", "usdc"))},
        {"label": f"規則用的總供給成長（資料到 {_d(use)}）",
         "value": (f"7 天 {_p(math.expm1(g7))}、30 天 {_p(math.expm1(g30))}" if _ok(g7) and _ok(g30) else "—")},
        {"label": "30 天成長在過去一年的分組（1 最慢、5 最快）", "value": f"第 {qq:.0f} 組" if _ok(qq) else "—"},
        {"label": "增發爆量（7 天成長在一年最高 20%）", "value": "—" if not _ok(st["burst"][-1]) else
         ("是" if st["burst"][-1] > 0.5 else "不是")},
        {"label": "30 天供給是增加還是縮水", "value": "—" if not _ok(st["grow"][-1]) else
         ("增加" if st["grow"][-1] > 0.5 else "縮水")},
    ]
    for k in KEYS:
        now.append({"label": NAMES[k], "value": word(pos[k][-1])})
    now.append({"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"})

    # ----- verdict -----
    wa = who["all"]
    lead, follow = wa["lead"], wa["follow"]
    if _ok(follow["est"]) and _ok(lead["est"]):
        if follow["lo"] > 0 and not lead["lo"] > 0:
            who_txt = "比較像價格先動、穩定幣供給跟上，不是供給帶動價格"
        elif lead["lo"] > 0 and not follow["lo"] > 0:
            who_txt = "比較像供給先動、價格跟上"
        elif lead["lo"] > 0 and follow["lo"] > 0:
            who_txt = "兩個方向都有關係，互相拉動"
        else:
            who_txt = "兩個方向都看不出可靠的先後"
    else:
        who_txt = "算不出先後"
    f30 = q["all"][30]["full"]
    first_s = (f"結論：{who_txt}。每週資料（{wa['weeks']} 週）裡，這週供給成長和之後 4 週比特幣報酬的相關是 {_c(lead['est'])}"
               f"（90% 區間 {_c(lead['lo'])} 到 {_c(lead['hi'])}），這週比特幣報酬和之後 4 週供給成長的相關是 {_c(follow['est'])}"
               f"（{_c(follow['lo'])} 到 {_c(follow['hi'])}）；樣本外加入供給預測比特幣 {_p(wa['g_sb']['oos'])}、"
               f"加入比特幣預測供給 {_p(wa['g_bs']['oos'])}"
               + ("（負數＝每週重估後反而猜得更差，這個先後只在回頭看整段時才明顯）。"
                  if _ok(wa["g_bs"]["oos"]) and wa["g_bs"]["oos"] < 0 else "。"))
    second = (f"30 天供給成長每高一組，之後 30 天比特幣平均 {_pt(f30[0])}（90% 區間 {_pt(f30[1])} 到 {_pt(f30[2])}；"
              f"三段 {'／'.join(_pt(v) for v in s1_per)}）；增發爆量之後 90 天比平常 {_pt(bs[90])}，只看價格的同類對照是 {_pt(bp[90])}；"
              f"供給縮水的日子之後 90 天比增加時 {_pt(bu[90][0])}。")
    third = (f"排名分數 S1 {_share(ex['s1']['low'])}、S2 {_share(ex['s2']['low'])}，50 日均線 {_share(m['low'])}"
             f"（同樣 {n_d} 個起算月；安慰劑比例 S1 {_share(r['placebo']['s1'])}、S2 {_share(r['placebo']['s2'])}）；"
             f"事先定的通過條件{'有規則過了：' + '、'.join(SHORT[k] for k in passed) if passed else '沒有一條過'}。")
    verdict = first_s + second + third

    # ----- notes -----
    cuts, bad = r["look"]
    jt = "；".join(f"{CN[c]} " + ("、".join(v) if v else "沒有") for c, v in r["jumps"].items())
    notes = [
        ("事先寫下的設計（照原文）：(1) lead/lag cross-correlation of daily/weekly supply growth vs BTC returns (both "
         "directions) and Granger-style test with block bootstrap; (2) forward 30/90-day BTC returns by trailing-year quintile "
         "of 30-day supply growth, split 2019-21 / 2022-23 / 2024-now; (3) rules on top of ma50: S1 = ma50 AND 30-day supply "
         "growth > 0; S2 = ma50 but enter early when supply-growth burst (top 20%) while price below ma50; fair exam "
         "(monthly starts × 24 clocks where data allows, ordinary-person execution 0.25% & 10% missed entries, other coins "
         "ETH/SOL/BNB/XRP, ranking score = worst of four \"beats holding\" shares, placebo = circular shift ≥180 days, 200 "
         "draws, median deepest loss). Pass test: predicted sign in all three periods AND ranking score ≥ ma50's AND median "
         "deepest loss no worse AND placebo ≤ 10%."),
        (f"通過條件：{'、'.join(SHORT[k] for k in passed) + ' 通過' if passed else '沒有一條規則通過'}。安慰劑用不含一般人那一項的排名分數"
         f"（真的規則同樣算法：S1 {_share(r['real_text']['s1'])}、S2 {_share(r['real_text']['s2'])}；打亂後的中位數 S1 "
         f"{_share(r['placebo_med']['s1'])}、S2 {_share(r['placebo_med']['s2'])}）。持有天數比例從第一個起算月算起。"),
        ("資料：DefiLlama 穩定幣 API（免金鑰），全部穩定幣＝各種法幣掛鉤的流通量換成美元加總，USDT、USDC 各自一條。"
         "DefiLlama 在 2019 年中以前收錄不完整（例如 2018 年 USDT 只算到幾千萬美元，實際是幾十億，Omni 鏈上的沒有收），"
         f"而且後來才加進的鏈會在某一天一次跳上去，所以成長只從 {START} 算起，單日變動超過 ±{JUMP * 100:.0f}% 的當成收錄跳動、算成 0："
         f"{jt}。DefiLlama 也會在收錄新幣、新鏈時回頭改歷史，這裡用的是今天看到的版本，不一定是當時即時看得到的數字。"),
        ("不偷看未來：某一天的供給要等那天（UTC）結束才定案，這裡再保守一天，從下一根日線收盤才用；換收盤時間時，落在 UTC 日期 E 的收盤只用"
         f"E−1 那天 UTC 0 點收盤時決定的狀態（供給到 E−2）。檢查：把 {'、'.join(CUTS)} 之後的價格和供給換成亂數再算，{cuts} 組狀態裡"
         + ("全部沒變。" if bad == 0 else f"有 {bad} 組變了，這次結果不可信。")),
        ("機制怎麼讀：如果「比特幣先漲、供給才跟上」的相關明顯比反方向強，代表供給大多是在追價（像 ETF 資金流在 etf_loop 研究裡的樣子），"
         "拿它來預測價格就會慢半拍；增發爆量如果和「只看價格的漲幅爆量」之後的表現差不多，代表它帶的資訊大多就是價格動能本身。"
         "穩定幣增加也可能只是放著領利息、拿去做套利或衍生品保證金，不一定拿去買比特幣。"),
        (f"增發爆量（30 天內只算第一次）的日子：{'、'.join(mech['burst_dates'][-12:]) if mech['burst_dates'] else '沒有'}"
         + ("（只列最近 12 次）" if len(mech["burst_dates"]) > 12 else "") + "。"),
        ("先前的 flow_filters 研究測過另一種用法（7 天成長高於一年中位數才持有，疊在 20 日新高／10 日新低規則上），這裡是不同的規則和考法，數字不能直接比。"),
        ("其他幣用的是比特幣大盤的穩定幣訊號（同一條總供給），不是它們自己的資料；其他幣是今天還在的大幣，有倖存者偏誤。"
         "相鄰起算日共用大部分的歷史，24 個收盤時間只差幾個小時，「幾個裡幾個」不是互相獨立的證據；"
         "穩定幣訊號只從 2020 年年中才有，起算月比 main_signal_check 少很多，50 日均線也只在同樣的起算月上比。"),
        (f"計算時間：公平考法 {r['t_exam']:.0f} 秒、全部 {r['t_all']:.0f} 秒。"),
        "過去的結果不保證未來會一樣。這裡只比較穩定幣供給和比特幣過去的關係、規則現在停在什麼狀態，不是買賣建議。",
    ]
    notes += r["problems"]

    return {
        "id": "stablecoin_supply",
        "title": "穩定幣供給成長：帶動比特幣，還是跟著價格走？",
        "question": ("穩定幣（USDT、USDC、全部加總）變多，是新錢進場、之後比特幣會漲，還是比特幣先漲、大家才換穩定幣進來追？"
                     "大量增發之後會不會比較強、供給縮水是不是走弱的訊號？拿它來加在 50 日均線上，能不能贏過均線和隨機對照？"),
        "asof": _d(t),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "DefiLlama 穩定幣總供給（stablecoincharts/all，免金鑰）", "url": LLAMA},
            {"name": "DefiLlama USDT（stablecoin=1）", "url": LLAMA + "?stablecoin=1"},
            {"name": "DefiLlama USDC（stablecoin=2）", "url": LLAMA + "?stablecoin=2"},
            {"name": "DefiLlama 穩定幣頁面", "url": LLAMA_PAGE},
            {"name": "KuCoin 轉述 CryptoQuant：穩定幣供給破 1,600 億美元，說供給增加領先比特幣漲勢", "url": "https://www.kucoin.com/news/flash/stablecoin-supply-hits-160-billion-seen-as-indicator-for-bitcoin-price"},
            {"name": "Cointelegraph：USDT 增發與比特幣行情的報導", "url": "https://cointelegraph.com/news/usdt-mints-bitcoin-price"},
            {"name": "幣安現貨日線與 1 小時線（BTC、ETH、SOL、BNB、XRP 對 USDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
