"""Historical study "power_law": the bitcoin power law (log price is a straight line in log of the days since the
genesis block, 2009-01-03), drawn only with the data that existed at the time. How did prices go in the following
year when they sat far above or below that line, and do rules built on it beat the 50-day average on the fair grid
of main_signal_check?

Pre-registered design (written before any result was seen; not changed afterwards):

Fit (point-in-time): on each day t, fit log10(price) = a + b*log10(days since 2009-01-03) by OLS on all daily closes
up to t-1 (expanding window; start using it from 2014-01-01, when there are >= 3 years of prices). Residual
z_t = (log price - fit) / std of past residuals. Also record the published constants (e.g. b ~ 5.8 as commonly
quoted) only as a description of look-ahead.
Rules (daily close, act next close; costs 0.10/0.30%):
P1 valuation exposure: weight = clip(0.5 - 0.25*z, 0, 1) (fully in when >= 2 std below, out when >= 2 std above;
   50% at fair).
P2 band rule: hold when z < +1, cash when z > +2, keep previous state in between.
P3 P2 combined with ma50 (hold only when both say hold).
P4 accumulation: hold when z < 0 regardless of ma50, else follow ma50 (buy cheap, trend when rich).
Compare with holding and ma50 on main_signal_check's fair grid (start dates from 2015 as it does; 1,728 start x clock
cells; ordinary-person execution); other assets: none have a comparable long history - say so; use ETH with genesis
2015-07-30 as a check, labelled exploratory.
Also: forward 1-year BTC return by z bucket (<-1, -1..0, 0..1, 1..2, >2) point-in-time; how stable the fitted slope b
is over time (b at each year end); placebo: shift the time origin (genesis +/- 1-3 years) and refit - do results
depend on the origin?
Pass test (stated now): a rule scores >= ma50 on the ranking score AND has a shallower median deepest loss AND the
time-origin placebo does not change the conclusion.
Today: fitted fair value, the bands, z, and each rule's state - states only.

Fixed details chosen while writing the code (before running it):
- Ranking score = the lowest of three "return beats holding" shares on bitcoin: the 108 monthly start dates
  2015-01..2023-12 (UTC 00:00 daily closes, textbook, 0.10%), the 72 x 24 = 1,728 start x clock cells 2018-01..2023-12
  (textbook, 0.10%), and the same cells done the ordinary-person way (trade 10 hours after the close, 0.30%, 10% of
  entries missed, 200 simulations per clock, fixed seed). main_signal_check also has an "other assets" check; no
  other asset has a comparable price history, so here every rule (ma50 included) is scored on the three bitcoin
  checks only. "Median deepest loss" = median over the 1,728 textbook cells.
- Before the first day with a z value, P2 starts in the "hold" state.
- Placebo origins: 2009-01-03 shifted by -3, -2, -1, +1, +2, +3 years. Only closes at least 180 days after the
  shifted origin enter the fit (log of a few days would dominate the line). Every placebo origin goes through the
  same three checks; "the placebo does not change the conclusion" = the rule passes the first two parts of the test
  at every one of the seven origins.
- Fit residual std = sqrt(sum of squared residuals / (n - 2)) of the fit through t-1.

No look-ahead: z_t uses closes up to t-1 for the line and the close of t itself for the distance; a rule decided at
the close of t is held from the next day (or 10 hours later for the ordinary person). Only public, key-less data
through the dashboard's own helpers (blockchain.info daily prices before 2017, Binance spot afterwards).
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
from dashboard.studies2 import trend_assets as ta
from dashboard.studies2 import trend_ensemble as te

GENESIS = pd.Timestamp("2009-01-03")
ETH_GENESIS = pd.Timestamp("2015-07-30")
FIT_FROM = pd.Timestamp("2014-01-01")
MIN_POINTS = 365
MIN_AGE = 180  # placebo origins: closes younger than this (in days after the origin) are left out of the fit
SHIFTS = (0, -3, -2, -1, 1, 2, 3)  # years
COST = 0.001
P_COST, P_MISS = 0.003, 0.10
SIMS = 200
SEED = 20090103
YEAR = 365
DAILY_STARTS = msc.DAILY_STARTS
CLOCK_STARTS = msc.CLOCK_STARTS
PUBLISHED_B, PUBLISHED_A = 5.84, -17.01  # the commonly quoted constants (AHR999's 2018 fit; Santostasi quotes ~5.8)
RULES = ("ma50", "p1", "p2", "p3", "p4")
PL_RULES = ("p1", "p2", "p3", "p4")
NAMES = {"hold": "一直抱著不動", "ma50": "收盤高於 50 日均線才持有（對照）",
         "p1": "P1 依偏離加減碼：持有 0.5 − 0.25×z（夾在 0 到 1）",
         "p2": "P2 上下緣：z 低於 +1 持有、高於 +2 換現金、中間維持",
         "p3": "P3 = P2 而且收盤高於 50 日均線", "p4": "P4 便宜（z < 0）就持有，否則照 50 日均線"}
SHORT = {"ma50": "50 日均線", "p1": "P1 加減碼", "p2": "P2 上下緣", "p3": "P3 上下緣＋均線", "p4": "P4 便宜就抱、否則均線"}
BUCKETS = (("z < −1（比趨勢線便宜一個標準差以上）", -np.inf, -1.0), ("−1 ≤ z < 0", -1.0, 0.0), ("0 ≤ z < 1", 0.0, 1.0),
           ("1 ≤ z < 2", 1.0, 2.0), ("z ≥ 2（比趨勢線貴兩個標準差以上）", 2.0, np.inf))

_pct, _pp, _d, _ok = cr._pct, cr._pp, cr._d, cr._ok


def _share(x) -> str:
    return "—" if not _ok(x) else f"{float(x) * 100:.0f}%"


def _money(v) -> str:
    return "—" if not _ok(v) else f"{float(v):,.0f} 美元"


# ---------- the point-in-time fit ----------

def _pit(age: np.ndarray, close: np.ndarray, dates: pd.DatetimeIndex | None, fit_from: pd.Timestamp | None,
         min_age: float = 1.0) -> dict:
    """Expanding OLS of log10(close) on log10(age) using only the closes BEFORE each day.

    Returns arrays a, b, sd (fit through t-1), the fitted log price at t and z_t = (log close_t - fit) / sd.
    z is NaN before fit_from or while fewer than MIN_POINTS closes are in the fit."""
    age = np.asarray(age, dtype=float)
    y = np.log10(np.asarray(close, dtype=float))
    use = np.isfinite(y) & (age >= min_age)
    x = np.where(use, np.log10(np.where(age > 0, age, 1.0)), np.nan)
    u = use.astype(float)
    xv, yv = np.where(use, x, 0.0), np.where(use, y, 0.0)

    def excl(v):  # sum over the earlier days only
        return np.cumsum(v) - v
    n = excl(u)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = excl(xv) / n, excl(yv) / n
        sxx = excl(xv * xv) - n * mx * mx
        sxy = excl(xv * yv) - n * mx * my
        syy = excl(yv * yv) - n * my * my
        b = sxy / sxx
        a = my - b * mx
        ssr = np.maximum(syy - b * sxy, 0.0)
        sd = np.sqrt(ssr / (n - 2))
        fit = a + b * x
        z = (y - fit) / sd
    ok = (n >= MIN_POINTS) & use & np.isfinite(z)
    if dates is not None and fit_from is not None:
        ok &= np.asarray(dates >= fit_from)
    nan = np.full(len(y), np.nan)
    return {"a": np.where(n >= MIN_POINTS, a, nan), "b": np.where(n >= MIN_POINTS, b, nan),
            "sd": np.where(n >= MIN_POINTS, sd, nan), "fit": np.where(ok, fit, nan), "z": np.where(ok, z, nan)}


def _ages(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Days from the origin to the close of each UTC day (a day's close is at the next midnight)."""
    return ((dates - origin) / pd.Timedelta(days=1)).values.astype(float) + 1.0


# ---------- the rules ----------

def _ma50(close: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(close, dtype=float))
    mean = s.rolling(50, min_periods=50).mean()
    return (s > mean).astype(float).where(mean.notna()).values  # exactly trend_ensemble's A50


def _rules(z: np.ndarray, ma: np.ndarray) -> dict[str, np.ndarray]:
    """What each rule wants to hold after each close (NaN while it has no value)."""
    zf = np.asarray(z, dtype=float)
    has = np.isfinite(zf)
    p1 = np.where(has, np.clip(0.5 - 0.25 * np.where(has, zf, 0.0), 0.0, 1.0), np.nan)
    p2 = np.full(len(zf), np.nan)
    state = 1.0
    for i in np.flatnonzero(has):
        if zf[i] < 1.0:
            state = 1.0
        elif zf[i] > 2.0:
            state = 0.0
        p2[i] = state
    p3 = np.where(np.isfinite(p2) & np.isfinite(ma), p2 * ma, np.nan)
    p4 = np.where(has & np.isfinite(ma), np.where(zf < 0, 1.0, ma), np.nan)
    return {"ma50": ma, "p1": p1, "p2": p2, "p3": p3, "p4": p4}


# ---------- the fair grid ----------

def _daily(close: pd.Series, origin: pd.Timestamp, first, last, fit_from: pd.Timestamp, min_age: float,
           keys=RULES, z=None) -> dict:
    """Textbook runs (0.10%) on one daily series from monthly start dates: ret / deep per start and rule."""
    if z is None:
        z = _pit(_ages(close.index, origin), close.values, close.index, fit_from, min_age)["z"]
    target = _rules(z, _ma50(close.values))
    change = close.pct_change().fillna(0.0).values
    valid = np.all([np.isfinite(target[k]) for k in keys], axis=0)
    if not valid.any():
        raise RuntimeError("沒有任何一天每條規則都有值")
    begin = max(close.index[int(np.argmax(valid))] + pd.Timedelta(days=1), pd.Timestamp(first))
    starts, labels = msc._month_starts(close.index, begin, last)
    out = {"starts": labels, "target": target, "z": z}
    hr, hd = msc._suffix(msc._equity(change), starts, YEAR)
    out["hold"] = (hr[:, 0], hd[:, 0])
    first_i = int(starts[0]) if len(starts) else 0
    for k in keys:
        net, _ = te._run(pd.Series(target[k]).shift(1).values, change, COST)
        r, dd = msc._suffix(msc._equity(net), starts, YEAR)
        out[k] = (r[:, 0], dd[:, 0])
        out[f"{k}_share"] = float(np.nanmean(target[k][first_i:]))
    return out


def _pre(b0: pd.Timestamp) -> pd.Series:
    """The daily closes used to warm up every clock (same selection as clock_robustness._prehistory), with dates."""
    from dashboard.collectors import classic

    close = classic._long_close()
    close = pd.Series(close.values.astype(float), index=pd.to_datetime(close.index)).sort_index()
    close = close[(close > 0) & (close.index < b0 - pd.Timedelta(days=1))]
    return close.asfreq("D").ffill()


class _Grid:
    """72 start dates x 24 closing hours on Binance 1-hour candles, textbook and ordinary person."""

    def __init__(self, price: np.ndarray, b0: pd.Timestamp, pre: pd.Series):
        self.price, self.b0, self.pre = price, b0, pre
        self.m = cr._Market(price, b0, pre.values)
        self.eng = rt._Engine(self.m)
        self.days, self.npre = self.m.days, len(pre)
        self.starts, self.labels = msc._month_starts(self.m.dates, CLOCK_STARTS[0], CLOCK_STARTS[1])
        hr, hd = msc._suffix(msc._equity(self.m.change), self.starts, YEAR)
        self.hold = (hr[:, 0], hd[:, 0])
        self.closes = [np.concatenate([pre.values, price[h::24]]) for h in range(24)]
        self.ma = [_ma50(c) for c in self.closes]

    def ages(self, h: int, origin: pd.Timestamp) -> np.ndarray:
        n = len(self.closes[h]) - self.npre
        start = (self.b0 + pd.Timedelta(hours=h) - origin) / pd.Timedelta(days=1)
        return np.concatenate([_ages(self.pre.index, origin), start + np.arange(n)])

    def dates(self, h: int) -> pd.DatetimeIndex:
        n = len(self.closes[h]) - self.npre
        return self.pre.index.append(self.b0 + pd.Timedelta(hours=h) + pd.to_timedelta(np.arange(n), unit="D"))

    def run(self, origin: pd.Timestamp, keys, min_age: float) -> dict:
        ns = len(self.starts)
        text = {k: (np.zeros((ns, 24)), np.zeros((ns, 24))) for k in keys}
        person = {k: (np.zeros((ns, 24, SIMS)), np.zeros((ns, 24, SIMS))) for k in keys}
        last_z = None
        for h in range(24):
            z = _pit(self.ages(h, origin), self.closes[h], self.dates(h), FIT_FROM, min_age)["z"]
            if h == 0:
                last_z = z
            target = _rules(z, self.ma[h])
            for k in keys:
                w = np.nan_to_num(target[k][self.npre:][:self.days].astype(float), nan=0.0)
                r, dd = msc._suffix(np.concatenate([[1.0], self.eng._mid(h, w, COST, None)]), self.starts, YEAR)
                text[k][0][:, h], text[k][1][:, h] = r[:, 0], dd[:, 0]
                b, wl = rt._later(h, w, self.days)
                rng = np.random.default_rng(SEED + 1000 * RULES.index(k) + h)
                pos = rt._missed(wl, P_MISS, rng, SIMS)
                mid = self.eng._mid(b, pos, P_COST, None)
                r, dd = msc._suffix(np.vstack([np.ones((1, SIMS)), mid]), self.starts, YEAR)
                person[k][0][:, h, :], person[k][1][:, h, :] = r, dd
        return {"text": text, "person": person, "z0": last_z}


def _score(daily: dict, grid: _Grid, clock: dict, k: str) -> dict:
    dr, dd = daily[k]
    hr, hd = grid.hold
    tr, td = clock["text"][k]
    pr, _ = clock["person"][k]
    edge_d = dr - daily["hold"][0]
    edge = tr - hr[:, None]
    pedge = pr - hr[:, None, None]
    x = {"d_n": len(dr), "d_beat": float((edge_d > 0).mean()), "d_edge": float(np.median(edge_d)),
         "c_n": edge.size, "c_beat": float((edge > 0).mean()), "c_edge": float(np.median(edge)),
         "c_p10": float(np.percentile(edge, 10)), "p_n": pedge.size, "p_beat": float((pedge > 0).mean()),
         "p_edge": float(np.median(pedge)), "deep_med": float(np.median(td)), "deep_worst": float(td.min()),
         "c_shallow": float((td > hd[:, None]).mean()), "d_deep_med": float(np.median(dd))}
    x["score"] = min(x["d_beat"], x["c_beat"], x["p_beat"])
    return x


def _passes(x: dict, base: dict) -> bool:
    return x["score"] >= base["score"] and x["deep_med"] > base["deep_med"]


# ---------- descriptive parts ----------

def _forward(close: pd.Series, z: np.ndarray) -> list[list]:
    fwd = (close.shift(-YEAR) / close - 1).values
    ok = np.isfinite(z) & np.isfinite(fwd)
    rows = []
    for label, lo, hi in BUCKETS:
        sel = ok & (z >= lo) & (z < hi)
        f = fwd[sel] * 100
        if sel.sum() == 0:
            rows.append([label, "0 天", "—", "—", "—", "—", "—"])
            continue
        years = sorted({d.year for d in close.index[sel]})
        rows.append([label, f"{int(sel.sum()):,} 天", f"{len(years)} 個（{years[0]}–{years[-1]}）",
                     _pct(float(np.median(f))), _share(float((f > 0).mean())),
                     _pct(float(np.percentile(f, 10))), _pct(float(np.percentile(f, 90)))])
    return rows


def _year_ends(close: pd.Series, fit: dict) -> list[dict]:
    """The fit through each year's last close (= the fit used on the first day of the next year)."""
    out = []
    idx = close.index
    for year in range(2013, idx[-1].year):
        i = int(idx.searchsorted(pd.Timestamp(f"{year + 1}-01-01")))
        if i >= len(idx) or not _ok(fit["b"][i]):
            continue
        out.append({"label": f"{year} 年底", "i": i - 1, "a": float(fit["a"][i]), "b": float(fit["b"][i]),
                    "sd": float(fit["sd"][i])})
    return out


def _fair(a: float, b: float, age: float) -> float:
    return 10 ** (a + b * math.log10(age))


def _eth(problems: list) -> dict | None:
    try:
        close = ta._crypto_close("ETHUSDT")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"ETH 這次取不到資料（{type(exc).__name__}），探索性檢查沒有做。")
        return None
    try:
        fit_from = close.index[0] + pd.DateOffset(years=3)
        three = pd.Timestamp(common.today_utc()) - pd.DateOffset(years=3)
        d = _daily(close, ETH_GENESIS, fit_from, three, fit_from, 1.0)
        n = len(d["starts"])
        if not n:
            raise RuntimeError("沒有起算日")
        out = {"n": n, "first": d["starts"][0], "last": d["starts"][-1], "since": close.index[0]}
        for k in RULES:
            out[k] = (int((d[k][0] > d["hold"][0]).sum()), int((d[k][1] > d["hold"][1]).sum()))
        fit = _pit(_ages(close.index, ETH_GENESIS), close.values, close.index, fit_from)
        out["b_now"] = float(fit["b"][-1])
        return out
    except Exception as exc:  # noqa: BLE001
        problems.append(f"ETH 探索性檢查算不出來（{type(exc).__name__}: {exc}）。")
        return None


# ---------- the whole computation ----------

def _compute() -> dict:
    close = te._daily_close()
    problems: list[str] = []
    rows, problem = cr._hourly("BTCUSDT")
    if problem:
        problems.append(problem)
    if len(rows) < cr.MIN_HOURS:
        raise RuntimeError(f"幣安 BTCUSDT 1 小時線只有 {len(rows)} 筆")
    price, b0, _ = cr._grid(rows)
    pre = _pre(b0)
    if len(pre) < 400:
        raise RuntimeError("拿不到 2017-08 以前的日線來暖機")
    grid = _Grid(price, b0, pre)
    origins = {}
    base_fit = _pit(_ages(close.index, GENESIS), close.values, close.index, FIT_FROM)
    for shift in SHIFTS:
        origin = GENESIS + pd.DateOffset(years=shift)
        min_age = 1.0 if shift == 0 else MIN_AGE
        keys = RULES if shift == 0 else PL_RULES
        fit = base_fit if shift == 0 else _pit(_ages(close.index, origin), close.values, close.index, FIT_FROM, min_age)
        daily = _daily(close, origin, DAILY_STARTS[0], DAILY_STARTS[1], FIT_FROM, min_age, keys, z=fit["z"])
        clock = grid.run(origin, keys, min_age)
        origins[shift] = {"origin": origin, "fit": fit, "daily": daily, "clock": clock,
                          "s": {k: _score(daily, grid, clock, k) for k in keys}}
    base = origins[0]["s"]["ma50"]
    for shift in SHIFTS:
        origins[shift]["s"]["ma50"] = base
    eth = _eth(problems)
    return {"close": close, "grid": grid, "origins": origins, "eth": eth, "problems": problems}


# ---------- the study ----------

def study() -> dict:
    r = _compute()
    close, grid, origins = r["close"], r["grid"], r["origins"]
    o0 = origins[0]
    s, daily, fit = o0["s"], o0["daily"], o0["fit"]
    base = s["ma50"]
    hold_r, hold_d = grid.hold
    d_hold_r = daily["hold"][0]
    last_day = close.index[-1]
    n_d, n_c = len(daily["starts"]), grid.hold[0].size * 24

    ok_at = {k: [_passes(origins[sh]["s"][k], base) for sh in SHIFTS] for k in PL_RULES}
    passed = [k for k in PL_RULES if all(ok_at[k])]

    # ----- table 1: fair grid -----
    rows1 = [[NAMES["hold"], "—", "—", "—", f"{_pct(float(np.median(hold_d)))}｜{_pct(float(hold_d.min()))}", "100%", "—",
              "—"]]
    for k in RULES:
        x = s[k]
        if k == "ma50":
            verdict_cell = "對照"
        else:
            parts = ["分數" + ("≥" if x["score"] >= base["score"] else "<") + "均線",
                     "虧損" + ("較淺" if x["deep_med"] > base["deep_med"] else "沒有較淺"),
                     f"起點換 7 種 {sum(ok_at[k])} 種過"]
            verdict_cell = ("通過｜" if k in passed else "不通過｜") + "，".join(parts)
        rows1.append([NAMES[k], f"{_share(x['d_beat'])}｜{_pp(x['d_edge'])}",
                      f"{_share(x['c_beat'])}｜{_pp(x['c_edge'])}（{_pp(x['c_p10'])}）", _share(x["p_beat"]),
                      f"{_pct(x['deep_med'])}｜{_pct(x['deep_worst'])}", _share(daily[f"{k}_share"]),
                      _share(x["score"]), verdict_cell])

    # ----- table 2: forward returns by z -----
    rows2 = _forward(close, fit["z"])

    # ----- table 3: the slope over time -----
    age_now = float(_ages(close.index[-1:], GENESIS)[0])
    rows3 = []
    for e in _year_ends(close, fit):
        i = e["i"]
        age_i = float(_ages(close.index[i:i + 1], GENESIS)[0])
        fair_i = _fair(e["a"], e["b"], age_i)
        z_i = (math.log10(float(close.iloc[i])) - math.log10(fair_i)) / e["sd"]
        rows3.append([e["label"], f"{e['b']:.2f}", f"{e['a']:.2f}", f"{_money(fair_i)}（收盤 {_money(float(close.iloc[i]))}）",
                      f"±{e['sd']:.3f}（約 {10 ** e['sd']:.2f} 倍）", f"{z_i:+.2f}",
                      _money(_fair(e["a"], e["b"], age_now))])
    a_now, b_now, sd_now = float(fit["a"][-1]), float(fit["b"][-1]), float(fit["sd"][-1])
    fair_now = _fair(a_now, b_now, age_now)
    z_now = float(fit["z"][-1])
    rows3.append([f"今天（用到 {_d(close.index[-2])} 為止）", f"{b_now:.2f}", f"{a_now:.2f}",
                  f"{_money(fair_now)}（收盤 {_money(float(close.iloc[-1]))}）", f"±{sd_now:.3f}（約 {10 ** sd_now:.2f} 倍）",
                  f"{z_now:+.2f}", _money(fair_now)])
    pub_fair = _fair(PUBLISHED_A, PUBLISHED_B, age_now)
    rows3.append(["常被引用的公開常數（b = 5.84、a = −17.01）", f"{PUBLISHED_B:.2f}", f"{PUBLISHED_A:.2f}", "—", "—",
                  f"{(math.log10(float(close.iloc[-1])) - math.log10(pub_fair)) / sd_now:+.2f}（用今天的標準差）",
                  _money(pub_fair)])
    b_years = [float(x[1]) for x in rows3[:-2]]

    # ----- table 4: placebo -----
    rows4 = []
    for sh in SHIFTS:
        o = origins[sh]
        f = o["fit"]
        label = ("2009-01-03（原本的創世區塊）" if sh == 0 else
                 f"{_d(o['origin'])}（{'晚' if sh > 0 else '早'} {abs(sh)} 年）")
        cells = [f"{_share(o['s'][k]['score'])}｜{_pct(o['s'][k]['deep_med'])}" for k in PL_RULES]
        ok = [SHORT[k].split(" ")[0] for k in PL_RULES if _passes(o["s"][k], base)]
        rows4.append([label, f"{float(f['b'][-1]):.2f}", f"{float(f['z'][-1]):+.2f}"] + cells + ["、".join(ok) or "沒有"])

    # ----- now -----
    t = {k: v[-1] for k, v in daily["target"].items()}
    last = float(close.iloc[-1])

    def word(v) -> str:
        if not _ok(v):
            return "—"
        return "全部持有" if v >= 0.999 else "全部現金" if v <= 0.001 else f"持有 {v * 100:.0f}%"

    band = lambda k: _money(fair_now * 10 ** (k * sd_now))  # noqa: E731
    now = [
        {"label": f"最後一根收完的日線（{_d(last_day)}，UTC 0 點收盤）", "value": f"比特幣 {last:,.0f} 美元"},
        {"label": "當時能畫出的冪次法則合理價（用到前一天為止的收盤）",
         "value": f"{_money(fair_now)}（斜率 b = {b_now:.2f}，收盤是合理價的 {last / fair_now:.2f} 倍）"},
        {"label": "上下緣（合理價 × 10 的 ±1、±2 個標準差次方）",
         "value": f"−2：{band(-2)}｜−1：{band(-1)}｜+1：{band(1)}｜+2：{band(2)}"},
        {"label": "偏離 z（離合理價幾個標準差）", "value": f"{z_now:+.2f}"},
        {"label": SHORT["ma50"], "value": word(t["ma50"])},
        {"label": SHORT["p1"], "value": word(t["p1"])},
        {"label": SHORT["p2"], "value": word(t["p2"])},
        {"label": SHORT["p3"], "value": word(t["p3"])},
        {"label": SHORT["p4"], "value": word(t["p4"])},
        {"label": "對照：用常被引用的公開常數（b = 5.84）畫的合理價", "value": _money(pub_fair)},
        {"label": "這幾格的意思", "value": "只描述規則在最後一根日線收盤後停在哪個狀態，不是買賣建議"},
    ]

    # ----- verdict -----
    cheap, rich = rows2[0], rows2[-1]
    tg = daily["target"]
    from_i = int(close.index.searchsorted(pd.Timestamp(DAILY_STARTS[0])))
    p3_diff = int(np.sum((np.nan_to_num(tg["p3"], nan=-1.0) != np.nan_to_num(tg["ma50"], nan=-1.0))[from_i:]))
    b_lo, b_hi = min(b_years), max(b_years)
    rich = np.nan_to_num(fit["z"], nan=0.0) > 2.0
    rich_days, rich_years = int(rich.sum()), [str(y) for y in sorted({d.year for d in close.index[rich]})]
    b_all = float(np.polyfit(np.log10(_ages(close.index, GENESIS) - 1.0), np.log10(close.values), 1)[0])  # classic.py
    first = (f"只用當時以前的資料畫線，比特幣比趨勢線便宜一個標準差以上時，一年後漲跌中位數 {cheap[3]}（{cheap[4]} 上漲），"
             f"貴兩個標準差以上時 {rich[3]}（{rich[4]} 上漲），但這些日子只來自少數幾個年份。")
    if passed:
        second = (f"公平檢驗（{n_c:,} 格、一般人照做）下，{'、'.join(SHORT[k] for k in passed)}的分數不輸 50 日均線、虧損也較淺；"
                  f"50 日均線分數 {_share(base['score'])}、最深虧損中位數 {_pct(base['deep_med'])}"
                  + (f"（P3 和均線只有 {p3_diff} 天不同）" if "p3" in passed else "") + "。")
    else:
        second = (f"照它加減碼或和 50 日均線合用，四條規則沒有一條通過事先定的檢驗（50 日均線分數 {_share(base['score'])}、"
                  f"最深虧損中位數 {_pct(base['deep_med'])}）：P1 虧損較淺（{_pct(s['p1']['deep_med'])}）但報酬只有 "
                  f"{_share(s['p1']['score'])} 贏過一直抱著；P3 和均線只有 {p3_diff} 天不同，幾乎打平；"
                  f"P4 分數 {_share(s['p4']['score'])}、虧損 {_pct(s['p4']['deep_med'])}。")
    changed = [SHORT[k].split(" ")[0] for k in PL_RULES if len(set(ok_at[k])) > 1]
    third = (f"2013 年底以來每年年底擬合的斜率在 {b_lo:.2f} 到 {b_hi:.2f} 之間，今天 {b_now:.2f}；起點前後移 1–3 年，"
             f"斜率變成 {min(float(origins[sh]['fit']['b'][-1]) for sh in SHIFTS):.1f} 到 "
             f"{max(float(origins[sh]['fit']['b'][-1]) for sh in SHIFTS):.1f}，"
             + ("通過與否不變。" if not changed else f"{'、'.join(changed)} 過不過關會跟著起點改變。"))
    verdict = first + second + third

    # ----- notes -----
    eth = r["eth"]
    notes = [
        "事先登記的設計（看結果之前寫好，之後沒有改）：每天只用前一天以前的所有日收盤，以最小平方法擬合 log10(價格) = a + b × log10(距 2009-01-03 的天數)，"
        "2014-01-01 起（已有 3 年以上價格）才開始用；z = （今天的對數價格 − 擬合值）÷ 過去殘差的標準差。規則在收盤決定、隔天照做，成本 0.10%／0.30%。"
        "P1：持有 0.5 − 0.25×z（夾在 0 到 1，低於趨勢兩個標準差全部持有、高於兩個標準差全部現金、剛好在線上持有一半）；"
        "P2：z < +1 持有、z > +2 換現金、中間維持原狀；P3：P2 而且 50 日均線也說持有；P4：z < 0 一律持有，否則照 50 日均線。"
        "通過條件：排名分數不低於 50 日均線、最深虧損中位數比 50 日均線淺、而且改變時間起點不改變結論。",
        "冪次法則就是一條隨時間越走越平的長期趨勢線：在「對數價格 vs 對數天數」的圖上是直線，在一般圖上越走越平。"
        "上下緣這裡用的是 ±1、±2 個標準差，不是網路上常見的「支撐線／頂部線」那種另外擬合的線。",
        f"排名分數 = 三項「報酬贏過一直抱著」比例裡最低的一項：比特幣 {n_d} 個起算日（{DAILY_STARTS[0][:7]}–{DAILY_STARTS[1][:7]}，UTC 0 點日線，照課本做 0.10%）、"
        f"{n_c:,} 格起算日×收盤時間（{CLOCK_STARTS[0][:7]}–{CLOCK_STARTS[1][:7]}，照課本做 0.10%）、同樣 {n_c:,} 格照一般人方式做"
        f"（收盤後 10 小時才做、每次 {P_COST * 100:.2f}%、{P_MISS * 100:.0f}% 的進場漏掉、每個收盤時間模擬 {SIMS} 次）。"
        "main_signal_check 還有第四項「其他資產」，但沒有別的資產有夠長的價格歷史可以畫冪次法則，所以這裡每條規則（含 50 日均線）都只用比特幣三項；"
        f"因此 50 日均線的分數 {_share(base['score'])} 和 main_signal_check 頁面上的數字不同，一般人做法的成本也從 0.25% 改成事先定的 0.30%。"
        "最深虧損中位數取照課本做的那一組格子。",
        "儀表板「週期」區顯示的冪律模型倍數和彩虹圖，是用整段歷史一次擬合的（classic.py 自己也寫明是事後擬合）：今天那條線和這裡「今天」的線幾乎一樣，"
        "但拿它回頭看過去哪裡便宜、哪裡貴，用到了當時還不知道的資料。AHR999 用的是 2018 年定下的公開常數（b = 5.84、a = −17.01），"
        f"今天的合理價 {_money(pub_fair)}，比這裡只用過去資料擬合的 {_money(fair_now)}{'高' if pub_fair > fair_now else '低'}；"
        f"照 classic.py 的做法用全部資料一次擬合，今天的斜率是 {b_all:.2f}；這裡每年年底只用當時資料畫的斜率在 {b_lo:.2f} 到 "
        f"{b_hi:.2f} 之間跳動（見斜率表）。",
        "時間起點安慰劑：把「第 0 天」從 2009-01-03 往前或往後移 1、2、3 年重新擬合（往後移時，起點後未滿 180 天的價格不放進擬合），"
        "四條規則再跑一次同樣三項檢查；50 日均線不受影響，所以基準不變。斜率 b 會隨起點大幅改變，這正是批評者說的「冪次」不是唯一的。",
        f"一年後漲跌那張表：z 是每天當時算得出的值，從 2014-01-01 到最後一個有一年後價格的日子；相鄰日子的一年後報酬幾乎重疊，"
        "所以「幾天」不是獨立的次數，真正獨立的大概只有「涵蓋幾個年份」那麼少；z 很低或很高的日子集中在幾次大跌底部和泡沫頂部。",
        "arXiv 2605.21316（Baquero 與 Menezes，2026）：用 2010-08 到 2026-03 的日價、2014–2024 年 11 個切點做逐年往前的樣本外預測，"
        "1–3 個月直接用今天價格（不變）預測比冪次法則準，6–12 個月差不多，18–24 個月冪次法則比所有對手（多段 S 曲線、樣條、ARIMA 等）準；"
        "但指數會隨時間起點大幅改變（起點不移 5.65、移 240 天 6.28、移 1,000 天 8.09），樣本內三段 S 曲線擬合得更好，"
        "作者的結論是它描述了穩定的長期平均斜率、是不錯的長期點預測，但不是唯一的「結構定律」；論文沒有測試買賣規則。",
        "批評（〈Description Without Prediction〉，Zenodo）認為冪次法則只是事後描述、沒有可檢驗的預測力；辯護（Quantodian，Medium）認為長期斜率穩定、"
        "批評用的檢驗不對。這裡沒有替任何一方背書，只照事先定的規則檢查。",
        f"P3 為什麼和 50 日均線幾乎一樣：P2 幾乎一直是「持有」（{DAILY_STARTS[0][:4]} 年起平均持有 {_share(daily['p2_share'])}），"
        f"z 超過 +2 的日子只出現在 {'、'.join(rich_years) or '—'} 年（共 {rich_days} 天），所以 P3 從 {DAILY_STARTS[0][:4]} 年起只有 {p3_diff} 天和 50 日均線不同；"
        "它的分數、虧損和均線差距都在小數點後面，換一個時間起點就可能從「打平稍好」變成「打平」，這種差別不能當成證據。各規則在哪些起點兩項都不輸均線："
        + "；".join(f"{SHORT[k]} " + ("、".join(("原起點" if sh == 0 else f"{sh:+d} 年") for sh, ok in zip(SHIFTS, ok_at[k]) if ok) or "沒有")
                   for k in PL_RULES) + "。",
        "七種時間起點、四條規則、三項檢查是一起看的；相鄰的起算日共用大部分歷史、24 個收盤時間只差幾個小時，所以「幾成贏」不是互相獨立的證據。"
        "2015 年以後比特幣只經歷約三次大漲大跌，任何根據「偏離趨勢線多少」的規則能檢驗的轉折都很少。",
        "成本很粗略：照課本做 0.10%、一般人做法 0.30%，P1 每天微調持有比例也照實扣成本；現金沒有利息、沒有算稅；一直抱著不扣成本。",
        "過去的結果不保證未來會一樣。這裡只是比較幾條固定規則過去在不同條件下做了什麼、現在停在什麼狀態，不是買賣建議。",
    ]
    if eth:
        cells = "；".join(f"{SHORT[k]} 報酬贏 {eth[k][0]} 個、虧損較淺 {eth[k][1]} 個" for k in RULES)
        notes.insert(5, f"探索性檢查（不算進通過條件）：除了比特幣，沒有其他資產有夠長、可比的價格歷史。ETH 以 2015-07-30 為起點，幣安日線 {_d(eth['since'])} 才有，"
                        f"所以 {_d(eth['since'] + pd.DateOffset(years=3))} 才開始用擬合；從 {_d(eth['first'])[:7]} 到 {_d(eth['last'])[:7]} 每月起算共 {eth['n']} 個，"
                        f"照課本做：{cells}。今天 ETH 擬合的斜率 {eth['b_now']:.2f}。")
    notes += r["problems"]

    tables = [
        {"caption": (f"公平檢驗（看結果之前就定好）。第二欄：{n_d} 個起算日（每月 1 日，{DAILY_STARTS[0][:7]}–{DAILY_STARTS[1][:7]}）報酬贏過同一天起一直抱著的比例｜"
                     f"每年多幾個百分點的中位數。第三欄：{n_c:,} 格起算日×收盤時間，括號是最差一成的格子。第四欄：同樣格子照一般人方式做（各 {SIMS} 次模擬）。"
                     f"第五欄是 {n_c:,} 格照課本做的最深虧損。第六欄是 {DAILY_STARTS[0][:4]} 年起平均持有幾成。分數 = 第二、三、四欄比例裡最低的。"
                     f"算到 {_d(last_day)}。"),
         "columns": ["規則", f"{n_d} 個起算日：報酬贏｜差距中位數", f"{n_c:,} 格：報酬贏｜差距中位數（最差一成）", "一般人照做：報酬贏",
                     "最深虧損：中位數｜最深一格", "平均持有", "分數（三項最低）", "通過嗎"],
         "rows": rows1},
        {"caption": ("比特幣每天當時算得出的偏離 z（只用前一天以前的價格擬合），之後一年的漲跌（2014-01-01 起）。"
                     "相鄰的日子幾乎是同一段一年，所以天數不是獨立的次數。"),
         "columns": ["偏離 z", "天數", "涵蓋幾個年份", "一年後漲跌中位數", "一年後上漲的比例", "最差一成", "最好一成"],
         "rows": rows2},
        {"caption": ("斜率 b 穩不穩：每年最後一天收盤後，只用到那天為止的價格擬合的線。最後一欄是把那條線延伸到今天的合理價，"
                     "可以看出同一條「法則」幾年前畫出來和今天差多少。標準差是殘差（對數價格）的標準差，倍數是 10 的那個次方。"),
         "columns": ["擬合用到", "斜率 b", "截距 a", "當時的合理價（當天收盤）", "殘差標準差", "當天 z", "這條線延伸到今天的合理價"],
         "rows": rows3},
        {"caption": (f"時間起點安慰劑：「第 0 天」換一天重新擬合，四條規則再跑同樣三項檢查。每格是「分數｜最深虧損中位數」；"
                     f"50 日均線不受影響，分數 {_share(base['score'])}、最深虧損中位數 {_pct(base['deep_med'])}。"
                     "最後一欄是分數不低於、虧損也比 50 日均線淺的規則。"),
         "columns": ["時間起點", "今天的斜率 b", "今天的 z", SHORT["p1"], SHORT["p2"], SHORT["p3"], SHORT["p4"], "兩項都不輸均線的"],
         "rows": rows4},
    ]

    return {
        "id": "power_law",
        "title": "冪次法則趨勢線：只用當時的資料，能拿來加減碼嗎",
        "question": ("比特幣的冪次法則（一條隨時間越走越平的長期趨勢線）如果只用當時以前的資料畫，價格偏貴或偏便宜之後一年怎麼走？"
                     "照它加減碼、或和 50 日均線一起用，在公平檢驗下有沒有贏過 50 日均線？斜率和時間起點改一下，結論會不會變？"),
        "asof": _d(last_day),
        "verdict": verdict,
        "now": now,
        "tables": tables,
        "notes": notes,
        "sources": [
            {"name": "Decrypt：The Bitcoin Power Law（Santostasi 的冪次法則介紹）",
             "url": "https://decrypt.co/resources/the-bitcoin-power-law-a-mathematical-approach-to-predicting-bitcoins-price"},
            {"name": "Baquero & Menezes (2026) Bitcoin's Power Law: Weak Structure, Strong Forecasts, arXiv 2605.21316",
             "url": "https://arxiv.org/html/2605.21316"},
            {"name": "Description Without Prediction（批評，Zenodo；未逐頁核對）", "url": "https://zenodo.org/records/19975701"},
            {"name": "Quantodian：Bitcoin's Power Law Really Debunked?（辯護，Medium；未逐頁核對）",
             "url": "https://medium.com/quantodian-publications/bitcoins-power-law-really-debunked-2e5add103ba9"},
            {"name": "blockchain.info 每日比特幣價格（2017 年以前）", "url": "https://www.blockchain.com/explorer/charts/market-price"},
            {"name": "幣安現貨日線與 1 小時線（BTCUSDT、ETHUSDT）", "url": "https://www.binance.com/zh-TC/trade/BTC_USDT"},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2, allow_nan=False))
